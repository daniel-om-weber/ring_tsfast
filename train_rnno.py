# %% Imports + Config
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import qmt
import torch
import torch.nn as nn
import torch_optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split

from tsfast.models.rnn import SimpleRNN
from tsfast.quaternions.ops import inclinationAngle, norm_quaternion, relativeAngle
from tsfast.training.learner import TbpttLearner
from tsfast.tsdata.pipeline import DataLoaders
from tsfast.tsdata.readers import HDF5Signals

HDF5_DIR = "ring_data_h5"
BS, EPISODES, LR, TBP = 256, 4800, 1e-3, 1000
N_VAL, RNN_W, RNN_D, LIN_D = 256, 400, 2, 2
CELLTYPE, LAYERNORM, SEED = "gru", True, 1
DROP_JA_1D, DROP_JA_2D, DROP_DOF = 0.5, 1.0, 1.0
BATCHES_PER_EPOCH = 100
NUM_WORKERS = 4

np.random.seed(SEED)
torch.manual_seed(SEED)


# %% FeatureConfig
@dataclass
class FeatureConfig:
    joint_axes_1d: bool = False
    joint_axes_2d: bool = False
    dof: bool = True
    dt: bool = True
    scale_acc: float = 9.81
    scale_gyr: float = 2.2
    scale_dt: float = 0.01
    scale_ja: float = 0.33

    def getSlices(self) -> dict:
        idx = 0
        slices = {}
        for name, size, enabled in [
            ("acc", 3, True), ("gyr", 3, True),
            ("ja_1d", 3, self.joint_axes_1d), ("ja_2d", 6, self.joint_axes_2d),
            ("dof", 3, self.dof), ("dt", 1, self.dt),
        ]:
            if enabled:
                slices[name] = slice(idx, idx + size)
                idx += size
        return slices

    def getF(self) -> int:
        return max(s.stop for s in self.getSlices().values())


# %% Dataset
SEGS = ["seg2", "seg3", "seg4", "seg5"]
PAIRS = [(SEGS[i], SEGS[i + 1]) for i in range(3)]


class RingDataset(Dataset):
    def __init__(self, files, imtp, drop_ja_1d):
        self.files = files
        self.imtp = imtp
        self.drop_ja_1d = drop_ja_1d

        # 3 HDF5Signals readers — one per segment pair, no mmap to avoid VMA pressure
        self._imu = []
        self._q = []
        for s0, s1 in PAIRS:
            self._imu.append(HDF5Signals([
                f'{s0}_acc_x', f'{s0}_acc_y', f'{s0}_acc_z',
                f'{s0}_gyr_x', f'{s0}_gyr_y', f'{s0}_gyr_z',
                f'{s1}_acc_x', f'{s1}_acc_y', f'{s1}_acc_z',
                f'{s1}_gyr_x', f'{s1}_gyr_y', f'{s1}_gyr_z',
            ], use_mmap=False))
            self._q.append(HDF5Signals([
                f'{s0}_q_w', f'{s0}_q_x', f'{s0}_q_y', f'{s0}_q_z',
                f'{s1}_q_w', f'{s1}_q_x', f'{s1}_q_y', f'{s1}_q_z',
            ], use_mmap=False))

        # Lazily cached per-file attrs (populated in __getitem__)
        self._attr_cache = {}

    def _get_attrs(self, idx):
        if idx not in self._attr_cache:
            path = self.files[idx]
            with h5py.File(path, 'r') as f:
                attrs = {'dt': np.float32(f.attrs['dt'])}
                for seg in ['seg3', 'seg4', 'seg5']:
                    attrs[seg] = {
                        'dof': int(f.attrs[f'{seg}_dof']),
                        'ja_rr': np.array(f.attrs[f'{seg}_ja_rr'], dtype=np.float32),
                        'ja_rsaddle': np.array(f.attrs[f'{seg}_ja_rsaddle'], dtype=np.float32),
                    }
            self._attr_cache[idx] = attrs
        return self._attr_cache[idx]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pair_idx = np.random.randint(3)
        path = self.files[idx]
        T = self._imu[pair_idx].file_len(path)

        imu = self._imu[pair_idx].read(path, 0, T)  # (T, 12) via mmap
        q = self._q[pair_idx].read(path, 0, T)       # (T, 8) via mmap

        imtp, slices = self.imtp, self.imtp.getSlices()
        F = imtp.getF()
        X = np.zeros((T, 2, F), dtype=np.float32)
        Y = np.zeros((T, 2, 4), dtype=np.float32)

        # IMU features: imu[:, 0:3]=seg0_acc, imu[:, 3:6]=seg0_gyr,
        #               imu[:, 6:9]=seg1_acc, imu[:, 9:12]=seg1_gyr
        for i, off in enumerate([0, 6]):
            X[:, i, slices['acc']] = imu[:, off:off+3] / imtp.scale_acc
            X[:, i, slices['gyr']] = imu[:, off+3:off+6] / imtp.scale_gyr

        attrs = self._get_attrs(idx)

        if imtp.dt:
            X[:, :, slices['dt']] = attrs['dt'] / imtp.scale_dt

        # ja_1d for child segment (index 1)
        _, child_seg = PAIRS[pair_idx]
        if child_seg in attrs and imtp.joint_axes_1d:
            dof = attrs[child_seg]['dof']
            if dof == 1:
                ja = attrs[child_seg]['ja_rr']
                drop = 1.0 - np.random.binomial(1, p=self.drop_ja_1d)
                X[:, 1, slices['ja_1d']] = ja / imtp.scale_ja * drop

        # Quaternion targets
        q_seg0 = q[:, 0:4]
        q_seg1 = q[:, 4:8]
        Y[:, 0, :] = q_seg0                       # absolute (for inclinationAngle)
        Y[:, 1, :] = qmt.qrel(q_seg0, q_seg1)    # relative (for relativeAngle)

        return X.reshape(T, -1).astype(np.float32), Y.reshape(T, -1).astype(np.float32)


# %% Model + Loss
class RNNOModel(nn.Module):
    def __init__(self, input_size, n_segments, hidden_size, num_layers,
                 linear_layers, rnn_type, normalization):
        super().__init__()
        self.n_segments = n_segments
        self.rnn = SimpleRNN(
            input_size=input_size, output_size=n_segments * 4,
            hidden_size=hidden_size, num_layers=num_layers,
            linear_layers=linear_layers, return_state=True,
            rnn_type=rnn_type, normalization=normalization)

    def forward(self, x, state=None):
        out, new_state = self.rnn(x, state=state)
        B, T, _ = out.shape
        out = norm_quaternion(out.view(B, T, self.n_segments, 4)).view(B, T, -1)
        return out, new_state


def rnno_loss_factory(lam, n_segments):
    def loss_fn(pred, targ):
        pred_q = pred.view(*pred.shape[:-1], n_segments, 4)
        targ_q = targ.view(*targ.shape[:-1], n_segments, 4)
        loss = 0.0
        for i, p in enumerate(lam):
            fn = inclinationAngle if p == -1 else relativeAngle
            loss += fn(pred_q[..., i, :], targ_q[..., i, :]).pow(2).mean()
        return loss / len(lam)
    return loss_fn


def _make_segment_metric(name, fn, n_segments, seg_idx):
    def metric(pred, targ):
        p = pred.view(*pred.shape[:-1], n_segments, 4)
        t = targ.view(*targ.shape[:-1], n_segments, 4)
        return (fn(p[..., seg_idx, :], t[..., seg_idx, :]) * (180 / math.pi)).mean()
    metric.__name__ = name
    return metric


# %% LR Schedule (pure cosine decay, matching ring's optax.cosine_decay_schedule)
def cosine_decay_schedule_factory(pct_decay=0.95, alpha=1e-7):
    def schedule_fn(optimizer, total_steps):
        decay_steps = int(pct_decay * total_steps)
        def lr_lambda(step):
            if step >= decay_steps:
                return alpha
            progress = step / max(1, decay_steps)
            return alpha + (1 - alpha) * 0.5 * (1 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda)
    return schedule_fn


# %% AGC Learner (Adaptive Gradient Clipping, matching ring's optax.adaptive_grad_clip)
class AGCLearner(TbpttLearner):
    def backward_step(self, loss):
        loss.backward()
        for p in self.model.parameters():
            if p.grad is not None:
                w_norm = p.data.norm(2).clamp(min=1e-3)
                g_norm = p.grad.data.norm(2)
                max_norm = 0.5 * w_norm
                if g_norm > max_norm:
                    p.grad.data.mul_(max_norm / g_norm)
        self.opt.step()
        self.opt.zero_grad()

# %% Build + Train + Save
lam, n_segments = (-1, 0), 2
imtp = FeatureConfig(joint_axes_1d=DROP_JA_1D != 1, joint_axes_2d=DROP_JA_2D != 1, dof=DROP_DOF != 1)

files = sorted(str(p) for p in Path(HDF5_DIR).glob('*.h5'))
ds = RingDataset(files, imtp, DROP_JA_1D)
ds_train, ds_val = random_split(ds, [len(ds) - N_VAL, N_VAL])
sampler = RandomSampler(ds_train, replacement=True, num_samples=BATCHES_PER_EPOCH * BS)
dl_train = DataLoader(ds_train, batch_size=BS, sampler=sampler, drop_last=True,
                      num_workers=NUM_WORKERS)
dl_val = DataLoader(ds_val, batch_size=min(len(ds_val), BS),
                    num_workers=NUM_WORKERS)
dls = DataLoaders(train=dl_train, valid=dl_val)
n_epoch = EPISODES // BATCHES_PER_EPOCH

model = RNNOModel(imtp.getF() * n_segments, n_segments, RNN_W, RNN_D, LIN_D,
                  CELLTYPE, "layernorm" if LAYERNORM else "")

from tsfast.models.cudagraph import GraphedStatefulModel
model = GraphedStatefulModel(model)

metrics = [
    _make_segment_metric("seg0_incl", inclinationAngle, n_segments, 0),
    _make_segment_metric("seg1_mae", relativeAngle, n_segments, 1)
]

learner = AGCLearner(
    model=model, dls=dls, loss_func=rnno_loss_factory(lam, n_segments),
    opt_func=lambda params, lr: torch_optimizer.Lamb(params, lr=lr, weight_decay=0.0),
    transforms=[], metrics=metrics, grad_clip=None, sub_seq_len=TBP)

learner.fit(n_epoch=n_epoch, lr=LR,
            scheduler_fn=cosine_decay_schedule_factory(pct_decay=0.95, alpha=1e-7))

torch.save(model.state_dict(), "rnno_final.pt")
