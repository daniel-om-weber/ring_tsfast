# %% Imports + Config
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from tsfast.models.rnn import SimpleRNN
from tsfast.models.scaling import ScaledModel, StandardScaler
from tsfast.quaternions.ops import (
    conjQuat, inclinationAngle, multiplyQuat, norm_quaternion, relativeAngle,
)
from tsfast.training.learner import TbpttLearner
from tsfast.tsdata.dataset import FileEntry, WindowedDataset
from tsfast.tsdata.pipeline import DataLoaders
from tsfast.tsdata.readers import HDF5Attrs, HDF5Signals

HDF5_DIR = "ring_data_h5"
BS, EPISODES, LR, TBP = 256, 4800, 1e-3, 1000
N_VAL, RNN_W, RNN_D, LIN_D = 256, 400, 2, 2
CELLTYPE, LAYERNORM, SEED = "gru", True, 1
DROP_JA_1D, DROP_JA_2D, DROP_DOF = 0.5, 1.0, 1.0
BATCHES_PER_EPOCH = 100
NUM_WORKERS = 2

np.random.seed(SEED)
torch.manual_seed(SEED)


# %% FeatureConfig
@dataclass
class FeatureConfig:
    joint_axes_1d: bool = False
    joint_axes_2d: bool = False
    dof: bool = True
    dt: bool = True

    SCALE_ACC: float = 9.81
    SCALE_GYR: float = 2.2
    SCALE_DT: float = 0.01
    SCALE_JA: float = 0.33

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

    def make_input_scaler(self) -> StandardScaler:
        F = self.getF()
        slices = self.getSlices()
        std = np.ones(2 * F, dtype=np.float32)
        for i in range(2):
            off = i * F
            std[off + slices['acc'].start : off + slices['acc'].stop] = self.SCALE_ACC
            std[off + slices['gyr'].start : off + slices['gyr'].stop] = self.SCALE_GYR
            if 'ja_1d' in slices:
                std[off + slices['ja_1d'].start : off + slices['ja_1d'].stop] = self.SCALE_JA
            if 'dt' in slices:
                std[off + slices['dt'].start : off + slices['dt'].stop] = self.SCALE_DT
        return StandardScaler(np.zeros(2 * F, dtype=np.float32), std)


# %% Dataset (tsdata-native: WindowedDataset + HDF5Signals + HDF5Attrs)
SEGS = ["seg2", "seg3", "seg4", "seg5"]
PAIRS = [(SEGS[i], SEGS[i + 1]) for i in range(3)]


def make_pair_dataset(files, pair_idx):
    """Create a WindowedDataset for one segment pair."""
    s0, s1 = PAIRS[pair_idx]
    imu = HDF5Signals([
        f'{s0}_acc', f'{s0}_gyr', f'{s1}_acc', f'{s1}_gyr',
    ], use_mmap=False)
    quat = HDF5Signals([f'{s0}_q', f'{s1}_q'], use_mmap=False)
    attrs = HDF5Attrs(['dt', f'{s1}_dof', f'{s1}_ja_rr'])
    entries = [FileEntry(path=f) for f in files]
    return WindowedDataset(entries, inputs=(imu, attrs), targets=quat, win_sz=None)


def assemble_batch(imu, attrs, quat, imtp, drop_ja_1d, training):
    """Assemble raw signals into feature/target tensors (no scaling).

    Args:
        imu: (B, T, 12) raw IMU signals
        attrs: (B, 5) flattened [dt, dof, ja_rr_0, ja_rr_1, ja_rr_2]
        quat: (B, T, 8) raw quaternions
        imtp: FeatureConfig
        drop_ja_1d: dropout probability for 1-DOF joint axes
        training: whether to apply stochastic dropout

    Returns:
        xb: (B, T, 2*F) unscaled features
        yb: (B, T, 8) targets [q_seg0, qrel]
    """
    B, T, _ = imu.shape
    slices = imtp.getSlices()
    F = imtp.getF()

    X = torch.zeros(B, T, 2, F, device=imu.device)
    for i, off in enumerate([0, 6]):
        X[:, :, i, slices['acc']] = imu[:, :, off:off+3]
        X[:, :, i, slices['gyr']] = imu[:, :, off+3:off+6]

    # attrs layout: [dt(1), dof(1), ja_rr(3)]
    dt = attrs[:, 0]       # (B,)
    dof = attrs[:, 1]      # (B,)
    ja_rr = attrs[:, 2:5]  # (B, 3)

    if imtp.dt:
        X[:, :, :, slices['dt']] = dt[:, None, None, None]

    if imtp.joint_axes_1d:
        is_1dof = (dof == 1)  # (B,)
        if is_1dof.any():
            if training:
                keep = 1.0 - torch.bernoulli(
                    torch.full((B,), drop_ja_1d, device=imu.device))
            else:
                keep = torch.ones(B, device=imu.device)
            mask = is_1dof.float() * keep  # (B,)
            X[:, :, 1, slices['ja_1d']] = (ja_rr * mask[:, None])[:, None, :]

    # Targets: q_seg0 and qrel = conj(q_seg0) * q_seg1
    q_seg0 = quat[:, :, 0:4]
    q_seg1 = quat[:, :, 4:8]
    Y = torch.zeros(B, T, 2, 4, device=quat.device)
    Y[:, :, 0, :] = q_seg0
    Y[:, :, 1, :] = multiplyQuat(conjQuat(q_seg0), q_seg1)

    return X.reshape(B, T, -1), Y.reshape(B, T, -1)


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
    def __init__(self, *args, imtp, drop_ja_1d, **kwargs):
        super().__init__(*args, **kwargs)
        self.imtp = imtp
        self.drop_ja_1d = drop_ja_1d

    def prepare_batch(self, batch, training=True):
        (imu_raw, attrs_raw), quat_raw = batch
        imu = imu_raw.to(self.device)
        attrs = attrs_raw.to(self.device)
        quat = quat_raw.to(self.device)
        xb, yb = assemble_batch(imu, attrs, quat, self.imtp, self.drop_ja_1d, training)
        for t in self.transforms:
            xb, yb = t(xb, yb)
        if training:
            for a in self.augmentations:
                xb, yb = a(xb, yb)
        return xb, yb

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
if __name__ == "__main__":
    lam, n_segments = (-1, 0), 2
    imtp = FeatureConfig(joint_axes_1d=DROP_JA_1D != 1, joint_axes_2d=DROP_JA_2D != 1, dof=DROP_DOF != 1)

    files = sorted(str(p) for p in Path(HDF5_DIR).glob('*.h5'))
    np.random.shuffle(files)
    train_files, val_files = files[N_VAL:], files[:N_VAL]

    ds_train = ConcatDataset([make_pair_dataset(train_files, p) for p in range(3)])
    ds_val = ConcatDataset([make_pair_dataset(val_files, p) for p in range(3)])

    sampler = RandomSampler(ds_train, replacement=True, num_samples=BATCHES_PER_EPOCH * BS)
    dl_train = DataLoader(ds_train, batch_size=BS, sampler=sampler, drop_last=True,
                          num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
                          pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=min(len(ds_val), BS),
                        num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
                        pin_memory=True)
    dls = DataLoaders(train=dl_train, valid=dl_val)
    n_epoch = EPISODES // BATCHES_PER_EPOCH

    model = RNNOModel(imtp.getF() * n_segments, n_segments, RNN_W, RNN_D, LIN_D,
                      CELLTYPE, "layernorm" if LAYERNORM else "")
    model = ScaledModel(model, imtp.make_input_scaler())

    from tsfast.models.cudagraph import GraphedStatefulModel
    model_graphed = GraphedStatefulModel(model)

    metrics = [
        _make_segment_metric("seg0_incl", inclinationAngle, n_segments, 0),
        _make_segment_metric("seg1_mae", relativeAngle, n_segments, 1)
    ]

    learner = AGCLearner(
        model=model_graphed, dls=dls, loss_func=rnno_loss_factory(lam, n_segments),
        opt_func=lambda params, lr: torch_optimizer.Lamb(params, lr=lr, weight_decay=0.0),
        transforms=[], metrics=metrics, grad_clip=None, sub_seq_len=TBP,
        imtp=imtp, drop_ja_1d=DROP_JA_1D)

    learner.fit(n_epoch=n_epoch, lr=LR,
                scheduler_fn=cosine_decay_schedule_factory(pct_decay=0.95, alpha=1e-7))

    torch.save(model, "rnno_final_v2.pt")
