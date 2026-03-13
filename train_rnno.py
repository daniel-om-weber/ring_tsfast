# %% Imports + Config
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import qmt
import torch
import torch.nn as nn
import torch_optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split

from tsfast.models.rnn import SimpleRNN
from tsfast.quaternions.ops import inclinationAngle, norm_quaternion, relativeAngle
from tsfast.training.learner import TbpttLearner
from tsfast.tsdata.pipeline import DataLoaders

PATH_LAM4 = "path/to/lam4"  # <-- set this
BS, EPISODES, LR, TBP = 4, 1000, 1e-3, 1000
N_VAL, RNN_W, RNN_D, LIN_D = 32, 200, 2, 0
CELLTYPE, LAYERNORM, SEED = "gru", False, 1
DROP_JA_1D, DROP_JA_2D, DROP_DOF = 1.0, 1.0, 0.0

np.random.seed(SEED)
torch.manual_seed(SEED)


# %% FeatureConfig + Transform
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


class Transform:
    chain = ["seg2_4Seg", "seg3_4Seg", "seg4_4Seg", "seg5_4Seg"]
    lam = [-1, 0]
    link_names = ["seg3_2Seg", "seg4_2Seg"]

    def __init__(self, imtp, drop_imu_1d, drop_imu_2d, drop_imu_3d,
                 drop_ja_1d, drop_ja_2d, drop_dof, rand_ori):
        self.imtp = imtp
        self.drop_imu = {1: drop_imu_1d, 2: drop_imu_2d, 3: drop_imu_3d}
        self.drop_ja_1d, self.drop_ja_2d, self.drop_dof = drop_ja_1d, drop_ja_2d, drop_dof
        self.rand_ori = rand_ori

    def _lamX_from_lam4(self, lam4, rename_to):
        N = len(rename_to)
        start = np.random.choice(list(range(5 - N)))
        rename_from = self.chain[start:(start + N)]
        X, y = lam4
        for old, new in zip(rename_from, rename_to):
            X[new], y[new] = X[old], y[old]
        for old in self.chain:
            X.pop(old); y.pop(old)
        if not self.rand_ori:
            return X, y
        for name in y:
            qrand = qmt.randomQuat()
            X[name]["acc"] = qmt.rotate(qrand, X[name]["acc"])
            X[name]["gyr"] = qmt.rotate(qrand, X[name]["gyr"])
            y[name] = qmt.qmult(y[name], qmt.qinv(qrand))
        return X, y

    def __call__(self, lam41, lam42, lam43, lam44):
        imtp, slices = self.imtp, self.imtp.getSlices()
        X1, Y1 = self._lamX_from_lam4(lam42, ["seg3_2Seg", "seg4_2Seg"])
        dt = X1.pop("dt")

        T = Y1["seg3_2Seg"].shape[0]
        X, Y = np.zeros((imtp.getF(), 2, T)), np.zeros((2, T, 4))

        if imtp.dt:
            X[slices["dt"], :] = dt / imtp.scale_dt

        draw = lambda p: 1.0 - np.random.binomial(1, p=p)

        for i, (name, p) in enumerate(zip(self.link_names, self.lam)):
            X[slices["acc"], i] = X1[name]["acc"].T / imtp.scale_acc
            X[slices["gyr"], i] = X1[name]["gyr"].T / imtp.scale_gyr

            if p != -1:
                dof = int(X1[name]["dof"])
                if imtp.joint_axes_1d and dof == 1:
                    X[slices["ja_1d"], i] = (
                        X1[name]["joint_params"]["rr"]["joint_axes"][:, None]
                        / imtp.scale_ja * draw(self.drop_ja_1d))
                if imtp.joint_axes_2d and dof == 2:
                    X[slices["ja_2d"], i] = (
                        X1[name]["joint_params"]["rsaddle"]["joint_axes"].reshape(6, 1)
                        / imtp.scale_ja * draw(self.drop_ja_2d))
                if imtp.dof:
                    dof_array = np.zeros((3,))
                    dof_array[dof - 1] = 1.0 * draw(self.drop_dof)
                    X[slices["dof"], i] = dof_array[:, None]

            q_p = np.array([1.0, 0, 0, 0]) if p == -1 else Y1[self.link_names[p]]
            Y[i] = qmt.qrel(q_p, Y1[name])

        X, Y = X.transpose((2, 1, 0)), Y.transpose((1, 0, 2))
        return X.astype(np.float32), Y.astype(np.float32)


# %% Dataset classes
class FolderOfFilesDataset(Dataset):
    def __init__(self, path):
        self.files = sorted(str(p) for p in Path(path).iterdir() if p.suffix == ".pickle")
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            return pickle.load(f)


class ShuffledDataset(Dataset):
    def __init__(self, ds):
        self.ds, self.idx = ds, np.random.permutation(len(ds))
    def __len__(self): return len(self.ds)
    def __getitem__(self, i): return self.ds[self.idx[i]]


class MultiDataset(Dataset):
    def __init__(self, datasets, transform=None):
        self.datasets, self.transform = datasets, transform
    def __len__(self): return min(len(ds) for ds in self.datasets)
    def __getitem__(self, idx):
        items = [ds[idx] for ds in self.datasets]
        return self.transform(*items) if self.transform else tuple(items)


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


class FlattenSegments:
    def __call__(self, xb, yb):
        B, T = xb.shape[:2]
        return xb.reshape(B, T, -1), yb.reshape(B, T, -1)


# %% LR Schedule
def warmup_cosine_schedule_factory(pct_warmup=0.65):
    def schedule_fn(optimizer, total_steps):
        warmup_steps = int(pct_warmup * total_steps)
        def lr_lambda(step):
            if step < warmup_steps:
                return 0.33 + 0.67 * (step / max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda)
    return schedule_fn


# %% Build + Train + Save
lam, n_segments = (-1, 0), 2
imtp = FeatureConfig(joint_axes_1d=DROP_JA_1D != 1, joint_axes_2d=DROP_JA_2D != 1, dof=DROP_DOF != 1)
transform = Transform(imtp, 0.0, 0.0, 0.0, DROP_JA_1D, DROP_JA_2D, DROP_DOF, rand_ori=False)

ds = MultiDataset([ShuffledDataset(FolderOfFilesDataset(PATH_LAM4)) for _ in range(4)], transform)
ds_train, ds_val = random_split(ds, [len(ds) - N_VAL, N_VAL])
dl_train = DataLoader(ds_train, batch_size=BS, shuffle=True, drop_last=True)
dl_val = DataLoader(ds_val, batch_size=min(len(ds_val), BS))
dls = DataLoaders(train=dl_train, valid=dl_val)
n_epoch = max(1, round(EPISODES / len(dl_train)))

model = RNNOModel(imtp.getF() * n_segments, n_segments, RNN_W, RNN_D, LIN_D,
                  CELLTYPE, "layernorm" if LAYERNORM else "")

learner = TbpttLearner(
    model=model, dls=dls, loss_func=rnno_loss_factory(lam, n_segments),
    opt_func=lambda params, lr: torch_optimizer.Lamb(params, lr=lr, weight_decay=0.0),
    transforms=[FlattenSegments()], grad_clip=0.5, sub_seq_len=TBP)

learner.fit(n_epoch=n_epoch, lr=LR,
            scheduler_fn=warmup_cosine_schedule_factory(0.65))

torch.save(model.state_dict(), "rnno_final.pt")
