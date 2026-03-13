"""RNNO training script using tsfast (PyTorch) instead of ring (JAX).

Equivalent to train_step2_trainRing_v2_rnno.py but with no ring dependency at
training time.  Data generation (step 1) still uses ring separately.
"""

import math
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import qmt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split

from tsfast.models.rnn import SimpleRNN
from tsfast.models.state import detach_state
from tsfast.quaternions.ops import (
    inclinationAngle,
    norm_quaternion,
    rad2deg,
    relativeAngle,
)
from tsfast.training.learner import TbpttLearner


# ──────────────────────────────────────────────────────────────────────────────
#  Inline data loading classes (replaces ring.extras.dataloader_torch)
# ──────────────────────────────────────────────────────────────────────────────


class FolderOfFilesDataset(Dataset):
    def __init__(self, path, transform=None):
        self.files = sorted(
            str(p) for p in Path(path).iterdir() if p.suffix == ".pickle"
        )
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            element = pickle.load(f)
        if self.transform is not None:
            element = self.transform(element)
        return element


class ShuffledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffled_indices = np.random.permutation(len(dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[self.shuffled_indices[idx]]


class MultiDataset(Dataset):
    def __init__(self, datasets, transform=None):
        self.datasets = datasets
        self.transform = transform

    def __len__(self):
        return min(len(ds) for ds in self.datasets)

    def __getitem__(self, idx):
        items = [ds[idx] for ds in self.datasets]
        if self.transform:
            return self.transform(*items)
        return tuple(items)


# ──────────────────────────────────────────────────────────────────────────────
#  FeatureConfig (replaces benchmark.IMTP)
# ──────────────────────────────────────────────────────────────────────────────


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
            ("acc", 3, True),
            ("gyr", 3, True),
            ("ja_1d", 3, self.joint_axes_1d),
            ("ja_2d", 6, self.joint_axes_2d),
            ("dof", 3, self.dof),
            ("dt", 1, self.dt),
        ]:
            if enabled:
                slices[name] = slice(idx, idx + size)
                idx += size
        return slices

    def getF(self) -> int:
        return max(s.stop for s in self.getSlices().values())


# ──────────────────────────────────────────────────────────────────────────────
#  Transform (from ring version, adapted for FeatureConfig)
# ──────────────────────────────────────────────────────────────────────────────


class Transform:
    chain = ["seg2_4Seg", "seg3_4Seg", "seg4_4Seg", "seg5_4Seg"]
    lam = [-1, 0]
    link_names = ["seg3_2Seg", "seg4_2Seg"]

    def __init__(
        self,
        imtp: FeatureConfig,
        drop_imu_1d,
        drop_imu_2d,
        drop_imu_3d,
        drop_ja_1d,
        drop_ja_2d,
        drop_dof,
        rand_ori: bool,
    ):
        self.imtp = imtp
        self.drop_imu = {1: drop_imu_1d, 2: drop_imu_2d, 3: drop_imu_3d}
        self.drop_ja_1d = drop_ja_1d
        self.drop_ja_2d = drop_ja_2d
        self.drop_dof = drop_dof
        self.rand_ori = rand_ori

    def _lamX_from_lam4(self, lam4, rename_to: list[str]):
        N = len(rename_to)
        start = np.random.choice(list(range((5 - N))))
        rename_from = self.chain[start : (start + N)]  # noqa: E203
        X, y = lam4
        for old_name, new_name in zip(rename_from, rename_to):
            X[new_name] = X[old_name]
            y[new_name] = y[old_name]
        for old_name in self.chain:
            X.pop(old_name)
            y.pop(old_name)
        return self._maybe_rand_ori(X, y)

    def _maybe_rand_ori(self, X, y):
        if not self.rand_ori:
            return X, y
        for name in y:
            qrand = qmt.randomQuat()
            X[name]["acc"] = qmt.rotate(qrand, X[name]["acc"])
            X[name]["gyr"] = qmt.rotate(qrand, X[name]["gyr"])
            y[name] = qmt.qmult(y[name], qmt.qinv(qrand))
        return X, y

    def __call__(self, lam41, lam42, lam43, lam44):
        imtp = self.imtp
        slices = imtp.getSlices()
        lam = self.lam
        link_names = self.link_names

        X1, Y1 = self._lamX_from_lam4(lam42, ["seg3_2Seg", "seg4_2Seg"])
        dt = X1.pop("dt")

        T = Y1["seg3_2Seg"].shape[0]
        X = np.zeros((imtp.getF(), 2, T))
        Y = np.zeros((2, T, 4))

        if imtp.dt:
            X[slices["dt"], :] = dt / imtp.scale_dt

        draw = lambda p: 1.0 - np.random.binomial(1, p=p)  # noqa: E731

        for i, (name, p) in enumerate(zip(link_names, lam)):
            X[slices["acc"], i] = X1[name]["acc"].T / imtp.scale_acc
            X[slices["gyr"], i] = X1[name]["gyr"].T / imtp.scale_gyr

            if p != -1:
                dof = int(X1[name]["dof"])
                if imtp.joint_axes_1d and dof == 1:
                    X[slices["ja_1d"], i] = (
                        X1[name]["joint_params"]["rr"]["joint_axes"][:, None]
                        / imtp.scale_ja
                        * draw(self.drop_ja_1d)
                    )
                if imtp.joint_axes_2d and dof == 2:
                    X[slices["ja_2d"], i] = (
                        X1[name]["joint_params"]["rsaddle"]["joint_axes"].reshape(
                            6, 1
                        )
                        / imtp.scale_ja
                        * draw(self.drop_ja_2d)
                    )
                if imtp.dof:
                    dof_array = np.zeros((3,))
                    dof_array[dof - 1] = 1.0 * draw(self.drop_dof)
                    X[slices["dof"], i] = dof_array[:, None]

            q_p = np.array([1.0, 0, 0, 0]) if p == -1 else Y1[link_names[p]]
            q_i = Y1[name]
            Y[i] = qmt.qrel(q_p, q_i)

        # (F, N, T) -> (T, N, F) and (N, T, 4) -> (T, N, 4)
        X, Y = X.transpose((2, 1, 0)), Y.transpose((1, 0, 2))
        return X.astype(np.float32), Y.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  FlattenSegments transform for tsfast (B, T, N, F) -> (B, T, N*F)
# ──────────────────────────────────────────────────────────────────────────────


class FlattenSegments:
    def __call__(self, xb, yb):
        B, T = xb.shape[:2]
        return xb.reshape(B, T, -1), yb.reshape(B, T, -1)


# ──────────────────────────────────────────────────────────────────────────────
#  RNNOModel (wraps SimpleRNN with quaternion normalization)
# ──────────────────────────────────────────────────────────────────────────────


class RNNOModel(nn.Module):
    def __init__(
        self,
        input_size,
        n_segments,
        hidden_size,
        num_layers,
        linear_layers,
        rnn_type,
        normalization,
    ):
        super().__init__()
        self.n_segments = n_segments
        self.rnn = SimpleRNN(
            input_size=input_size,
            output_size=n_segments * 4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            linear_layers=linear_layers,
            return_state=True,
            rnn_type=rnn_type,
            normalization=normalization,
        )

    def forward(self, x, state=None):
        out, new_state = self.rnn(x, state=state)
        B, T, _ = out.shape
        out = out.view(B, T, self.n_segments, 4)
        out = norm_quaternion(out)
        out = out.view(B, T, -1)
        return out, new_state


# ──────────────────────────────────────────────────────────────────────────────
#  Loss function
# ──────────────────────────────────────────────────────────────────────────────


def rnno_loss_factory(lam, n_segments):
    def loss_fn(pred, targ):
        pred_q = pred.view(*pred.shape[:-1], n_segments, 4)
        targ_q = targ.view(*targ.shape[:-1], n_segments, 4)
        loss = 0.0
        for i, p in enumerate(lam):
            qi_pred = pred_q[..., i, :]
            qi_targ = targ_q[..., i, :]
            if p == -1:
                loss += inclinationAngle(qi_pred, qi_targ).pow(2).mean()
            else:
                loss += relativeAngle(qi_pred, qi_targ).pow(2).mean()
        return loss / len(lam)

    return loss_fn


# ──────────────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────────────


def _make_metric(name, fn, n_segments):
    def metric(pred, targ):
        p = pred.view(*pred.shape[:-1], n_segments, 4)
        t = targ.view(*targ.shape[:-1], n_segments, 4)
        return rad2deg(fn(p, t).mean())

    metric.__name__ = name
    return metric


# ──────────────────────────────────────────────────────────────────────────────
#  LR schedule: linear warmup + cosine decay
# ──────────────────────────────────────────────────────────────────────────────


def warmup_cosine_schedule_factory(pct_warmup=0.65, n_chunks_per_seq=1):
    """Create a warmup + cosine decay LR schedule.

    The schedule is stepped per TBPTT chunk (not per batch), matching ring's
    behavior.  ``n_chunks_per_seq`` multiplies the episode count from
    ``Learner.fit`` so the total number of LR steps equals
    ``episodes * n_chunks_per_seq``.
    """

    def schedule_fn(optimizer, total_steps_from_learner):
        # Learner passes total_steps = n_epoch * n_batches.
        # With _OneBatchDataLoader, n_batches=1, so total_steps = episodes.
        # Multiply by n_chunks_per_seq to get total LR steps (one per chunk).
        total_steps = total_steps_from_learner * n_chunks_per_seq
        warmup_steps = int(pct_warmup * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return 0.33 + 0.67 * (step / max(1, warmup_steps))
            else:
                progress = (step - warmup_steps) / max(
                    1, total_steps - warmup_steps
                )
                return 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    return schedule_fn


# ──────────────────────────────────────────────────────────────────────────────
#  RNNOLearner (extends TbpttLearner with wandb + checkpointing)
# ──────────────────────────────────────────────────────────────────────────────


class RNNOLearner(TbpttLearner):
    def __init__(
        self,
        *args,
        use_wandb: bool = False,
        save_path: Optional[str] = None,
        checkpoint_every: int = 5,
        checkpoint_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_wandb = use_wandb
        self.save_path = save_path
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self._best_val_loss = float("inf")
        self._sched = None  # set during train_one_epoch

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def train_one_epoch(self, sched=None, pbar=None, epoch=0, n_epoch=1):
        """Override to step LR scheduler per TBPTT chunk (inside training_step)
        rather than per batch, matching ring's behavior."""
        self._sched = sched
        result = super().train_one_epoch(
            sched=None, pbar=pbar, epoch=epoch, n_epoch=n_epoch
        )
        self._sched = None
        return result

    def training_step(self, xb, yb):
        """TBPTT training step that steps the LR scheduler per chunk."""
        xb_chunks = xb.split(self.sub_seq_len, dim=1)
        yb_chunks = yb.split(self.sub_seq_len, dim=1)

        state = None
        losses = []
        for i, (xb_sub, yb_sub) in enumerate(zip(xb_chunks, yb_chunks)):
            skip = self.n_skip if i == 0 else 0

            if state is not None:
                result = self.model(xb_sub, state=state)
            else:
                result = self.model(xb_sub)

            if isinstance(result, tuple):
                pred, new_state = result
            else:
                pred, new_state = result, None

            loss = self.compute_loss(pred, yb_sub, xb_sub, n_skip=skip)

            if torch.isnan(loss):
                self.opt.zero_grad()
                state = None
                continue

            self.backward_step(loss)
            losses.append(loss.item())
            state = detach_state(new_state)

            # Step LR scheduler per TBPTT chunk (matching ring)
            if self._sched is not None:
                self._sched.step()

        if not losses:
            return float("nan")
        return sum(losses) / len(losses)

    def log_epoch(self, epoch, n_epoch, train_loss, val_loss, metrics, pbar):
        super().log_epoch(epoch, n_epoch, train_loss, val_loss, metrics, pbar)

        if self.use_wandb:
            import wandb

            log_dict = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch,
            }
            log_dict.update(metrics)
            if self.opt is not None:
                log_dict["lr"] = self.opt.param_groups[0]["lr"]
            wandb.log(log_dict)

        # Checkpoint periodically
        if (
            self.checkpoint_dir is not None
            and (epoch + 1) % self.checkpoint_every == 0
        ):
            path = os.path.join(self.checkpoint_dir, f"checkpoint_ep{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                path,
            )

        # Save best model
        if self.save_path is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            torch.save(self.model.state_dict(), self.save_path)


# ──────────────────────────────────────────────────────────────────────────────
#  DataLoaders helper
# ──────────────────────────────────────────────────────────────────────────────


class _OneBatchDataLoader:
    """Wraps a DataLoader to yield exactly 1 batch per __iter__ call.

    Cycles through the underlying DataLoader across epochs, matching ring's
    behavior where each 'episode' processes exactly 1 batch.
    """

    def __init__(self, dl: DataLoader):
        self._dl = dl
        self._iter = iter(dl)

    def __iter__(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self._dl)
            batch = next(self._iter)
        return iter([batch])

    def __len__(self):
        return 1


class _SimpleDataLoaders:
    """Minimal DataLoaders container matching the interface TbpttLearner expects."""

    def __init__(self, train, valid):
        self.train = train
        self.valid = valid


# ──────────────────────────────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────────────────────────────


def main(
    path_lam4,
    bs: int,
    episodes: int,
    use_wandb: bool = False,
    wandb_project: str = "RING",
    wandb_name: str = None,
    warmstart: str = None,
    seed: int = 1,
    lr: float = 1e-3,
    tbp: int = 1000,
    drop_imu_1d: float = 0.0,
    drop_imu_2d: float = 0.0,
    drop_imu_3d: float = 0.0,
    drop_ja_1d: float = 1.0,
    drop_ja_2d: float = 1.0,
    drop_dof: float = 0.0,
    n_val: int = 32,
    rnn_w: int = 200,
    rnn_d: int = 2,
    lin_w: int = 200,
    lin_d: int = 0,
    layernorm: bool = False,
    celltype: str = "gru",
    rand_ori: bool = False,
    save_dir: str = "~/params",
    checkpoint_dir: str = "~/ring_checkpoints",
    checkpoint_every: int = 5,
    num_workers: int = 0,
):
    """Train RNNO using tsfast (PyTorch).

    Parameters:
        path_lam4: Path to the dataset folder containing lam4 pickle files.
        bs: Batch size.
        episodes: Number of training epochs.
        use_wandb: Log to Weights & Biases.
        wandb_project: W&B project name.
        wandb_name: W&B run name.
        warmstart: Path to a .pt state dict to warm-start from.
        seed: Random seed.
        lr: Learning rate.
        tbp: Truncated backpropagation sequence length.
        drop_imu_1d: Probability of dropping 1-DOF IMUs.
        drop_imu_2d: Probability of dropping 2-DOF IMUs.
        drop_imu_3d: Probability of dropping 3-DOF IMUs.
        drop_ja_1d: Probability of dropping 1D joint axes.
        drop_ja_2d: Probability of dropping 2D joint axes.
        drop_dof: Probability of dropping DOF info.
        n_val: Number of validation samples.
        rnn_w: RNN hidden size.
        rnn_d: Number of RNN layers.
        lin_w: Linear head hidden size (unused when lin_d=0).
        lin_d: Number of linear head hidden layers.
        layernorm: Use layer normalization.
        celltype: RNN cell type (gru, lstm, rnn).
        rand_ori: Randomly rotate IMU measurements.
        save_dir: Directory for saving model weights.
        checkpoint_dir: Directory for periodic checkpoints.
        checkpoint_every: Checkpoint every N epochs.
        num_workers: DataLoader workers.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if rand_ori:
        warnings.warn(
            "Currently, the way random IMU orientation is implemented (`rand_ori`"
            "=True) is by rotating the `acc` and `gyr` measurements. This means that"
            "joint axes information is not correctly rotated also."
        )

    lam = (-1, 0)
    n_segments = len(lam)

    # Feature config
    imtp = FeatureConfig(
        joint_axes_1d=drop_ja_1d != 1,
        joint_axes_2d=drop_ja_2d != 1,
        dof=drop_dof != 1,
        dt=True,
        scale_acc=9.81,
        scale_gyr=2.2,
        scale_dt=0.01,
        scale_ja=0.33,
    )

    # Dataset
    transform = Transform(
        imtp,
        drop_imu_1d,
        drop_imu_2d,
        drop_imu_3d,
        drop_ja_1d,
        drop_ja_2d,
        drop_dof,
        # NOTE: ring hardcodes rand_ori=False here (the CLI flag is accepted
        # but never actually passed to Transform).  We replicate that behavior.
        rand_ori=False,
    )
    ds = MultiDataset(
        [ShuffledDataset(FolderOfFilesDataset(p)) for p in [path_lam4] * 4],
        transform,
    )
    ds_train, ds_val = random_split(ds, [len(ds) - n_val, n_val])

    dl_train = DataLoader(
        ds_train,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=min(len(ds_val), bs),
        shuffle=False,
        num_workers=0,
    )
    # Wrap training DataLoader to yield 1 batch per epoch, matching ring's
    # 1-batch-per-episode semantics.
    dls = _SimpleDataLoaders(_OneBatchDataLoader(dl_train), dl_val)

    # Model
    input_f = imtp.getF()
    input_size = input_f * n_segments  # flattened: N segments * F features

    if lin_d > 0 and lin_w != rnn_w:
        warnings.warn(
            f"lin_w ({lin_w}) != rnn_w ({rnn_w}) with lin_d={lin_d}. "
            f"tsfast's SimpleRNN uses rnn_w for the linear head width. "
            f"Ring uses lin_w. The architectures will differ."
        )

    model = RNNOModel(
        input_size=input_size,
        n_segments=n_segments,
        hidden_size=rnn_w,
        num_layers=rnn_d,
        linear_layers=lin_d,
        rnn_type=celltype,
        normalization="layernorm" if layernorm else "",
    )

    if warmstart is not None:
        state_dict = torch.load(
            os.path.expanduser(warmstart), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)
        print(f"Loaded warmstart weights from {warmstart}")

    # Wandb
    if use_wandb:
        import wandb

        wandb.init(project=wandb_project, config=locals(), name=wandb_name)

    # Paths
    save_dir = os.path.expanduser(save_dir)
    checkpoint_dir = os.path.expanduser(checkpoint_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_rnno.pt")

    # Loss, metrics, transforms
    loss_fn = rnno_loss_factory(lam, n_segments)
    metrics = [
        _make_metric("mae_deg", relativeAngle, n_segments),
        _make_metric("incl_deg", inclinationAngle, n_segments),
    ]

    # Ring uses LAMB optimizer (optax.lamb) with no weight decay.
    # Use torch_optimizer.Lamb for equivalence.
    import torch_optimizer

    opt_func = lambda params, lr: torch_optimizer.Lamb(  # noqa: E731
        params, lr=lr, weight_decay=0.0
    )

    # Ring steps the LR schedule per TBPTT chunk.  It hardcodes
    # n_steps_per_episode = int(6000 / tbp) based on T=6000.
    n_chunks_per_seq = int(6000 / tbp)

    # Learner
    learner = RNNOLearner(
        model=model,
        dls=dls,
        loss_func=loss_fn,
        metrics=metrics,
        opt_func=opt_func,
        transforms=[FlattenSegments()],
        grad_clip=0.5,
        sub_seq_len=tbp,
        use_wandb=use_wandb,
        save_path=save_path,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
    )

    # Train
    learner.fit(
        n_epoch=episodes,
        lr=lr,
        scheduler_fn=warmup_cosine_schedule_factory(
            pct_warmup=0.65, n_chunks_per_seq=n_chunks_per_seq
        ),
    )

    # Save final model
    final_path = os.path.join(save_dir, "final_rnno.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    if save_path and os.path.exists(save_path):
        print(f"Best model (by val loss) saved to {save_path}")

    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
