from dataclasses import replace
from typing import Optional
import warnings

from benchmark import benchmark as benchmark_fn
from benchmark import IMTP
import fire
import jax.numpy as jnp
import numpy as np
import optax
import qmt
import ring
from ring import maths
from ring import ml
from ring.extras.dataloader_torch import dataset_to_generator
from ring.extras.dataloader_torch import dataset_to_Xy
from ring.extras.dataloader_torch import FolderOfFilesDataset
from ring.extras.dataloader_torch import MultiDataset
from ring.extras.dataloader_torch import ShuffledDataset
from torch.utils.data import random_split

import wandb


def _cb_metrices_factory(warmup: int = 0):
    return dict(
        mae_deg=lambda q, qhat: jnp.rad2deg(
            jnp.mean(maths.angle_error(q, qhat)[:, warmup:])
        ),
        incl_deg=lambda q, qhat: jnp.rad2deg(
            jnp.mean(maths.inclination_loss(q, qhat)[:, warmup:])
        ),
    )


def _params(unique_id: str = ring.ml.unique_id()) -> str:
    home = "/bigwork/nhkbbach/" if ring.ml.on_cluster() else "~/"
    return home + f"params/{unique_id}.pickle"


def _model(unique_id: str = ring.ml.unique_id()) -> str:
    home = "/bigwork/nhkbbach/" if ring.ml.on_cluster() else "~/"
    return home + f"params/model_{unique_id}.pickle"


def _checkpoints(unique_id: Optional[str] = None) -> str:
    home = "/bigwork/nhkbbach/" if ring.ml.on_cluster() else "~/"
    if unique_id is not None:
        return home + f"ring_checkpoints/{unique_id}.pickle"
    else:
        return home + "ring_checkpoints"


class DumpModelCallback(ring.ml.training_loop.TrainingLoopCallback):
    def __init__(
        self,
        path: str,
        ringnet: ring.ml.ringnet.RING,
        overwrite: bool = False,
        dump_every: Optional[int] = None,
    ):
        self.path = ring.utils.parse_path(
            path,
            extension="pickle",
            file_exists_ok=overwrite,
        )
        self.ringnet = ringnet.unwrapped_deep
        self.params = None
        self.dump_every = dump_every

    def after_training_step(
        self, i_episode, metrices, params, grads, sample_eval, loggers, opt_state
    ):
        self.params = params
        if self.dump_every is not None and ((i_episode % self.dump_every) == 0):
            self.close()

    def close(self):
        if self.params is not None:
            self.ringnet.params = self.params
            ring.utils.pickle_save(self.ringnet.nojit(), self.path, overwrite=True)


def act_fn_rnno(X):
    return X


def _make_net(lam, warmstart, rnn_w, rnn_d, lin_w, lin_d, layernorm, celltype):

    kwargs = {
        "forward_factory": ring.ml.rnno_v1.rnno_v1_forward_factory,
        "rnn_layers": [rnn_w] * rnn_d,
        "linear_layers": [lin_w] * lin_d,
        "act_fn_rnn": act_fn_rnno,
        "output_dim": len(lam) * 4,
    }

    net = ring.ml.RING(
        params=_params(hex(warmstart)) if warmstart else None,
        celltype=celltype,
        lam=lam,
        layernorm=layernorm,
        **kwargs,
    )
    net = ring.ml.base.NoGraph_FilterWrapper(net, quat_normalize=True)

    return net


def _loss_fn_ring_factory(lam):
    def _loss_fn_ring(q, qhat):
        "T, N, F -> Scalar"
        loss = jnp.array(0.0)
        for i, p in enumerate(lam):
            if p == -1:
                loss += jnp.mean(ring.maths.inclination_loss(q[:, i], qhat[:, i]) ** 2)
            else:
                loss += jnp.mean(ring.maths.angle_error(q[:, i], qhat[:, i]) ** 2)
        return loss / len(lam)

    return _loss_fn_ring


class Transform:
    chain = ["seg2_4Seg", "seg3_4Seg", "seg4_4Seg", "seg5_4Seg"]
    lam = [-1, 0]

    link_names = [
        "seg3_2Seg",
        "seg4_2Seg",
    ]

    def __init__(
        self,
        imtp: IMTP,
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

        for name in y:  # not X because it also has `dt` key
            # let this be from B -> B'
            qrand = qmt.randomQuat()
            X[name]["acc"] = qmt.rotate(qrand, X[name]["acc"])
            X[name]["gyr"] = qmt.rotate(qrand, X[name]["gyr"])
            # Y is B -> E; we need B' -> E
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

        draw = lambda p: 1.0 - np.random.binomial(1, p=p)

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
                        X1[name]["joint_params"]["rsaddle"]["joint_axes"].reshape(6, 1)
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

        X, Y = X.transpose((2, 1, 0)), Y.transpose((1, 0, 2))
        return X, Y


def _make_exp_callbacks(ringnet, imtp: IMTP):

    callbacks, metrices_name = [], []

    def add_callback(segments: list[str], exp_id, motion_start):
        cb = benchmark_fn(
            imtp=replace(imtp, segments=segments),
            exp_id=exp_id,
            motion_start=motion_start,
            filter=ringnet,
            return_cb=True,
            cb_metrices=_cb_metrices_factory(500),
        )
        callbacks.append(cb)
        # exclude the first element because it connects to -1 and thus its `mae_deg`
        # loss will be not meaningful
        for segment in segments[1:]:
            metrices_name.append([cb.metric_identifier, "mae_deg", segment])

    add_callback(["seg1", "seg2"], 1, "slow1")
    add_callback(["seg3", "seg4"], 1, "fast")
    add_callback(["seg1", "seg2"], 6, "slow1")
    add_callback(["seg2", "seg3"], 6, "fast")
    # add_callback(["seg3", "seg4", "seg5"], 1, "slow1")
    # add_callback(["seg3", "seg4", "seg5"], 1, "fast")
    # add_callback(["seg2", "seg3", "seg4", "seg5"], 1, "slow1")
    # add_callback(["seg2", "seg3", "seg4", "seg5"], 1, "fast")

    for zoom_in in metrices_name:
        print(zoom_in)
    callbacks += [ml.callbacks.AverageMetricesTLCB(metrices_name, "exp_val_mae_deg")]

    return callbacks


def main(
    path_lam4,
    bs: int,
    episodes: int,
    use_wandb: bool = False,
    wandb_project: str = "RING",
    wandb_name: str = None,
    warmstart: str = None,
    seed: int = 1,
    exp_cbs: bool = False,
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
    skip_first: bool = False,
    rand_ori: bool = False,
    disable_dump_model=False,
    disable_checkpoint_model=False,
    disable_save_params=False,
):
    """
    Main function for training and benchmarking RING neural networks on motion datasets.

    Parameters:
        path_lam4 (str): Path to the dataset containing lam4 sequences.
        bs (int): Batch size for training.
        episodes (int): Number of training episodes.
        use_wandb (bool, optional): Whether to log training progress using Weights & Biases. Defaults to False.
        wandb_project (str, optional): Name of the Weights & Biases project. Defaults to "RING".
        wandb_name (str, optional): Optional name for the Weights & Biases run. Defaults to None.
        warmstart (str, optional): Path to warmstart parameters. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 1.
        exp_cbs (bool, optional): Whether to include experimental evaluation callbacks. Defaults to False.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        tbp (int, optional): Truncated backpropagation length. Defaults to 1000.
        drop_imu_1d (float, optional): Probability of dropping IMUs of segments that connect to parent via 1D
            joints during training. Defaults to 0.75.
        drop_imu_2d (float, optional): Probability of dropping IMUs of segments that connect to parent via 2D
            joints during training. Defaults to 0.25.
        drop_imu_3d (float, optional): Probability of dropping IMUs of segments that connect to parent via 3D
            joints during training. Defaults to 0.1.
        drop_ja_1d (float, optional): Probability of dropping joint axes information of segments that connect to
            parent via 1D joint. Defaults to 0.5.
        drop_ja_2d (float, optional): Probability of dropping 2D joint axes information of segments that connect
            to parent via 2D joint. Defaults to 0.5.
        drop_dof (float, optional): Probability of dropping degrees of freedom. Defaults to 0.0.
        n_val (int, optional): Number of samples to use for validation. Defaults to 256.
        rnno (bool, optional): Whether to use RNNO or RING. If `True` use RNNO. Defaults to False.
        rnn_w (int, optional): Width of the RNN layers. Defaults to 400.
        rnn_d (int, optional): Depth of the RNN layers. Defaults to 2.
        lin_w (int, optional): Width of the linear layers. Defaults to 200.
        lin_d (int, optional): Depth of the linear layers. Defaults to 0.
        layernorm (bool, optional): Whether to use layer normalization. Defaults to False.
        celltype (str, optional): Type of RNN cell to use (e.g., "gru", "lstm"). Defaults to "gru".
        three_seg (bool, optional): Whether to train on 1-Seg, 2-Seg, and 3-Seg chains. Defaults to False.
        four_seg (bool, optional): Whether to train on 1-Seg, 2-Seg, 3-Seg, and 4-Seg chains. Defaults to False.
        skip_first (bool, optional): If `True` skips the first TBPTT minibatch, so the one from (t=0, `tbp`). Defaults to False.
        grad_accu (int, optional): Number of batches per gradient step to accumulate. Defaults to 1.
        rand_ori (bool, optional): If `True` randomly rotate IMU measurements. Defaults to False.
        disable_dump_model (bool, optional): If `True` does not pickle and dump the entire model after training.

    Returns:
        None: Trains the network and optionally logs results to Weights & Biases.
    """  # noqa: E501
    np.random.seed(seed)

    if rand_ori:
        warnings.warn(
            "Currently, the way random IMU orientation is implemented (`rand_ori`"
            "=True) is by rotating the `acc` and `gyr` measurements. This means that"
            "joint axes information is not correctly rotated also."
        )

    if use_wandb:
        unique_id = ring.ml.unique_id()
        wandb.init(project=wandb_project, config=locals(), name=wandb_name)

    lam = (-1, 0)
    link_names = ["seg1", "seg2"]

    net = _make_net(lam, warmstart, rnn_w, rnn_d, lin_w, lin_d, layernorm, celltype)
    imtp = IMTP(
        segments=None,
        sparse=(
            False
            if ((drop_imu_1d == 0) and (drop_imu_2d == 0) and (drop_imu_3d == 0))
            else True
        ),
        joint_axes_1d=False if drop_ja_1d == 1 else True,
        joint_axes_1d_field=False if drop_ja_1d == 1 else True,
        joint_axes_2d=False if drop_ja_2d == 1 else True,
        joint_axes_2d_field=False if drop_ja_2d == 1 else True,
        dof=False if drop_dof == 1 else True,
        dof_field=False if drop_dof == 1 else True,
        dt=True,
        scale_acc=9.81,
        scale_gyr=2.2,
        scale_dt=0.01,
        scale_ja=0.33,
    )
    ds = MultiDataset(
        [ShuffledDataset(FolderOfFilesDataset(p)) for p in [path_lam4] * 4],
        Transform(
            imtp,
            drop_imu_1d,
            drop_imu_2d,
            drop_imu_3d,
            drop_ja_1d,
            drop_ja_2d,
            drop_dof,
            rand_ori=False,
        ),
    )
    ds_train, ds_val = random_split(ds, [len(ds) - n_val, n_val])
    generator = dataset_to_generator(
        ds_train,
        batch_size=bs,
        seed=seed,
        drop_last=True,
        num_workers=None if ring.ml.on_cluster() else 0,
    )
    X_val, y_val = dataset_to_Xy(ds_val)
    callbacks = [
        ring.ml.callbacks.EvalXyTrainingLoopCallback(
            net,
            _cb_metrices_factory(),
            X_val,
            y_val,
            lam,
            "val",
            link_names=link_names,
        )
    ]

    if exp_cbs:
        callbacks.extend(_make_exp_callbacks(net, imtp))

    if not disable_dump_model:
        callbacks.append(
            DumpModelCallback(_model(), net, overwrite=True, dump_every=None)
        )
    if not disable_checkpoint_model:
        callbacks.append(
            ml.callbacks.CheckpointCallback(
                checkpoint_every=5, checkpoint_folder=_checkpoints()
            )
        )

    n_warmup_episodes = int(0.65 * episodes)
    n_decay_episodes = int(0.35 * episodes)
    n_steps_per_episode = int(6000 / tbp)
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.lamb(
            optax.join_schedules(
                [
                    optax.schedules.linear_schedule(
                        0.33 * lr, lr, n_warmup_episodes * n_steps_per_episode
                    ),
                    optax.schedules.cosine_decay_schedule(
                        lr, n_decay_episodes * n_steps_per_episode
                    ),
                ],
                [n_warmup_episodes * n_steps_per_episode],
            )
        ),
    )

    ml.train_fn(
        generator,
        episodes,
        net,
        optimizer=optimizer,
        callbacks=callbacks,
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        callback_save_params=False if disable_save_params else _params(),
        callback_save_params_track_metrices=(
            [["exp_val_mae_deg"]] if (exp_cbs and not disable_save_params) else None
        ),
        seed_network=seed,
        link_names=link_names,
        tbp=tbp,
        loss_fn=_loss_fn_ring_factory(lam),
        metrices=None,
        skip_first_tbp_batch=skip_first,
        callback_create_checkpoint=False,
    )


if __name__ == "__main__":
    fire.Fire(main)
