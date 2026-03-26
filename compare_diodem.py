import ctypes
import glob
import os

import torch

torch.cuda.is_available()  # preload CUDA libs so onnxruntime can find them

# Preload TensorRT libs (pip-installed, not on LD_LIBRARY_PATH)
for _lib in sorted(glob.glob(os.path.join(os.path.dirname(__import__("tensorrt_libs").__file__), "*.so*"))):
    try:
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

import onnxruntime as ort  # noqa: E402

# Make onnxruntime default to all GPU providers when none are specified
_orig_init = ort.InferenceSession.__init__


def _patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
    if providers is None:
        providers = ort.get_available_providers()
    _orig_init(self, path_or_bytes, sess_options=sess_options, providers=providers, provider_options=provider_options, **kwargs)


ort.InferenceSession.__init__ = _patched_init

from diodem import load_data, load_all_valid_motions_in_trial  # noqa: E402
import fire  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402

# Required for torch.load to unpickle the saved model
from train_rnno import RNNOModel, FeatureConfig  # noqa: F401, E402

SL_ACC = slice(0, 3)
SL_GYR = slice(3, 6)

N_EXPERIMENTS = 11
BACKEND = "dataverse"

# ARM (exp01-05): seg1→seg2→seg3→seg4→seg5
# GAIT (exp06-11): seg5→seg1→seg2→seg3→seg4
ARM_SEGMENTS = ["seg1", "seg2", "seg3", "seg4", "seg5"]
GAIT_SEGMENTS = ["seg5", "seg1", "seg2", "seg3", "seg4"]


def _segments_for_exp(exp_id):
    return ARM_SEGMENTS if exp_id <= 5 else GAIT_SEGMENTS


def _detect_f_per_seg(model):
    """Detect features-per-segment from the model's first RNN weight."""
    w = next(iter(model.model.rnn.parameters()))
    return w.shape[1] // 2


def _run_pair(model, acc_p, gyr_p, acc_i, gyr_i, Ts, f_per_seg):
    """Run our model on a single segment pair, return full (T, 8) output."""
    T_len = acc_p.shape[0]
    X = np.zeros((1, T_len, 2 * f_per_seg), dtype=np.float32)
    X[0, :, SL_ACC] = acc_p
    X[0, :, SL_GYR] = gyr_p
    if f_per_seg > 6:
        X[0, :, 6:7] = Ts
    off = f_per_seg
    X[0, :, off + SL_ACC.start : off + SL_ACC.stop] = acc_i
    X[0, :, off + SL_GYR.start : off + SL_GYR.stop] = gyr_i
    if f_per_seg > 6:
        X[0, :, off + 6 : off + 7] = Ts

    x = torch.from_numpy(X).cuda()
    with torch.no_grad():
        pred, _ = model(x)
    return pred[0].cpu().numpy()


def _run_imt(method, acc_p, gyr_p, acc_i, gyr_i, Ts):
    """Run an IMT method on a single segment pair, return (T, 4) quats."""
    method.setTs(Ts)
    method.reset()
    qrel, _ = method.apply(
        T=acc_p.shape[0],
        acc1=acc_p, gyr1=gyr_p, mag1=None,
        acc2=acc_i, gyr2=gyr_i, mag2=None,
    )
    return qrel


def _h5_overwrite(grp, name, data):
    """Write a dataset, replacing it if it already exists."""
    if name in grp:
        del grp[name]
    grp[name] = data


def main(
    save: str = "results.h5",
    Ts: float = 0.01,
    warmup: float = 5,
    model_path: str = "rnno_v3.pt",
    name: str | None = None,
    ours_only: bool = False,
):
    # Derive dataset key from model filename if not provided
    if name is None:
        name = os.path.splitext(os.path.basename(model_path))[0]
    pred_key = f"pred_{name}"
    print(f"Model predictions will be stored as '{pred_key}'")

    model = torch.load(model_path, map_location="cuda", weights_only=False)
    model.eval()
    f_per_seg = _detect_f_per_seg(model)
    print(f"Detected {f_per_seg} features per segment (total input: {2 * f_per_seg})")

    if not ours_only:
        from imt.methods import RNNO, RNNO_rO
        rnno = RNNO()
        rnno_ro = RNNO_rO()

    with h5py.File(save, "a") as f:
        f.attrs["Ts"] = Ts
        f.attrs["warmup"] = warmup

        for exp_id in range(1, N_EXPERIMENTS + 1):
            segments = _segments_for_exp(exp_id)
            pairs = [(segments[i], segments[i + 1]) for i in range(len(segments) - 1)]
            n_motions = len(load_all_valid_motions_in_trial(exp_id, backend=BACKEND))

            for motion in range(1, n_motions + 1):
                data = load_data(
                    exp_id,
                    motion_start=motion,
                    motion_stop=motion,
                    resample_to_hz=1 / Ts,
                    backend=BACKEND,
                )

                for imu_key in ("imu_rigid", "imu_nonrigid"):
                    imu_label = imu_key.removeprefix("imu_")
                    key = f"exp{exp_id:02d}/motion{motion:02d}/{imu_label}"
                    print(f"{key} ...", end=" ", flush=True)

                    for seg_p, seg_i in pairs:
                        acc_p = data[seg_p][imu_key]["acc"]
                        gyr_p = data[seg_p][imu_key]["gyr"]
                        acc_i = data[seg_i][imu_key]["acc"]
                        gyr_i = data[seg_i][imu_key]["gyr"]
                        pair_key = f"{key}/{seg_p}_{seg_i}"

                        grp = f.require_group(pair_key)
                        _h5_overwrite(grp, "q_parent", data[seg_p]["quat"])
                        _h5_overwrite(grp, "q_child", data[seg_i]["quat"])
                        _h5_overwrite(grp, pred_key, _run_pair(
                            model, acc_p, gyr_p, acc_i, gyr_i, Ts, f_per_seg
                        ))
                        if not ours_only:
                            _h5_overwrite(grp, "qrel_rnno", _run_imt(
                                rnno, acc_p, gyr_p, acc_i, gyr_i, Ts
                            ))
                            _h5_overwrite(grp, "qrel_rnno_rO", _run_imt(
                                rnno_ro, acc_p, gyr_p, acc_i, gyr_i, Ts
                            ))

                    print("done")

    print(f"Saved to {save}")


if __name__ == "__main__":
    fire.Fire(main)
