"""Evaluate IMT RNNO and DFJIMU MEKF on the DFJIMU robotic-manipulator dataset.

Follows the same pattern as compare_ring.py: loads data, runs methods,
saves ground-truth + predictions to HDF5.
"""

import os

import fire
import h5py
import numpy as np
import scipy.io as sio
from scipy.signal import resample
from dfjimu import mekf_acc
from imt.methods import RNNO

# Optional: load trained model for comparison
try:
    import torch
    from train_rnno import RNNOModel, FeatureConfig  # noqa: F401
    HAS_MODEL = True
except Exception:
    HAS_MODEL = False

RNNO_TS = 0.01  # RNNO requires 100 Hz

DATA_FILES = [
    "data_1D_01", "data_1D_02", "data_1D_03", "data_1D_04", "data_1D_05",
    "data_2D_01", "data_2D_02", "data_2D_03", "data_2D_05", "data_2D_07",
    "data_3D_01", "data_3D_02", "data_3D_03", "data_3D_04", "data_3D_05",
]


def _h5_overwrite(grp, name, data):
    if name in grp:
        del grp[name]
    grp[name] = data


def _resample_to_100hz(arr, src_hz=50):
    """Resample (N, C) array from src_hz to 100 Hz."""
    N = arr.shape[0]
    N_new = int(N * 100 / src_hz)
    return resample(arr, N_new, axis=0).astype(np.float64)


def _run_imt(method, acc_p, gyr_p, acc_i, gyr_i):
    """Run RNNO at 100 Hz (resamples internally)."""
    method.setTs(RNNO_TS)
    method.reset()
    qrel, _ = method.apply(
        T=acc_p.shape[0],
        acc1=acc_p, gyr1=gyr_p, mag1=None,
        acc2=acc_i, gyr2=gyr_i, mag2=None,
    )
    return qrel


def _detect_f_per_seg(model):
    w = next(iter(model.model.rnn.parameters()))
    return w.shape[1] // 2


def _run_pair(model, acc_p, gyr_p, acc_i, gyr_i, Ts, f_per_seg):
    T_len = acc_p.shape[0]
    X = np.zeros((1, T_len, 2 * f_per_seg), dtype=np.float32)
    X[0, :, 0:3] = acc_p
    X[0, :, 3:6] = gyr_p
    if f_per_seg > 6:
        X[0, :, 6:7] = Ts
    off = f_per_seg
    X[0, :, off:off + 3] = acc_i
    X[0, :, off + 3:off + 6] = gyr_i
    if f_per_seg > 6:
        X[0, :, off + 6:off + 7] = Ts

    x = torch.from_numpy(X).cuda()
    with torch.no_grad():
        pred, _ = model(x)
    return pred[0].cpu().numpy()


def main(
    data_dir="/Users/daniel/Development/dfjimu/data",
    save="results_dfjimu.h5",
    model_path=None,
):
    Ts = 0.02  # 50 Hz

    rnno = RNNO()

    model = None
    if model_path and HAS_MODEL:
        model = torch.load(model_path, map_location="cuda", weights_only=False)
        model.eval()
        f_per_seg = _detect_f_per_seg(model)
        print(f"Detected {f_per_seg} features per segment (total input: {2 * f_per_seg})")

    with h5py.File(save, "a") as out:
        for fname in DATA_FILES:
            path = os.path.join(data_dir, f"{fname}.mat")
            if not os.path.exists(path):
                print(f"  skipping {fname} (not found)")
                continue

            mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
            data = mat["data"]

            sensor = data.sensorData  # (N, 12)
            acc1 = sensor[:, 0:3]
            gyr1 = sensor[:, 3:6]
            acc2 = sensor[:, 6:9]
            gyr2 = sensor[:, 9:12]

            q1_ref = data.ref[:, 0:4]  # (N, 4) [w,x,y,z]
            q2_ref = data.ref[:, 4:8]
            r1 = -np.asarray(data.r_12, dtype=np.float64)
            r2 = -np.asarray(data.r_21, dtype=np.float64)

            grp = out.require_group(fname)
            pair_grp = grp.require_group("seg1_seg2")

            # Resample to 100 Hz for RNNO
            acc1_100 = _resample_to_100hz(acc1)
            gyr1_100 = _resample_to_100hz(gyr1)
            acc2_100 = _resample_to_100hz(acc2)
            gyr2_100 = _resample_to_100hz(gyr2)
            q1_ref_100 = _resample_to_100hz(q1_ref)
            q2_ref_100 = _resample_to_100hz(q2_ref)

            _h5_overwrite(pair_grp, "q_parent", q1_ref_100)
            _h5_overwrite(pair_grp, "q_child", q2_ref_100)

            _h5_overwrite(pair_grp, "qrel_rnno", _run_imt(
                rnno, acc1_100, gyr1_100, acc2_100, gyr2_100
            ))

            # MEKF at 100 Hz — init from ground-truth relative quat at t=0
            q_init = q1_ref[0]  # use original (unit) quaternion
            q1_mekf, q2_mekf = mekf_acc(
                gyr1_100, gyr2_100, acc1_100, acc2_100,
                r1, r2, Fs=100.0, q_init=q_init,
            )
            _h5_overwrite(pair_grp, "q1_mekf", q1_mekf)
            _h5_overwrite(pair_grp, "q2_mekf", q2_mekf)

            if model is not None:
                _h5_overwrite(pair_grp, "pred_ours", _run_pair(
                    model, acc1_100, gyr1_100, acc2_100, gyr2_100, RNNO_TS, f_per_seg
                ))

            print(f"  {fname} done  (N={acc1.shape[0]})")

    print(f"Saved to {save}")


if __name__ == "__main__":
    fire.Fire(main)
