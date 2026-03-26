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

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from imt.methods import RNNO  # noqa: E402

# Required for torch.load to unpickle the saved model
from train_rnno import RNNOModel, FeatureConfig  # noqa: F401, E402

SEGS = ["seg2", "seg3", "seg4", "seg5"]
PAIRS = [(SEGS[i], SEGS[i + 1]) for i in range(len(SEGS) - 1)]


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


def _run_imt(method, acc_p, gyr_p, acc_i, gyr_i, Ts):
    method.setTs(Ts)
    method.reset()
    qrel, _ = method.apply(
        T=acc_p.shape[0],
        acc1=acc_p, gyr1=gyr_p, mag1=None,
        acc2=acc_i, gyr2=gyr_i, mag2=None,
    )
    return qrel


def _h5_overwrite(grp, name, data):
    if name in grp:
        del grp[name]
    grp[name] = data


def _find_100hz_files(data_dir, n):
    """Find the first n files with dt~=0.01 by scanning sequentially."""
    found = []
    idx = 0
    while len(found) < n:
        path = os.path.join(data_dir, f"seq{idx}.h5")
        if not os.path.exists(path):
            break
        with h5py.File(path, "r") as f:
            if abs(float(f.attrs["dt"]) - 0.01) < 1e-5:
                found.append(path)
        idx += 1
    return found


def main(data_dir, n_seq=2, model_path="rnno_v4_fixed_dt.pt", save="results_ring.h5"):
    model = torch.load(model_path, map_location="cuda", weights_only=False)
    model.eval()
    f_per_seg = _detect_f_per_seg(model)
    print(f"Detected {f_per_seg} features per segment (total input: {2 * f_per_seg})")

    seq_files = _find_100hz_files(data_dir, n_seq)
    print(f"Found {len(seq_files)} sequences at 100 Hz")

    rnno = RNNO()

    with h5py.File(save, "a") as out:
        for seq_file in seq_files:
            seq_name = os.path.splitext(os.path.basename(seq_file))[0]
            with h5py.File(seq_file, "r") as src:
                for seg_p, seg_c in PAIRS:
                    acc_p = src[f"{seg_p}_acc"][:]
                    gyr_p = src[f"{seg_p}_gyr"][:]
                    acc_c = src[f"{seg_c}_acc"][:]
                    gyr_c = src[f"{seg_c}_gyr"][:]

                    pair_key = f"{seq_name}/{seg_p}_{seg_c}"
                    grp = out.require_group(pair_key)

                    _h5_overwrite(grp, "q_parent", src[f"{seg_p}_q"][:])
                    _h5_overwrite(grp, "q_child", src[f"{seg_c}_q"][:])

                    _h5_overwrite(grp, "pred_ours", _run_pair(
                        model, acc_p, gyr_p, acc_c, gyr_c, 0.01, f_per_seg
                    ))
                    _h5_overwrite(grp, "qrel_rnno", _run_imt(
                        rnno, acc_p, gyr_p, acc_c, gyr_c, 0.01
                    ))

                    print(f"{seq_name}/{seg_p}_{seg_c} done")

    print(f"Saved to {save}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
