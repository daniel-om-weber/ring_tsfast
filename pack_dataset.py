"""Convert pickle dataset to per-file HDF5 with 2-D signal datasets (tsdata-compatible).

Each pickle produces one HDF5 file with 12 datasets of length T=6000:
  seg{2..5}_acc   (T, 3)
  seg{2..5}_gyr   (T, 3)
  seg{2..5}_q     (T, 4)

Plus scalar/array attrs: dt, seg{3..5}_dof, seg{3..5}_ja_rr, seg{3..5}_ja_rsaddle.
All datasets are contiguous (chunks=None, no compression) for mmap by HDF5Signals.
"""

import pickle
import random
from pathlib import Path

import fire
import h5py
import numpy as np
from tqdm import tqdm

OLD_SEGMENTS = ["seg2_4Seg", "seg3_4Seg", "seg4_4Seg", "seg5_4Seg"]
SEGS = ["seg2", "seg3", "seg4", "seg5"]

def convert_one(pickle_path: str, h5_path: str, max_samples: int | None = None):
    with open(pickle_path, "rb") as f:
        X, Y = pickle.load(f)

    T = slice(None, max_samples)
    with h5py.File(h5_path, "w") as f:
        f.attrs["dt"] = np.float32(X["dt"])

        for old_seg, seg in zip(OLD_SEGMENTS, SEGS):
            f.create_dataset(f"{seg}_acc", data=X[old_seg]["acc"][T].astype(np.float32), chunks=None)
            f.create_dataset(f"{seg}_gyr", data=X[old_seg]["gyr"][T].astype(np.float32), chunks=None)
            f.create_dataset(f"{seg}_q", data=Y[old_seg][T].astype(np.float32), chunks=None)

            if seg != "seg2":
                dof = int(X[old_seg]["dof"])
                f.attrs[f"{seg}_dof"] = np.int32(dof)

                ja_rr = np.zeros(3, dtype=np.float32)
                if dof == 1 and "rr" in X[old_seg].get("joint_params", {}):
                    ja_rr = X[old_seg]["joint_params"]["rr"]["joint_axes"].astype(np.float32)
                f.attrs[f"{seg}_ja_rr"] = ja_rr

                ja_rsaddle = np.zeros(6, dtype=np.float32)
                if dof == 2 and "rsaddle" in X[old_seg].get("joint_params", {}):
                    ja_rsaddle = X[old_seg]["joint_params"]["rsaddle"]["joint_axes"].astype(np.float32)
                f.attrs[f"{seg}_ja_rsaddle"] = ja_rsaddle


def verify_sample(pickle_path: str, h5_path: str):
    with open(pickle_path, "rb") as f:
        X_pkl, Y_pkl = pickle.load(f)

    with h5py.File(h5_path, "r") as f:
        T = f[f"{SEGS[0]}_acc"].shape[0]
        assert np.isclose(f.attrs["dt"], X_pkl["dt"], atol=1e-7)
        for old_seg, seg in zip(OLD_SEGMENTS, SEGS):
            np.testing.assert_allclose(f[f"{seg}_acc"][()], X_pkl[old_seg]["acc"][:T].astype(np.float32), atol=1e-7)
            np.testing.assert_allclose(f[f"{seg}_gyr"][()], X_pkl[old_seg]["gyr"][:T].astype(np.float32), atol=1e-7)
            np.testing.assert_allclose(f[f"{seg}_q"][()], Y_pkl[old_seg][:T].astype(np.float32), atol=1e-7)
            if seg != "seg2":
                assert int(f.attrs[f"{seg}_dof"]) == int(X_pkl[old_seg]["dof"])


def main(pickle_dir: str = "ring_data", h5_dir: str = "ring_data_h5", max_samples: int | None = None):
    pickle_dir = Path(pickle_dir)
    h5_dir = Path(h5_dir)
    h5_dir.mkdir(exist_ok=True)

    pickle_files = sorted(str(p) for p in pickle_dir.iterdir() if p.suffix == ".pickle")
    print(f"Found {len(pickle_files)} pickle files")
    print(f"Output directory: {h5_dir}")

    for pkl_path in tqdm(pickle_files, desc="Converting"):
        name = Path(pkl_path).stem
        h5_path = str(h5_dir / f"{name}.h5")
        convert_one(pkl_path, h5_path, max_samples)

    n_verify = min(10, len(pickle_files))
    verify_indices = random.sample(range(len(pickle_files)), n_verify)
    print(f"Verifying {n_verify} random samples...")
    for i in verify_indices:
        pkl_path = pickle_files[i]
        h5_path = str(h5_dir / f"{Path(pkl_path).stem}.h5")
        verify_sample(pkl_path, h5_path)
    print("Verification passed.")


if __name__ == "__main__":
    fire.Fire(main)
