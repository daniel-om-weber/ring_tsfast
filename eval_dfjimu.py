"""Evaluate relative orientation error on the DFJIMU dataset.

Reads results_dfjimu.h5 (produced by compare_dfjimu.py) and prints per-trial
and per-category RMSE for all methods found in the file.
"""

from collections import defaultdict

import fire
import h5py
import numpy as np
import qmt


def _normalize(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _angle_err(q_est, q_true):
    return np.rad2deg(np.abs(qmt.quatAngle(qmt.qrel(q_true, q_est))))


def _rmse(err):
    return np.sqrt(np.nanmean(err ** 2))


def main(h5_path: str = "results_dfjimu.h5", warmup: float = 5.0, Ts: float = 0.01):
    warmup_idx = int(warmup / Ts)

    # {method_name: {dim: [err_array, ...]}}
    errors = defaultdict(lambda: defaultdict(list))
    trial_names = []
    trial_results = []

    with h5py.File(h5_path, "r") as f:
        for name in sorted(f.keys()):
            grp = f[name]["seg1_seg2"]
            q_parent = _normalize(np.array(grp["q_parent"]))
            q_child = _normalize(np.array(grp["q_child"]))
            qrel_truth = qmt.qrel(q_parent, q_child)
            dim = name.split("_")[1]  # "1D", "2D", "3D"

            methods = {}

            # RNNO: stored as relative quaternion directly
            if "qrel_rnno" in grp:
                methods["RNNO"] = np.array(grp["qrel_rnno"])

            # MEKF: stored as absolute q1, q2 — compute relative
            if "q1_mekf" in grp and "q2_mekf" in grp:
                q1 = _normalize(np.array(grp["q1_mekf"]))
                q2 = _normalize(np.array(grp["q2_mekf"]))
                methods["MEKF"] = qmt.qrel(q1, q2)

            # Any pred_* keys (trained models)
            for key in grp.keys():
                if key.startswith("pred_"):
                    label = key.removeprefix("pred_")
                    methods[label] = np.array(grp[key])[:, 4:8]

            row = {"name": name, "dim": dim}
            for method, qrel_est in methods.items():
                err = _angle_err(qrel_est, qrel_truth)[warmup_idx:]
                errors[method][dim].append(err)
                row[method] = _rmse(err)

            trial_names.append(name)
            trial_results.append(row)

    method_names = sorted(errors.keys())

    # Per-trial table
    header = f"{'Trial':<15s}"
    for m in method_names:
        header += f"  {m:>10s}"
    print(header)
    print("-" * len(header))

    for row in trial_results:
        line = f"{row['name']:<15s}"
        for m in method_names:
            if m in row:
                line += f"  {row[m]:>9.2f}°"
            else:
                line += f"  {'—':>10s}"
        print(line)

    # Summary by dimensionality
    print()
    header = f"{'Category':<10s}"
    for m in method_names:
        header += f"  {m:>10s}"
    print(header)
    print("-" * len(header))

    for dim in ["1D", "2D", "3D"]:
        line = f"{dim:<10s}"
        for m in method_names:
            if dim in errors[m]:
                cat = np.concatenate(errors[m][dim])
                line += f"  {_rmse(cat):>9.2f}°"
            else:
                line += f"  {'—':>10s}"
        print(line)

    line = f"{'Overall':<10s}"
    for m in method_names:
        all_err = np.concatenate([e for errs in errors[m].values() for e in errs])
        line += f"  {_rmse(all_err):>9.2f}°"
    print(line)


if __name__ == "__main__":
    fire.Fire(main)
