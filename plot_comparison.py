import matplotlib
matplotlib.use("Agg")
from collections import defaultdict
import numpy as np
import qmt
import matplotlib.pyplot as plt
import h5py
import fire


# ARM (exp01-05): seg1→seg2→seg3→seg4→seg5
_ARM_DOF = {"seg1_seg2": 3, "seg2_seg3": 1, "seg3_seg4": 1, "seg4_seg5": 1}
# GAIT (exp06-11): seg5→seg1→seg2→seg3→seg4
_GAIT_DOF = {"seg5_seg1": 1, "seg1_seg2": 2, "seg2_seg3": 2, "seg3_seg4": 1}


def _get_dof(exp_name, pair_name):
    exp_id = int(exp_name[3:])
    dof_map = _ARM_DOF if exp_id <= 5 else _GAIT_DOF
    return dof_map[pair_name]


def _angle_err(q_est, q_true):
    return np.rad2deg(np.abs(qmt.quatAngle(qmt.qrel(q_true, q_est))))


def _rmse(err):
    return np.sqrt(np.nanmean(err ** 2))


def main(h5_path: str = "results.h5", out: str = "comparison.png"):
    with h5py.File(h5_path, "r") as f:
        Ts = f.attrs["Ts"]
        warmup = f.attrs["warmup"]

        entries = []
        for exp_name in sorted(f.keys()):
            for mot_name in sorted(f[exp_name].keys()):
                for imu_name in sorted(f[exp_name][mot_name].keys()):
                    for pair_name in sorted(f[exp_name][mot_name][imu_name].keys()):
                        grp = f[exp_name][mot_name][imu_name][pair_name]
                        entry = {
                            "exp_name": exp_name,
                            "imu_name": imu_name,
                            "pair_name": pair_name,
                            "q_parent": np.array(grp["q_parent"]),
                            "q_child": np.array(grp["q_child"]),
                        }
                        # Collect all pred_* and qrel_* datasets
                        for ds_name in grp.keys():
                            if ds_name.startswith("pred_") or ds_name.startswith("qrel_"):
                                entry[ds_name] = np.array(grp[ds_name])
                        entries.append(entry)

    warmup_idx = int(warmup / Ts)

    # Collect errors grouped by (dof, imu_type)
    # errors_by_group[(dof, imu_name)][method_name] = list of error arrays
    errors_by_group = defaultdict(lambda: defaultdict(list))

    for entry in entries:
        dof = _get_dof(entry["exp_name"], entry["pair_name"])
        group_key = (dof, entry["imu_name"])
        qrel_truth = qmt.qrel(entry["q_parent"], entry["q_child"])

        methods = []
        for ds_name in sorted(entry.keys()):
            if ds_name.startswith("qrel_"):
                label = ds_name.removeprefix("qrel_")
                methods.append((label, entry[ds_name]))
            elif ds_name.startswith("pred_"):
                label = ds_name.removeprefix("pred_")
                methods.append((label, entry[ds_name][:, 4:8]))

        for name, q_est in methods:
            err = _angle_err(q_est, qrel_truth)
            errors_by_group[group_key][name].append(err[warmup_idx:])

    # Compute aggregated RMSE per group
    group_keys = sorted(errors_by_group.keys())
    method_names = list(next(iter(errors_by_group.values())).keys())

    print(f"{'Variant':<20s}", end="")
    for m in method_names:
        print(f"  {m:>10s}", end="")
    print()
    print("-" * (20 + 12 * len(method_names)))

    group_rmses = {}
    for gk in group_keys:
        dof, imu = gk
        label = f"{dof}DOF / {imu}"
        print(f"{label:<20s}", end="")
        group_rmses[gk] = {}
        for m in method_names:
            cat = np.concatenate(errors_by_group[gk][m])
            rmse = _rmse(cat)
            group_rmses[gk][m] = rmse
            print(f"  {rmse:>9.2f}°", end="")
        print()

    # Time-series plot: 3 rows (DOF) × 2 cols (rigid/nonrigid)
    dof_levels = sorted(set(d for d, _ in group_keys))
    imu_levels = ["rigid", "nonrigid"]
    n_rows = len(dof_levels)
    n_cols = len(imu_levels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows),
                             sharex=False, sharey="row")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_colors = {m: colors[i] for i, m in enumerate(method_names)}

    for row, dof in enumerate(dof_levels):
        for col, imu in enumerate(imu_levels):
            ax = axes[row, col]
            gk = (dof, imu)
            if gk not in errors_by_group:
                ax.set_visible(False)
                continue

            for m in method_names:
                seqs = errors_by_group[gk][m]
                c = method_colors[m]

                # Concatenate sequences and plot raw error
                cat = np.concatenate(seqs)
                t = np.arange(len(cat)) * Ts
                ax.plot(t, cat, color=c, alpha=0.25, lw=0.3)

                # Per-sequence RMSE as step line + vertical separators
                offset = 0
                seg_rmses_t = []
                seg_rmses_v = []
                for seq in seqs:
                    n_pts = len(seq)
                    rmse_seq = _rmse(seq)
                    t_start = offset * Ts
                    t_end = (offset + n_pts) * Ts
                    seg_rmses_t.extend([t_start, t_end])
                    seg_rmses_v.extend([rmse_seq, rmse_seq])
                    if offset > 0:
                        ax.axvline(t_start, color="gray", lw=0.3, alpha=0.3)
                    offset += n_pts

                ax.plot(seg_rmses_t, seg_rmses_v, color=c, lw=2, alpha=0.9,
                        label=f"{m} (RMSE {group_rmses[gk][m]:.1f}°)")

            ax.set_title(f"{dof}DOF / {imu}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=7, loc="upper right")
            ax.set_ylabel("Error [deg]")
            if row == n_rows - 1:
                ax.set_xlabel("Concatenated time [s]")

    fig.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    fire.Fire(main)
