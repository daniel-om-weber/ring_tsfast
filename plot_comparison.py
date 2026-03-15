import matplotlib
matplotlib.use("Agg")
import numpy as np
import qmt
import matplotlib.pyplot as plt
import h5py
import fire


def _angle_err(q_est, q_true):
    return np.rad2deg(np.abs(qmt.quatAngle(qmt.qrel(q_true, q_est))))


def _rmse(err):
    return np.sqrt(np.mean(err ** 2))


def main(h5_path: str = "results.h5", out: str = "comparison.png"):
    with h5py.File(h5_path, "r") as f:
        Ts = f.attrs["Ts"]
        warmup = f.attrs["warmup"]

        # Collect all experiment/motion/pair groups
        entries = []
        for exp_name in sorted(f.keys()):
            for mot_name in sorted(f[exp_name].keys()):
                for imu_name in sorted(f[exp_name][mot_name].keys()):
                    for pair_name in sorted(f[exp_name][mot_name][imu_name].keys()):
                        grp = f[exp_name][mot_name][imu_name][pair_name]
                        entry = {
                            "label": f"{exp_name}/{mot_name}/{imu_name}/{pair_name}",
                            "q_parent": np.array(grp["q_parent"]),
                            "q_child": np.array(grp["q_child"]),
                            "pred_ours": np.array(grp["pred_ours"]),
                        }
                        if "qrel_rnno" in grp:
                            entry["qrel_rnno"] = np.array(grp["qrel_rnno"])
                        if "qrel_rnno_rO" in grp:
                            entry["qrel_rnno_rO"] = np.array(grp["qrel_rnno_rO"])
                        entries.append(entry)

    n = len(entries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    warmup_idx = int(warmup / Ts)

    for ax, entry in zip(axes, entries):
        qrel_truth = qmt.qrel(entry["q_parent"], entry["q_child"])
        t = np.arange(len(entry["q_parent"])) * Ts

        methods = []
        if "qrel_rnno" in entry:
            methods.append(("RNNO", entry["qrel_rnno"]))
        if "qrel_rnno_rO" in entry:
            methods.append(("RNNO_rO", entry["qrel_rnno_rO"]))
        methods.append(("Ours", entry["pred_ours"][:, 4:8]))

        for name, q_est in methods:
            err = _angle_err(q_est, qrel_truth)
            rmse = _rmse(err[warmup_idx:])
            ax.plot(t, err, label=f"{name} (RMSE {rmse:.2f}°)", alpha=0.8)
            print(f"{entry['label']:50s} {name:10s} RMSE = {rmse:.2f}°")

        ax.axvline(warmup, color="gray", ls="--", lw=0.8)
        ax.set_ylabel("Error [deg]")
        ax.set_title(entry["label"], fontsize=9)
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    fire.Fire(main)
