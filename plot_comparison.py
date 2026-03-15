import matplotlib
matplotlib.use("Agg")
import numpy as np
import qmt
import matplotlib.pyplot as plt
import h5py
import fire


def _angle_err(q_est, q_true):
    return np.rad2deg(np.abs(qmt.quatAngle(qmt.qrel(q_true, q_est))))


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
                        entries.append((
                            f"{exp_name}/{mot_name}/{imu_name}/{pair_name}",
                            np.array(grp["q_parent"]),
                            np.array(grp["q_child"]),
                            np.array(grp["pred_ours"]),
                            np.array(grp["qrel_rnno"]),
                            np.array(grp["qrel_rnno_rO"]),
                        ))

    n = len(entries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (label, q_p, q_c, pred, rnno, rnno_ro) in zip(axes, entries):
        qrel_truth = qmt.qrel(q_p, q_c)
        t = np.arange(len(q_p)) * Ts

        ax.plot(t, _angle_err(rnno, qrel_truth), label="RNNO", alpha=0.8)
        ax.plot(t, _angle_err(rnno_ro, qrel_truth), label="RNNO_rO", alpha=0.8)
        ax.plot(t, _angle_err(pred[:, 4:8], qrel_truth), label="Ours", alpha=0.8)
        ax.axvline(warmup, color="gray", ls="--", lw=0.8)
        ax.set_ylabel("Error [deg]")
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    fire.Fire(main)
