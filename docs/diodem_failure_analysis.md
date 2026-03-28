# DIODEM Failure Analysis

Analysis date: 2026-03-28

## Overview

Models `rnno_v6_mogen` and `rnno_v6_mogen_adamw` trained on MoGen-simulated data
(`data_mogen_100_noart/`, 48k sequences) were evaluated on the DIODEM dataset.
DIODEM uses a 3D-printed five-segment kinematic chain with Xsens MTw Awinda IMUs
(40 Hz, resampled to 100 Hz) and OptiTrack ground truth (120 Hz).

Aggregate RMSE from `plot_comparison.py` on `results.h5`:

| Variant            | rnno_v6_mogen | rnno_v6_mogen_adamw |
|--------------------|---------------|---------------------|
| 1DOF / rigid       | 16.1°         | 18.8°               |
| 1DOF / nonrigid    | 20.6°         | 22.6°               |
| 2DOF / rigid       | 75.1°         | 67.7°               |
| 2DOF / nonrigid    | 70.8°         | 68.9°               |
| 3DOF / rigid       | 55.8°         | 57.8°               |
| 3DOF / nonrigid    | 56.8°         | 59.0°               |

For reference, on DFJIMU (robotic manipulator, 1-DOF only) the model achieves ~10° RMSE.
On simulated validation data: a few degrees.

## Root Cause 1: No multi-DOF inter-segment joints in training data

The training data was generated with `train_step1_mujoco_diverse.py`. Every non-anchor
segment uses `type="hinge"` plus a residual hinge capped at +/-7.5 degrees. The anchor
segment has a 3-DOF ball joint, but this is the root-to-world joint, not an
inter-segment joint.

Measured off-axis rotation in training pairs: mean ~2.5 degrees, max ~9 degrees.
The inter-segment dynamics are effectively 1-DOF.

DIODEM joint types:
- ARM (exp01-05): seg1-seg2 = 3-DOF spherical, seg2-seg3/seg3-seg4/seg4-seg5 = 1-DOF hinge
- GAIT (exp06-11): seg5-seg1 = 1-DOF hinge, seg1-seg2/seg2-seg3 = 2-DOF saddle, seg3-seg4 = 1-DOF hinge

No 2-DOF or 3-DOF inter-segment joints exist in the training data. This is the primary
cause of the catastrophic failure on 2-DOF (66-75°) and 3-DOF (45-57°) pairs.

Training data DOF distribution:
- 12000 files per anchor position (seg2, seg3, seg4, seg5)
- Non-anchor segments: always 1-DOF hinge + residual hinge
- `ja_rsaddle` attribute is always zero (no saddle joints generated)

## Root Cause 2: No static/quasi-static motions in training data

Training data relative angle range distribution (1-DOF pairs, excluding anchor):
- 0-5° (static):    0.0%
- 5-30° (slow):     1.9%
- 30-100° (moderate): 16.9%
- 100-180° (large):  81.2%

DIODEM 1-DOF distribution:
- 0-5° (static):    41.0%
- 5-30° (slow):     4.1%
- 30-100° (moderate): 30.9%
- 100-180° (large):  24.0%

41% of DIODEM motions are near-static. The model has never seen static data during
training, resulting in ~21.6° RMSE on static motions (the model cannot hold a stable
orientation estimate without active motion).

## Root Cause 3: Extreme angular velocities out of distribution

Catastrophic 1-DOF failures (55-79° RMSE) were found exclusively on exp03/exp04
seg2_seg3 pairs. These are fast, vigorous motions:

| Entry                           | gyr_p std | gyr_p max  | RMSE  |
|---------------------------------|-----------|------------|-------|
| exp03/mot04/seg2_seg3 (FAIL)    | 3.4 rad/s | 28.4 rad/s | 76.6° |
| exp04/mot05/seg2_seg3 (FAIL)    | 3.0 rad/s | 18.3 rad/s | 78.0° |
| exp01/mot05/seg2_seg3 (OK)      | 1.6 rad/s | 10.4 rad/s | 5.4°  |

Training gyr distribution: p99 = 15.4 rad/s, max = 23.7 rad/s, std max = 3.1.
The failing entries have gyr_std at or beyond the training distribution edge.

## Root Cause 4: No IMU motion artifacts in training

Training data folder is `data_mogen_100_noart` — generated without motion artifacts
(`imu_motion_artifacts=False`). DIODEM has both rigid and nonrigid (foam-mounted) IMUs.

Nonrigid adds ~5-10° RMSE consistently across all conditions. This is a pure training
data gap — the generation script supports motion artifacts (`_sample_imu_stiffness_damping`)
but they were not enabled.

## Where the model actually works well

On 1-DOF, rigid, moderate motion (30-100° angle range), the model achieves:

| Pair          | Chain position | Median RMSE |
|---------------|----------------|-------------|
| ARM/seg3_seg4 | middle         | 2.2°        |
| ARM/seg4_seg5 | chain end      | 2.5°        |
| GAIT/seg3_seg4| middle         | 2.2°        |
| GAIT/seg5_seg1| chain end      | 2.6°        |

These are excellent results, comparable to simulated validation performance.

## Verified non-issues

### Gravity in accelerometer
MuJoCo's native accelerometer sensor DOES include gravity. Verified by running a
standstill simulation with mogen: acc norm converges to ~9.88 m/s^2 (gravity).
The t=0 zero reading is a simulation initialization artifact only.

### Units and conventions
- Accelerometer: m/s^2 in both training and DIODEM
- Gyroscope: rad/s in both
- Quaternions: wxyz (scalar-first) in both, confirmed from diodem source (`_stack_from_df(omc, seg + "_quat_", "wxyz")`)
- `qmt.qrel(q1, q2)` computes `inv(q1) * q2`, matching the training target `conj(q_seg0) * q_seg1`

### Chain effects (2-segment vs full chain simulation)
Compared signal statistics and model performance by chain position (anchor=seg2):
- seg3 (2 downstream): acc_std=9.1, gyr_std=1.05
- seg4 (1 downstream): acc_std=10.1, gyr_std=1.24
- seg5 (0 downstream): acc_std=11.6, gyr_std=1.40

Downstream mass dampens motion, but model performance is similar across positions
(2.2-2.6° median on moderate 1-DOF). A 2-segment simulation would be sufficient for
the physics, but a full chain provides 3x data efficiency (3 pairs per simulation).

## Training data generation details

- Script: `train_step1_mujoco_diverse.py` (confirmed, not the mogen version)
- 48000 sequences, 12 motion configs (CONFIG_NAMES_DIVERSE), 4 anchor positions
- Sampling rates: uniform across [40, 60, 80, 100, 120, 140, 160, 180, 200] Hz
  (only ~11% at 100 Hz, but model receives dt as feature)
- Model `rnno_v6_mogen` has 7 features/segment: acc(3) + gyr(3) + dt(1)
- Segments: seg2, seg3, seg4, seg5 (4-segment chain)
- Pairs: seg2-seg3, seg3-seg4, seg4-seg5

## Recommended fixes

1. Add multi-DOF inter-segment joints (ball for 3-DOF, constrained ball / two hinges
   for 2-DOF) to the chain builder — addresses the 2-DOF/3-DOF failure
2. Add static/quasi-static motions to the training distribution (standstill_prob,
   very low velocity configs) — addresses the 21° static error
3. Enable motion artifacts (`imu_motion_artifacts=True`) — addresses the nonrigid gap
4. Fix sampling rate to 100 Hz and drop dt feature — simplifies the model and ensures
   all training data matches the evaluation regime
5. Keep the full 4-segment chain for 3x data efficiency per simulation
