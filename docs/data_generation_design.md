# Training Data Generation Design

Design rationale for `train_step1_mujoco_diverse.py`. See `diodem_failure_analysis.md`
for the analysis that motivated these choices.

## Goal

Generate simulated IMU data with maximum variability so the trained model
generalizes to real-world datasets (DIODEM, DFJIMU) without task-specific tuning.

## Chain structure

4-segment chain (seg2-seg5) simulated in MuJoCo with gravity. One segment is
the "anchor" (root with 6-DOF free joint), the others branch left/right with
inter-segment joints. The anchor rotates through all 4 positions across files,
giving 3 segment pairs per simulation for data efficiency.

## Joint DOF variety

Each non-anchor segment is randomly assigned 1-DOF (hinge), 2-DOF (two
perpendicular hinges), or 3-DOF (ball joint) with distribution 25%/25%/50%.
3-DOF is overrepresented because it is the most general case — a model that
handles arbitrary rotation should generalize to constrained joints more easily
than the reverse.

- 1-DOF: single hinge, driven by MuJoCo position actuator
- 2-DOF: two perpendicular hinges, both driven by position actuators
- 3-DOF: ball joint, tracked via spring/damper (`qpos_spring`), no actuator

Joint axes are randomized per sequence (`_random_perpendicular_axes`).

## Motion configs (16 presets)

Presets are cycled round-robin across files. Each preset defines motion
parameters for hinge, free, and cor_slide joints.

| Config | Purpose | Key parameters |
|---|---|---|
| standard | Baseline diverse motion | vel 0.1-3.0, keyframe 0.05-0.30s |
| expFast | Fast exploratory | vel 0.1-PI, sigmoid ROM |
| expSlow | Slow exploratory | vel 0.1-1.0, sigmoid ROM |
| hinUndHer | Back-and-forth oscillation | delta_ang_min=0.5, 5 CDF bins |
| verySlow | Very low velocity | vel 0.017-0.524 |
| langsam | Moderate-slow | vel 0.1-2.0 |
| expSlow-S | Gait-like (slow trunk) | Slow free joint, active hinges |
| standard-S | Gait-like (slow trunk) | Slow free joint, fast hinges |
| verySlow-S+ | Ultra-slow gait | Minimal trunk + slow limbs |
| cyclic-fast | Fast oscillation, tight ROM | rom_halfsize=0.5 |
| cyclic-slow | Slow oscillation, tight ROM | rom_halfsize=0.35 |
| slow-standstills | Burst-pause pattern | standstill_prob=0.15 |
| fast-explosive | High angular velocity | vel 1.0-2PI, low damping |
| rom-small | Small range of motion | rom_halfsize=0.3 (~34 deg total) |
| rom-medium | Medium range of motion | rom_halfsize=0.75 (~86 deg total) |
| rom-large | Large range of motion | rom_halfsize=1.25 (~143 deg total) |

## Standstill periods

Most configs have `standstill_prob=0.1` which injects random pauses (0.5-3s)
at keyframe transitions. This teaches the model to hold a stable orientation
estimate during stillness. Sequences always start with active motion (the
probability that ALL joints simultaneously standstill at t=0 is negligible),
and the 1-second warmup is trimmed anyway.

Configs without standstill: verySlow, verySlow-S+ (already near-static),
fast-explosive (pure high-velocity).

## Warmup

The first 100 timesteps (1 second at 100 Hz) are simulated but discarded from
the output. This ensures the MuJoCo accelerometer reads gravity correctly from
the first recorded timestep (without warmup, acc[0] = 0 due to simulation
initialization).

## Randomized parameters per sequence

- Anchor position (which segment is root): cycles 0-3
- Body positions: uniform within POS_BOUNDS per segment
- Joint damping: uniform [0.5, 10.0] per segment
- Joint axes: random perpendicular pairs per segment
- DOF type: 25%/25%/50% (1/2/3-DOF) per non-anchor segment
- Sampling rate: fixed 100 Hz
- Motion config: cycled round-robin from 16 presets
- Per-sequence RNG seed: deterministic from global seed

## Sampling rate

Fixed at 100 Hz. All evaluation datasets (DIODEM, DFJIMU) are resampled to
100 Hz, and the model does not receive dt as a feature.

## What is NOT varied (potential future work)

- IMU motion artifacts (non-rigid mounting): supported by the script
  (`imu_motion_artifacts` flag) but currently disabled
- Segment geometry (mass, size): fixed across all sequences
- IMU placement orientation: fixed relative to segment body
- Magnetometer data: not simulated
