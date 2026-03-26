"""Generate training data for a 4-segment chain using mogen.

Equivalent to:
  python train_step1_generateData_v2.py 65536 ring_data --mot-art --dof-configuration "['111']"

Usage:
  python train_step1_mujoco.py 65536 ring_data
  python train_step1_mujoco.py 64 test_output --seed 1
  python train_step1_mujoco.py 64 test_output --format pkl   # legacy pickle output
"""

import multiprocessing
import pickle
from pathlib import Path

import fire
import mujoco
import numpy as np
from tqdm import tqdm


class _MjWarning(Exception):
    """Raised from MuJoCo warning callback to abort unstable simulations."""
    pass


def _raise_on_warning(msg):
    raise _MjWarning(msg)

import mogen
from mogen.generator import _extract_joint_info
from mogen.simulate import simulate
from mogen.trajectory import generate_q_ref

# ── Constants ────────────────────────────────────────────────────────────────

SEGMENTS = ["seg2", "seg3", "seg4", "seg5"]
IMUS = ["imu2", "imu3", "imu4", "imu5"]
DEFAULT_SAMPLING_RATES = [40, 60, 80, 100, 120, 140, 160, 180, 200]
# Segment geometry (half-sizes for MuJoCo box geoms)
SEG_HALF_SIZE = "0.1 0.025 0.025"
SEG_MASS = 1.0
IMU_HALF_SIZE = "0.025 0.015 0.01"
IMU_MASS = 0.1
IMU_Z_OFFSET = 0.035
# Geom center offset along the segment (segment length = 0.2, geom centered)
SEG_GEOM_OFFSET = 0.1
# Default link offset along x-axis
CHAIN_OFFSET = 0.2

# Position randomization bounds (from original ring XML)
POS_BOUNDS = {
    "seg2": (np.array([0.15, -0.05, -0.05]), np.array([0.35, 0.05, 0.05])),
    "seg3": (np.array([0.0, -0.05, -0.05]), np.array([0.35, 0.05, 0.05])),
    "seg4": (np.array([0.0, -0.05, -0.05]), np.array([0.35, 0.05, 0.05])),
    "seg5": (np.array([0.0, -0.05, -0.05]), np.array([0.35, 0.05, 0.05])),
}

DEFAULT_HINGE_DAMPING = 3.0

# PD gains for position actuators (from ring's finalize_fns.py)
_P_ROT = 100.0   # rotational joints (hinge)
_P_POS = 250.0   # translational joints (slide, free joint position)

# IMU motion artifact stiffness/damping (from ring's motion_artifacts.py)
IMU_RIGID_ROT_STIFFNESS = 200.0
IMU_RIGID_POS_STIFFNESS = 20000.0
IMU_RIGID_DAMP_RATIO = 0.2  # damping = ratio * stiffness

IMU_FLEX_ROT_STIFFNESS = (0.2, 10.0)    # log-uniform range
IMU_FLEX_POS_STIFFNESS = (25.0, 1000.0)  # log-uniform range
IMU_FLEX_DAMP_RATIO = (0.05, 0.5)        # log-uniform range

# Motion config names
CONFIG_NAMES_DEFAULT = ["standard", "expSlow", "expFast", "hinUndHer"]
CONFIG_NAMES_DIVERSE = [
    "standard", "expFast", "expSlow", "hinUndHer",  # baseline
    "verySlow", "langsam",                           # slow gap
    "expSlow-S", "standard-S",                       # gait-like trunk
    "verySlow-S+",                                   # ultra-slow gait
    "cyclic-fast", "cyclic-slow",                    # periodic
    "slow-standstills",                              # burst-pause
]


# ── Axis helpers (rr_imp joint) ──────────────────────────────────────────────


def _random_unit_quaternion(rng):
    """Uniform random unit quaternion (scalar-first: w, x, y, z)."""
    u1, u2, u3 = rng.uniform(0, 1, 3)
    return np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])


def _rotate_vec_by_quat(vec, q):
    """Rotate vector by unit quaternion (scalar-first: w, x, y, z)."""
    w, x, y, z = q
    u = np.array([x, y, z])
    t = 2 * np.cross(u, vec)
    return vec + w * t + np.cross(u, t)


def _random_perpendicular_axes(rng):
    """Generate random perpendicular primary/residual axes (matches ring's rr_imp)."""
    phi = rng.uniform(0, 2 * np.pi)
    pri = np.array([0.0, 0.0, 1.0])
    res = np.array([np.cos(phi), np.sin(phi), 0.0])
    q = _random_unit_quaternion(rng)
    pri = _rotate_vec_by_quat(pri, q)
    res = _rotate_vec_by_quat(res, q)
    return pri, res


# ── Motion Config Mapping ────────────────────────────────────────────────────


def _make_motion_presets():
    """Map motion config names to mogen motion type dicts."""
    import math

    PI = math.pi

    # Helper: create a standard cor_slide config
    def _cor(vel_max=0.5, bins_min=1, bins_max=3):
        return mogen.HingeMotion(
            vel_range=(0.00001, vel_max), pos_range=(-0.4, 0.4),
            keyframe_interval=(0.2, 2.0), initial_pos=0.0,
            range_of_motion=False, randomized_interpolation=True,
            cdf_bins_min=bins_min, cdf_bins_max=bins_max,
        )

    # Helper: create a standard residual hinge config
    def _res(bins_min=1, bins_max=3):
        return mogen.HingeMotion(
            vel_range=(0.1, 5.0), keyframe_interval=(0.05, 0.4),
            randomized_interpolation=True, cdf_bins_min=bins_min, cdf_bins_max=bins_max,
        )

    return {
        # ── Group 1: original 4 configs ──
        "standard": {
            "hinge": mogen.HingeMotion(
                vel_range=(0.1, 3.0), keyframe_interval=(0.05, 0.30),
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=5,
            ),
            "residual_hinge": _res(1, 5),
            "free": mogen.FreeMotion(
                ang_vel_range=(0.1, 3.0), keyframe_interval=(0.05, 0.30),
                translation="track", lin_vel_range=(0.01, 0.1), pos_range=(-0.5, 0.5),
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=5,
            ),
            "cor_slide": _cor(0.5, 1, 5),
        },
        "expSlow": {
            "hinge": mogen.HingeMotion(
                vel_range=(0.1, 1.0), keyframe_interval=(0.75, 3.0),
                delta_ang_min=0.4, range_of_motion_method="sigmoid",
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=5,
            ),
            "residual_hinge": _res(1, 5),
            "free": mogen.FreeMotion(
                ang_vel_range=(0.1, 1.0), keyframe_interval=(0.75, 3.0),
                translation="track", lin_vel_range=(0.01, 0.1), pos_range=(-0.5, 0.5),
                delta_ang_min=0.4, randomized_interpolation=True,
                cdf_bins_min=1, cdf_bins_max=5,
            ),
            "cor_slide": _cor(0.3, 1, 5),
        },
        "expFast": {
            "hinge": mogen.HingeMotion(
                vel_range=(0.1, PI), keyframe_interval=(0.4, 1.1),
                delta_ang_min=PI / 3, delta_ang_max=11 * PI / 18,
                range_of_motion_method="sigmoid",
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=3,
            ),
            "residual_hinge": _res(1, 3),
            "free": mogen.FreeMotion(
                ang_vel_range=(0.1, PI), keyframe_interval=(0.4, 1.1),
                translation="track", lin_vel_range=(0.01, 0.1), pos_range=(-0.5, 0.5),
                delta_ang_min=PI / 3, delta_ang_max=11 * PI / 18,
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=3,
            ),
            "cor_slide": _cor(0.5, 1, 3),
        },
        "hinUndHer": {
            "hinge": mogen.HingeMotion(
                vel_range=(0.1, 3.0), keyframe_interval=(0.3, 1.5),
                delta_ang_min=0.5, randomized_interpolation=True, cdf_bins_min=5,
            ),
            "residual_hinge": _res(5),
            "free": mogen.FreeMotion(
                ang_vel_range=(0.1, 3.0), keyframe_interval=(0.3, 1.5),
                translation="track", lin_vel_range=(0.01, 0.1), pos_range=(-0.5, 0.5),
                delta_ang_min=0.5, randomized_interpolation=True, cdf_bins_min=5,
            ),
            "cor_slide": _cor(0.5, 5),
        },

        # ── Group 2: fill velocity gaps ──
        "verySlow": {
            "hinge": mogen.HingeMotion(
                vel_range=(0.017, 0.524), keyframe_interval=(1.5, 5.0),
                delta_ang_min=0.349, cdf_bins_min=1, cdf_bins_max=3,
            ),
            "residual_hinge": mogen.HingeMotion(
                vel_range=(0.017, 0.524), keyframe_interval=(1.5, 5.0),
                cdf_bins_min=1, cdf_bins_max=3,
            ),
            "free": mogen.FreeMotion(
                ang_vel_range=(0.017, 0.175), keyframe_interval=(1.5, 5.0),
                translation="track", lin_vel_range=(0.001, 0.3), pos_range=(-0.5, 0.5),
                cdf_bins_min=1, cdf_bins_max=3,
            ),
            "cor_slide": _cor(0.3, 1, 3),
        },
        "langsam": {
            "hinge": mogen.HingeMotion(
                vel_range=(0.1, 2.0), keyframe_interval=(0.2, 1.25),
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=3,
            ),
            "residual_hinge": _res(1, 3),
            "free": mogen.FreeMotion(
                ang_vel_range=(0.1, 2.0), keyframe_interval=(0.2, 1.25),
                translation="track", lin_vel_range=(0.01, 0.1), pos_range=(-0.5, 0.5),
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=3,
            ),
            "cor_slide": _cor(0.5, 1, 3),
        },

        # ── Group 3: gait-like (constrained trunk) ──
        "expSlow-S": {
            "hinge": mogen.HingeMotion(
                vel_range=(0.1, 1.0), keyframe_interval=(0.75, 3.0),
                delta_ang_min=0.4, range_of_motion_method="sigmoid",
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=5,
            ),
            "residual_hinge": _res(1, 5),
            "free": mogen.FreeMotion(
                ang_vel_range=(0.1, 0.2), keyframe_interval=(1.5, 15.0),
                translation="track", lin_vel_range=(0.001, 0.1), pos_range=(-0.5, 0.5),
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=5,
            ),
            "cor_slide": _cor(0.1, 1, 5),
        },
        "standard-S": {
            "hinge": mogen.HingeMotion(
                vel_range=(0.1, 3.0), keyframe_interval=(0.05, 0.30),
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=5,
            ),
            "residual_hinge": _res(1, 5),
            "free": mogen.FreeMotion(
                ang_vel_range=(0.1, 0.2), keyframe_interval=(1.5, 15.0),
                translation="track", lin_vel_range=(0.001, 0.1), pos_range=(-0.5, 0.5),
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=5,
            ),
            "cor_slide": _cor(0.1, 1, 5),
        },

        # ── Group 4: custom configs ──
        # Ultra-slow gait: very slow limbs + maximally constrained trunk
        "verySlow-S+": {
            "hinge": mogen.HingeMotion(
                vel_range=(0.017, 0.524), keyframe_interval=(1.5, 5.0),
                delta_ang_min=0.349, cdf_bins_min=1, cdf_bins_max=3,
            ),
            "residual_hinge": mogen.HingeMotion(
                vel_range=(0.017, 0.524), keyframe_interval=(1.5, 5.0),
                cdf_bins_min=1, cdf_bins_max=3,
            ),
            "free": mogen.FreeMotion(
                ang_vel_range=(0.017, 0.1), keyframe_interval=(1.5, 15.0),
                translation="track", lin_vel_range=(0.001, 0.05), pos_range=(-0.5, 0.5),
                cdf_bins_min=1, cdf_bins_max=3,
            ),
            "cor_slide": _cor(0.05, 1, 3),
        },
        # Cyclic-fast: tight ROM oscillation at high speed
        "cyclic-fast": {
            "hinge": mogen.HingeMotion(
                vel_range=(0.1, PI), keyframe_interval=(0.4, 1.1),
                delta_ang_min=PI / 6, delta_ang_max=55 * PI / 180,
                rom_halfsize=0.5, range_of_motion_method="sigmoid",
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=3,
            ),
            "residual_hinge": _res(1, 3),
            "free": mogen.FreeMotion(
                ang_vel_range=(0.1, PI), keyframe_interval=(0.4, 1.1),
                translation="track", lin_vel_range=(0.01, 0.1), pos_range=(-0.5, 0.5),
                delta_ang_min=PI / 6, delta_ang_max=55 * PI / 180,
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=3,
            ),
            "cor_slide": _cor(0.5, 1, 3),
        },
        # Cyclic-slow: tight ROM oscillation at low speed
        "cyclic-slow": {
            "hinge": mogen.HingeMotion(
                vel_range=(0.1, 1.0), keyframe_interval=(0.75, 3.0),
                delta_ang_min=0.4, rom_halfsize=0.35,
                range_of_motion_method="sigmoid",
                randomized_interpolation=True, cdf_bins_min=1, cdf_bins_max=5,
            ),
            "residual_hinge": _res(1, 5),
            "free": mogen.FreeMotion(
                ang_vel_range=(0.1, 1.0), keyframe_interval=(0.75, 3.0),
                translation="track", lin_vel_range=(0.01, 0.1), pos_range=(-0.5, 0.5),
                delta_ang_min=0.4, randomized_interpolation=True,
                cdf_bins_min=1, cdf_bins_max=5,
            ),
            "cor_slide": _cor(0.3, 1, 5),
        },
        # Slow with burst-pause: very slow motion with standstills
        "slow-standstills": {
            "hinge": mogen.HingeMotion(
                vel_range=(0.017, 0.524), keyframe_interval=(1.5, 5.0),
                delta_ang_min=0.349, cdf_bins_min=1, cdf_bins_max=3,
                standstill_prob=0.15, standstill_duration=(0.5, 3.0),
            ),
            "residual_hinge": mogen.HingeMotion(
                vel_range=(0.017, 0.524), keyframe_interval=(1.5, 5.0),
                cdf_bins_min=1, cdf_bins_max=3,
                standstill_prob=0.15, standstill_duration=(0.5, 3.0),
            ),
            "free": mogen.FreeMotion(
                ang_vel_range=(0.017, 0.175), keyframe_interval=(1.5, 5.0),
                translation="track", lin_vel_range=(0.001, 0.3), pos_range=(-0.5, 0.5),
                cdf_bins_min=1, cdf_bins_max=3,
                standstill_prob=0.15, standstill_duration=(0.5, 3.0),
            ),
            "cor_slide": _cor(0.3, 1, 3),
        },
    }


def _build_motion_dict(preset, model):
    """Build per-joint motion config dict from a preset and Model."""
    motion = {}
    free_cfg = preset["free"]  # FreeMotion preset (used for ball + root slides)
    for name, typ in model.joint_type.items():
        if typ == "free":
            motion[name] = free_cfg
        elif typ == "ball" and name == "root_ball":
            motion[name] = mogen.BallMotion(
                ang_vel_range=free_cfg.ang_vel_range,
                keyframe_interval=free_cfg.keyframe_interval,
                delta_ang_min=free_cfg.delta_ang_min,
                delta_ang_max=free_cfg.delta_ang_max,
                randomized_interpolation=free_cfg.randomized_interpolation,
                cdf_bins_min=free_cfg.cdf_bins_min,
                cdf_bins_max=free_cfg.cdf_bins_max,
            )
        elif typ == "slide" and name.startswith("root_"):
            motion[name] = mogen.HingeMotion(
                vel_range=free_cfg.lin_vel_range,
                pos_range=free_cfg.pos_range,
                keyframe_interval=free_cfg.keyframe_interval,
                range_of_motion=False,
                randomized_interpolation=free_cfg.randomized_interpolation,
                cdf_bins_min=free_cfg.cdf_bins_min,
                cdf_bins_max=free_cfg.cdf_bins_max,
            )
        elif typ == "slide" and name.startswith("cor_"):
            motion[name] = preset["cor_slide"]
        elif typ == "hinge":
            if name.endswith("_res"):
                motion[name] = preset["residual_hinge"]
            else:
                motion[name] = preset["hinge"]
    return motion


# ── MJCF Model Building ─────────────────────────────────────────────────────


def _fmt(arr):
    return f"{arr[0]} {arr[1]} {arr[2]}"


def _imu_joint_xml(imu, indent, rot_stiffness, rot_damping, pos_stiffness, pos_damping):
    """Generate ball + 3 slide joint XML lines for a compliant IMU body."""
    lines = []
    lines.append(
        f'{indent}<joint name="{imu}_ball" type="ball"'
        f' stiffness="{rot_stiffness}" damping="{rot_damping}"/>'
    )
    for axis_name, axis_vec in [("sx", "1 0 0"), ("sy", "0 1 0"), ("sz", "0 0 1")]:
        lines.append(
            f'{indent}<joint name="{imu}_{axis_name}" type="slide"'
            f' axis="{axis_vec}" stiffness="{pos_stiffness}" damping="{pos_damping}"/>'
        )
    return lines


def _build_branch(seg_indices, body_positions, hinge_dampings, joint_axes, reversed_dir, indent, imu_stiffness_damping=None):
    """Recursively build nested body XML for a chain branch."""
    if not seg_indices:
        return []

    idx = seg_indices[0]
    remaining = seg_indices[1:]

    seg = SEGMENTS[idx]
    imu = IMUS[idx]
    pos = body_positions[seg].copy()
    if reversed_dir:
        pos[0] = -abs(pos[0])

    damp = hinge_dampings[seg]
    pri_axis, res_axis = joint_axes[seg]
    geom_x = -SEG_GEOM_OFFSET if reversed_dir else SEG_GEOM_OFFSET
    imu_x = geom_x

    lines = []
    lines.append(f'{indent}<body name="{seg}" pos="{_fmt(pos)}">')
    lines.append(
        f'{indent}  <joint name="j_{seg}_pri" type="hinge"'
        f' axis="{_fmt(pri_axis)}" damping="{damp}"/>'
    )
    lines.append(
        f'{indent}  <joint name="j_{seg}_res" type="hinge"'
        f' axis="{_fmt(res_axis)}" range="-7.5 7.5" damping="{damp}"/>'
    )
    lines.append(
        f'{indent}  <geom type="box" size="{SEG_HALF_SIZE}"'
        f' mass="{SEG_MASS}" pos="{geom_x} 0 0"/>'
    )
    lines.append(f'{indent}  <body name="{imu}" pos="{imu_x} 0 {IMU_Z_OFFSET}">')
    if imu_stiffness_damping and imu in imu_stiffness_damping:
        rs, rd, ps, pd = imu_stiffness_damping[imu]
        lines.extend(_imu_joint_xml(imu, indent + "    ", rs, rd, ps, pd))
    lines.append(
        f'{indent}    <geom type="box" size="{IMU_HALF_SIZE}" mass="{IMU_MASS}"/>'
    )
    lines.append(f'{indent}    <site name="site_{imu}" pos="0 0 0"/>')
    lines.append(f"{indent}  </body>")

    # Nested children
    lines.extend(
        _build_branch(remaining, body_positions, hinge_dampings, joint_axes, reversed_dir, indent + "  ", imu_stiffness_damping)
    )

    lines.append(f"{indent}</body>")
    return lines


def build_chain_xml(anchor_idx, dt, body_positions=None, hinge_dampings=None, joint_axes=None, cor=True, imu_stiffness_damping=None):
    """Build MJCF XML for a 4-segment chain with the given anchor as root.

    Args:
        anchor_idx: Index (0-3) into SEGMENTS for the root body (free joint).
        dt: Simulation timestep.
        body_positions: Dict mapping segment name to (x,y,z) position offset.
        hinge_dampings: Dict mapping segment name to damping value.
        joint_axes: Dict mapping segment name to (pri_axis, res_axis) tuples.
        cor: If True, add a pivot body with the free joint and 3 COR slide
            joints on the anchor body (9-DOF root matching ring's cor=True).
        imu_stiffness_damping: Dict mapping IMU name to
            (rot_stiffness, rot_damping, pos_stiffness, pos_damping). When
            provided, adds ball + 3 slide joints to IMU bodies for motion
            artifacts. When None, IMUs are rigidly attached.

    Returns:
        MJCF XML string.
    """
    if body_positions is None:
        body_positions = {s: np.array([CHAIN_OFFSET, 0.0, 0.0]) for s in SEGMENTS}
    if hinge_dampings is None:
        hinge_dampings = {s: DEFAULT_HINGE_DAMPING for s in SEGMENTS}
    if joint_axes is None:
        rng = np.random.default_rng(0)
        joint_axes = {s: _random_perpendicular_axes(rng) for s in SEGMENTS}

    # Left branch: segments before anchor (reversed direction)
    left = list(range(anchor_idx - 1, -1, -1))
    # Right branch: segments after anchor (original direction)
    right = list(range(anchor_idx + 1, 4))

    anchor_seg = SEGMENTS[anchor_idx]
    anchor_imu = IMUS[anchor_idx]
    anchor_pos = body_positions[anchor_seg]

    # Track joints for actuator generation
    actuated_joints = []  # (name, type) for hinge/slide joints needing actuators

    lines = [
        "<mujoco>",
        f'  <option timestep="{dt}" gravity="0 0 -9.81" integrator="implicitfast">',
        '    <flag contact="disable"/>',
        '  </option>',
        "  <worldbody>",
    ]

    if cor:
        # Pivot body holds root translation (3 slides) + rotation (ball);
        # anchor body gets 3 COR slide joints for moving center of rotation.
        lines.append(f'    <body name="cor_pivot" pos="{_fmt(anchor_pos)}">')
        lines.append('      <joint name="root_x" type="slide" axis="1 0 0" damping="25"/>')
        lines.append('      <joint name="root_y" type="slide" axis="0 1 0" damping="25"/>')
        lines.append('      <joint name="root_z" type="slide" axis="0 0 1" damping="25"/>')
        lines.append('      <joint name="root_ball" type="ball" stiffness="100" damping="5"/>')
        actuated_joints.extend([("root_x", "slide"), ("root_y", "slide"), ("root_z", "slide")])
        lines.append('      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>')
        lines.append(f'      <body name="{anchor_seg}" pos="0 0 0">')
        lines.append('        <joint name="cor_x" type="slide" axis="1 0 0" damping="5.0"/>')
        lines.append('        <joint name="cor_y" type="slide" axis="0 1 0" damping="5.0"/>')
        lines.append('        <joint name="cor_z" type="slide" axis="0 0 1" damping="5.0"/>')
        actuated_joints.extend([("cor_x", "slide"), ("cor_y", "slide"), ("cor_z", "slide")])
        lines.append(
            f'        <geom type="box" size="{SEG_HALF_SIZE}"'
            f' mass="{SEG_MASS}" pos="{SEG_GEOM_OFFSET} 0 0"/>'
        )
        lines.append(f'        <body name="{anchor_imu}" pos="{SEG_GEOM_OFFSET} 0 {IMU_Z_OFFSET}">')
        if imu_stiffness_damping and anchor_imu in imu_stiffness_damping:
            rs, rd, ps, pd = imu_stiffness_damping[anchor_imu]
            lines.extend(_imu_joint_xml(anchor_imu, "          ", rs, rd, ps, pd))
        lines.append(f'          <geom type="box" size="{IMU_HALF_SIZE}" mass="{IMU_MASS}"/>')
        lines.append(f'          <site name="site_{anchor_imu}" pos="0 0 0"/>')
        lines.append("        </body>")

        branch_indent = "        "
    else:
        lines.append(f'    <body name="{anchor_seg}" pos="{_fmt(anchor_pos)}">')
        lines.append('      <joint name="root_x" type="slide" axis="1 0 0" damping="25"/>')
        lines.append('      <joint name="root_y" type="slide" axis="0 1 0" damping="25"/>')
        lines.append('      <joint name="root_z" type="slide" axis="0 0 1" damping="25"/>')
        lines.append('      <joint name="root_ball" type="ball" stiffness="100" damping="5"/>')
        actuated_joints.extend([("root_x", "slide"), ("root_y", "slide"), ("root_z", "slide")])
        lines.append(
            f'      <geom type="box" size="{SEG_HALF_SIZE}"'
            f' mass="{SEG_MASS}" pos="{SEG_GEOM_OFFSET} 0 0"/>'
        )
        lines.append(f'      <body name="{anchor_imu}" pos="{SEG_GEOM_OFFSET} 0 {IMU_Z_OFFSET}">')
        if imu_stiffness_damping and anchor_imu in imu_stiffness_damping:
            rs, rd, ps, pd = imu_stiffness_damping[anchor_imu]
            lines.extend(_imu_joint_xml(anchor_imu, "        ", rs, rd, ps, pd))
        lines.append(f'        <geom type="box" size="{IMU_HALF_SIZE}" mass="{IMU_MASS}"/>')
        lines.append(f'        <site name="site_{anchor_imu}" pos="0 0 0"/>')
        lines.append("      </body>")

        branch_indent = "      "

    # Collect hinge joints from branch segments
    for idx in left + right:
        seg = SEGMENTS[idx]
        actuated_joints.append((f"j_{seg}_pri", "hinge"))
        actuated_joints.append((f"j_{seg}_res", "hinge"))

    # Left branch (reversed)
    lines.extend(
        _build_branch(left, body_positions, hinge_dampings, joint_axes, True, branch_indent, imu_stiffness_damping)
    )
    # Right branch (original direction)
    lines.extend(
        _build_branch(right, body_positions, hinge_dampings, joint_axes, False, branch_indent, imu_stiffness_damping)
    )

    if cor:
        lines.extend([
            "      </body>",
            "    </body>",
        ])
    else:
        lines.append("    </body>")

    lines.append("  </worldbody>")

    # Add position actuators for all PD-controlled joints
    lines.append("  <actuator>")
    for jnt_name, jnt_type in actuated_joints:
        kp = _P_POS if jnt_type == "slide" else _P_ROT
        lines.append(f'    <position joint="{jnt_name}" kp="{kp}"/>')
    lines.append("  </actuator>")

    # Add native accelerometer/gyro sensors for each IMU body
    lines.append("  <sensor>")
    for imu in IMUS:
        lines.append(f'    <accelerometer name="acc_{imu}" site="site_{imu}"/>')
        lines.append(f'    <gyro name="gyr_{imu}" site="site_{imu}"/>')
    lines.append("  </sensor>")

    lines.append("</mujoco>")

    return "\n".join(lines)


# ── Per-Sequence Generation ──────────────────────────────────────────────────


def _sample_imu_stiffness_damping(rng, prob_rigid=0.5, all_rigid_or_flex=False):
    """Sample per-IMU stiffness/damping for motion artifacts.

    Returns:
        Dict mapping IMU name to (rot_stiffness, rot_damping, pos_stiffness, pos_damping).
    """
    if all_rigid_or_flex:
        is_rigid = rng.random() < prob_rigid
        rigid_flags = {imu: is_rigid for imu in IMUS}
    else:
        rigid_flags = {imu: rng.random() < prob_rigid for imu in IMUS}

    result = {}
    for imu, rigid in rigid_flags.items():
        if rigid:
            rs = IMU_RIGID_ROT_STIFFNESS
            ps = IMU_RIGID_POS_STIFFNESS
            dr = IMU_RIGID_DAMP_RATIO
        else:
            rs = np.exp(rng.uniform(np.log(IMU_FLEX_ROT_STIFFNESS[0]), np.log(IMU_FLEX_ROT_STIFFNESS[1])))
            ps = np.exp(rng.uniform(np.log(IMU_FLEX_POS_STIFFNESS[0]), np.log(IMU_FLEX_POS_STIFFNESS[1])))
            dr = np.exp(rng.uniform(np.log(IMU_FLEX_DAMP_RATIO[0]), np.log(IMU_FLEX_DAMP_RATIO[1])))
        result[imu] = (rs, dr * rs, ps, dr * ps)
    return result


def _generate_one(
    anchor_idx,
    config_name,
    hz,
    seed,
    motion_presets,
    n_timesteps,
    randomize_positions=True,
    randomize_joint_params=True,
    imu_motion_artifacts=False,
    prob_rigid=0.5,
    all_rigid_or_flex=False,
    substeps=1,
):
    """Generate a single training sequence and return (X, y) dicts."""
    rng = np.random.default_rng(seed)

    dt = 1.0 / hz
    dt_phys = dt / substeps

    # Randomize body positions
    body_positions = {}
    for seg in SEGMENTS:
        if randomize_positions:
            lo, hi = POS_BOUNDS[seg]
            body_positions[seg] = rng.uniform(lo, hi)
        else:
            body_positions[seg] = np.array([CHAIN_OFFSET, 0.0, 0.0])

    # Randomize hinge dampings
    hinge_dampings = {}
    for seg in SEGMENTS:
        if randomize_joint_params:
            hinge_dampings[seg] = float(rng.uniform(1.0, 10.0))
        else:
            hinge_dampings[seg] = DEFAULT_HINGE_DAMPING

    # Generate random perpendicular axes per segment (rr_imp joint)
    joint_axes = {seg: _random_perpendicular_axes(rng) for seg in SEGMENTS}

    # Sample IMU stiffness/damping if motion artifacts enabled
    imu_sd = None
    if imu_motion_artifacts:
        imu_sd = _sample_imu_stiffness_damping(rng, prob_rigid, all_rigid_or_flex)

    # Build model (use smaller physics timestep when substeps > 1)
    xml = build_chain_xml(anchor_idx, dt_phys, body_positions, hinge_dampings, joint_axes, imu_stiffness_damping=imu_sd)
    model = mogen.Model(xml)

    # Motion config
    preset = motion_presets[config_name]
    motion = _build_motion_dict(preset, model)

    # Generate reference trajectory at output rate
    sensors = mogen.SensorConfig(imus=IMUS)
    control = mogen.ControlConfig(kinematic=False, substeps=substeps)
    joint_info = _extract_joint_info(model.mj_model)
    duration = n_timesteps * dt
    q_ref = generate_q_ref(motion, joint_info, duration, dt, rng)

    # Post-process residual trajectories: scale to ±7.5° and center
    for ji in joint_info:
        if ji["name"].endswith("_res"):
            adr = ji["qpos_adr"]
            q_ref[:, adr] = q_ref[:, adr] * (np.deg2rad(7.5) / np.pi)
            q_ref[:, adr] -= np.mean(q_ref[:, adr])

    # Forward simulation
    traj = simulate(model, q_ref, motion, sensors, control)

    body_quat_rec = traj.body_quat_array
    mj_model = model.mj_model

    # ── Build per-segment SegmentData ──
    from mogen import SegmentData

    seg_data = {}
    for i, seg in enumerate(SEGMENTS):
        imu = IMUS[i]
        acc = traj.imu[imu]["acc"].astype(np.float32) if traj.imu and imu in traj.imu else np.zeros((n_timesteps, 3), dtype=np.float32)
        gyr = traj.imu[imu]["gyr"].astype(np.float32) if traj.imu and imu in traj.imu else np.zeros((n_timesteps, 3), dtype=np.float32)

        # Body quaternion (body-to-world, matching ring convention)
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, seg)
        quat = body_quat_rec[:, bid].copy()

        dof = None
        ja = None
        if i > 0:  # non-root segments (seg3, seg4, seg5)
            pri_axis, _res_axis = joint_axes[seg]
            dof = 1
            ja = pri_axis.astype(np.float32)

        seg_data[seg] = SegmentData(
            acc=acc, gyr=gyr, q=quat.astype(np.float32),
            dof=dof, joint_axes=ja,
        )

    return np.float32(dt), seg_data


# ── Worker for multiprocessing ───────────────────────────────────────────────


_MAX_SUBSTEPS = 4


def _generate_one_safe(motion_presets, **kwargs):
    """Run _generate_one, retrying with more substeps on solver warnings."""
    substeps = 1
    while True:
        try:
            return _generate_one(motion_presets=motion_presets, substeps=substeps, **kwargs)
        except _MjWarning:
            substeps *= 2
            if substeps > _MAX_SUBSTEPS:
                raise


def _save_sample(i, dt, seg_data, output_dir, fmt):
    """Save a sample in the requested format."""
    from mogen import save_sample_h5

    if fmt == "h5":
        save_sample_h5(Path(output_dir) / f"seq{i}.h5", dt, seg_data)
    else:
        X, y = _seg_data_to_pickle(dt, seg_data)
        with open(Path(output_dir) / f"seq{i}.pkl", "wb") as f:
            pickle.dump((X, y), f)


def _seg_data_to_pickle(dt, seg_data):
    """Convert SegmentData dict back to the legacy (X, y) pickle format."""
    X = {"dt": dt}
    y = {}
    for seg, data in seg_data.items():
        imu = seg.replace("seg", "imu")
        X[imu] = {"acc": data.acc, "gyr": data.gyr}
        y[seg] = data.q
        if data.dof is not None:
            X[seg] = {
                "dof": data.dof,
                "joint_params": {"joint_axes": data.joint_axes},
            }
    return X, y


def _worker(args):
    """Generate and save a single sequence. Top-level function for pickling."""
    (
        i, anchor_idx, config_name, hz, seq_seed,
        n_timesteps, randomize_positions, randomize_joint_params,
        imu_motion_artifacts, prob_rigid, all_rigid_or_flex, output_dir, fmt,
    ) = args
    mujoco.set_mju_user_warning(_raise_on_warning)
    motion_presets = _make_motion_presets()
    dt, seg_data = _generate_one_safe(
        motion_presets=motion_presets,
        anchor_idx=anchor_idx,
        config_name=config_name,
        hz=hz,
        seed=seq_seed,
        n_timesteps=n_timesteps,
        randomize_positions=randomize_positions,
        randomize_joint_params=randomize_joint_params,
        imu_motion_artifacts=imu_motion_artifacts,
        prob_rigid=prob_rigid,
        all_rigid_or_flex=all_rigid_or_flex,
    )
    _save_sample(i, dt, seg_data, output_dir, fmt)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(
    size: int,
    output_path: str,
    config_preset: str = "diverse",
    seed: int = 1,
    sampling_rates: list = None,
    T: float = 150.0,
    randomize_positions: bool = True,
    randomize_joint_params: bool = True,
    workers: int = 1,
    imu_motion_artifacts: bool = False,
    prob_rigid: float = 0.5,
    all_rigid_or_flex: bool = False,
    format: str = "h5",
):
    """Generate training data using mogen with diverse motion configs.

    Args:
        size: Number of sequences to generate.
        output_path: Folder to save files (seq0.h5, seq1.h5, ...).
        config_preset: "diverse" (12 configs) or "default" (original 4).
        seed: Random seed.
        sampling_rates: List of Hz values to randomly sample from.
        T: Maximum trial length in seconds.
        randomize_positions: Randomize segment positions per sequence.
        randomize_joint_params: Randomize joint damping per sequence.
        workers: Number of parallel workers (default 1, no multiprocessing).
        imu_motion_artifacts: Add compliant spring-damper joints to IMU
            bodies, producing non-rigid sensor attachment (matches ring's
            --mot-art flag).
        prob_rigid: Probability that each IMU gets rigid (high-stiffness)
            springs instead of flexible ones. Default 0.5.
        all_rigid_or_flex: If True, all IMUs in a sequence share the same
            rigid/flex decision.
        format: Output format — "h5" (default, ring-compatible HDF5) or "pkl"
            (legacy pickle).
    """
    if format not in ("h5", "pkl"):
        raise ValueError(f"Unknown format {format!r}, expected 'h5' or 'pkl'")

    if config_preset == "diverse":
        config_names = CONFIG_NAMES_DIVERSE
    elif config_preset == "default":
        config_names = CONFIG_NAMES_DEFAULT
    else:
        raise ValueError(f"Unknown config_preset: {config_preset!r}")

    if sampling_rates is None:
        sampling_rates = DEFAULT_SAMPLING_RATES

    n_timesteps = int(min(sampling_rates) * T)
    n_anchors = len(SEGMENTS)
    n_configs = len(config_names)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute all job parameters (deterministic regardless of worker count)
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(size):
        anchor_idx = i % n_anchors
        config_name = config_names[(i // n_anchors) % n_configs]
        hz = float(rng.choice(sampling_rates))
        seq_seed = int(rng.integers(0, 2**63))
        jobs.append((
            i, anchor_idx, config_name, hz, seq_seed,
            n_timesteps, randomize_positions, randomize_joint_params,
            imu_motion_artifacts, prob_rigid, all_rigid_or_flex,
            str(output_dir), format,
        ))

    mujoco.set_mju_user_warning(_raise_on_warning)

    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            list(tqdm(
                pool.imap_unordered(_worker, jobs),
                total=size,
                desc=f"Generating ({workers} workers)",
            ))
    else:
        motion_presets = _make_motion_presets()
        for job in tqdm(jobs, desc="Generating"):
            (
                i, anchor_idx, config_name, hz, seq_seed,
                n_timesteps_, rp, rjp, ima, pr, arof, _od, _fmt,
            ) = job
            dt, seg_data = _generate_one_safe(
                motion_presets=motion_presets,
                anchor_idx=anchor_idx,
                config_name=config_name,
                hz=hz,
                seed=seq_seed,
                n_timesteps=n_timesteps_,
                randomize_positions=rp,
                randomize_joint_params=rjp,
                imu_motion_artifacts=ima,
                prob_rigid=pr,
                all_rigid_or_flex=arof,
            )
            _save_sample(i, dt, seg_data, output_dir, format)


if __name__ == "__main__":
    fire.Fire(main)
