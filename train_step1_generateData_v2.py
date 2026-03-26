from dataclasses import replace
import itertools
from typing import Optional

import fire
import jax
import jax.numpy as jnp
import numpy as np
import ring
from ring.extras.randomize_sys import randomize_anchors
from ring.sys_composer.morph_sys import _autodetermine_new_parents
from ring.sys_composer.morph_sys import identify_system
from ring.sys_composer.morph_sys import Node
from ring.utils import dict_union

sys_str = """
<x_xy model="lam4">
  <options dt="0.01" gravity="0.0 0.0 9.81"/>
  <worldbody>
    <body joint="free" name="seg2_4Seg" pos="0.2 0.0 0.0" pos_min="0.15 -0.05 -0.05" pos_max="0.35 0.05 0.05" damping="5.0 5.0 5.0 25.0 25.0 25.0">
      <geom pos="0.1 0.0 0.0" mass="1.0" color="dustin_exp_white" edge_color="black" type="box" dim="0.2 0.05 0.05"/>
      <geom pos="0.03 -0.05 0.0" mass="0.1" color="dustin_exp_white" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
      <geom pos="0.17 -0.05 0.0" mass="0.1" color="dustin_exp_white" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
      <omc pos="0.0 0.0 -0.02" name="seg2" pos_marker="2"/>
      <body joint="frozen" name="imu2_4Seg" pos="0.10000001 0.0 0.035" pos_min="0.049999997 -0.05 -0.05" pos_max="0.15000002 0.05 0.05">
        <geom mass="0.1" color="dustin_exp_orange" edge_color="black" type="box" dim="0.05 0.03 0.02"/>
        <omc pos="0.1 0.0 0.015" name="seg2" pos_marker="2"/>
      </body>
      <body joint="rr_imp" name="seg3_4Seg" pos="0.2 0.0 0.0" pos_min="0.0 -0.05 -0.05" pos_max="0.35000002 0.05 0.05" damping="3.0 3.0">
        <geom pos="0.1 0.0 0.0" mass="1.0" color="dustin_exp_blue" edge_color="black" type="box" dim="0.2 0.05 0.05"/>
        <geom pos="0.05 0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
        <geom pos="0.15 -0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
        <omc pos="0.0 0.0 -0.02" name="seg3" pos_marker="1"/>
        <body joint="frozen" name="imu3_4Seg" pos="0.099999994 0.0 0.035" pos_min="0.050000012 -0.05 -0.05" pos_max="0.15 0.05 0.05">
          <geom mass="0.1" color="dustin_exp_orange" edge_color="black" type="box" dim="0.05 0.03 0.02"/>
          <omc pos="0.1 0.0 0.015" name="seg3" pos_marker="1"/>
        </body>
        <body joint="rr_imp" name="seg4_4Seg" pos="0.20000002 0.0 0.0" pos_min="0.0 -0.05 -0.05" pos_max="0.35 0.05 0.05" damping="3.0 3.0">
          <geom pos="0.1 0.0 0.0" mass="1.0" color="dustin_exp_white" edge_color="black" type="box" dim="0.2 0.05 0.05"/>
          <geom pos="0.1 0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
          <geom pos="0.15 -0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
          <omc pos="0.0 0.0 -0.02" name="seg4" pos_marker="2"/>
          <body joint="frozen" name="imu4_4Seg" pos="0.100000024 0.0 0.035" pos_min="0.050000012 -0.05 -0.05" pos_max="0.14999998 0.05 0.05">
            <geom mass="0.1" color="dustin_exp_orange" edge_color="black" type="box" dim="0.05 0.03 0.02"/>
            <omc pos="0.1 0.0 0.015" name="seg4" pos_marker="2"/>
          </body>
          <body joint="rr_imp" name="seg5_4Seg" pos="0.19999999 0.0 0.0" pos_min="0.0 -0.05 -0.05" pos_max="0.35000002 0.05 0.05" damping="3.0 3.0">
            <geom pos="0.1 0.0 0.0" mass="1.0" color="dustin_exp_white" edge_color="black" type="box" dim="0.2 0.05 0.05"/>
            <geom pos="0.03 -0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
            <geom pos="0.17 -0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
            <omc pos="0.0 0.0 -0.02" name="seg5" pos_marker="4"/>
            <body joint="frozen" name="imu5_4Seg" pos="0.100000024 0.0 0.035" pos_min="0.050000012 -0.05 -0.05" pos_max="0.15000004 0.05 0.05">
              <geom mass="0.1" color="dustin_exp_orange" edge_color="black" type="box" dim="0.05 0.03 0.02"/>
              <omc pos="0.1 0.0 0.015" name="seg5" pos_marker="4"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</x_xy>
"""  # noqa: E501


def _change_joint_type(sys: ring.System, name: str, dof: int | str):
    dof_joint_types = {
        "0": "frozen",
        "1": "rr",
        "1a": "rr",
        "1b": "rr_imp",
        "2": "rsaddle",
        "3": "spherical",
    }
    dof_joint_dampings = {
        "0": np.array([]),
        "1": np.array([3.0]),
        "1a": np.array([3.0]),
        "1b": np.array([3.0, 3.0]),
        "2": np.array([3.0, 3.0]),
        "3": np.array([5.0, 5.0, 5.0]),
    }
    dof = str(dof)
    return sys.change_joint_type(
        name, dof_joint_types[dof], new_damp=dof_joint_dampings[dof], warn=False
    )


def body_to_eps_rots(
    sys: ring.System,
    x: ring.Transform,
    sys_x: ring.System,
) -> dict[str, jax.Array]:
    # (time, nlinks, 4) -> (nlinks, time, 4)
    rots = x.rot.transpose((1, 0, 2))
    l_map = sys_x.idx_map("l")
    segments = sys.findall_segments()

    y = dict()

    def f(_, __, name: str):
        if name not in segments:
            return
        q_i = rots[l_map[name]]
        q_i = ring.maths.quat_inv(q_i)
        y[name] = q_i

    sys.scan(f, "l", sys.link_names)

    return y


def _lookup_new_index(structure: Node):
    "Returns link index in new indicies for joint params and dof"
    if not structure.parent_changed:
        return structure.link_idx_new_indices
    return structure.old_parent_new_indices


def finalize_fn_factory(sys_pred: ring.System, verbose=False):
    segs = set(sys_pred.findall_segments()) - set(
        [sys_pred.find_body_to_world(name=True)]
    )

    dof_map = dict(rr=1, rsaddle=2, spherical=3, rr_imp=1)

    def finalize_fn(key, q, x, sys: ring.System):
        anchor = sys.find_body_to_world(name=True)
        structures, *_ = identify_system(
            sys_pred,
            _autodetermine_new_parents(
                sys_pred.link_parents,
                sys_pred.name_to_idx(anchor),
            ),
        )

        X = {"dt": jnp.array(sys.dt)}
        for seg in segs:
            structure = structures[sys_pred.name_to_idx(seg)]
            i = _lookup_new_index(structure)
            if verbose:
                print(
                    f"For `{seg}` and anchor `{anchor}` use link `{sys.idx_to_name(i)}`"
                )
            X = dict_union(
                X,
                {
                    seg: dict(
                        dof=dof_map[sys.link_types[i]],
                        joint_params=sys.links[i].joint_params,
                        parent_changed=jnp.array(structure.parent_changed),
                    )
                },
            )

        y = body_to_eps_rots(sys_pred, x, sys)
        return X, y

    return finalize_fn


def _add_rom(mconfig: ring.MotionConfig) -> ring.MotionConfig:
    overwrites = dict(
        rom_halfsize=0.4,
        dpos_max=0.1,
        cor_pos_min=-0.05,
        cor_pos_max=0.05,
        cor_dpos_max=0.03,
        dang_max_free_spherical=0.8,
        t_max=5.0,
    )
    existing = dict(mconfig.joint_type_specific_overwrites or {})
    for jt in ("cor", "free"):
        # existing constraints (e.g. from -S variants) take precedence
        existing[jt] = {**overwrites, **existing.get(jt, {})}
    return replace(mconfig, joint_type_specific_overwrites=existing)


def _build_diverse_configs(T: float) -> list[ring.MotionConfig]:
    """Build 12 MotionConfigs covering the full motion diversity needed for DIODEM."""
    # Group 1+2: registered configs
    registered = [
        "standard", "expFast", "expSlow", "hinUndHer",  # existing baseline
        "verySlow", "langsam",                           # fill velocity gaps
        "expSlow-S", "standard-S",                       # gait-like trunk constraints
    ]
    configs = [replace(ring.MotionConfig.from_register(c), T=T) for c in registered]

    # Custom: verySlow with maximally constrained trunk (ultra-slow gait)
    configs.append(replace(ring.MotionConfig.from_register("verySlow"), T=T,
        joint_type_specific_overwrites=dict(
            free=dict(dang_max_free_spherical=0.1, dpos_max=0.05,
                      cor_dpos_max=0.05, t_min=1.5, t_max=15.0),
            cor=dict(dang_max_free_spherical=0.1, dpos_max=0.05,
                     cor_dpos_max=0.05, t_min=1.5, t_max=15.0),
        )))

    # Custom: cyclic-fast — tight ROM oscillation at high speed
    configs.append(replace(ring.MotionConfig.from_register("expFast"), T=T,
        rom_halfsize=0.5, delta_ang_min=np.deg2rad(30), delta_ang_max=np.deg2rad(55)))

    # Custom: cyclic-slow — tight ROM oscillation at low speed
    configs.append(replace(ring.MotionConfig.from_register("expSlow"), T=T,
        rom_halfsize=0.35))

    # Custom: slow with burst-pause pattern (standstills)
    configs.append(replace(ring.MotionConfig.from_register("verySlow"), T=T,
        include_standstills_prob=0.15,
        include_standstills_t_min=0.5, include_standstills_t_max=3.0))

    return configs


def main(
    size: int,  # 32 * n_mconfigs * n_gens (= n_anchors * 3**(N-1)) * X
    output_path: str,
    configs: list[str] = ["standard", "expSlow", "expFast", "hinUndHer"],
    config_preset: Optional[str] = None,
    seed: int = 1,
    anchors: Optional[list[str]] = None,
    mot_art: bool = False,
    dyn_sim: bool = False,
    sampling_rates: list[float] = [40, 60, 80, 100, 120, 140, 160, 180, 200],
    T: float = 150.0,
    dof_configuration: Optional[list[str]] = ["111"],
    embc_rom_limitation: bool = False,
    save_rcmg_to_json: Optional[str] = None,
    include_standstills_prob: float = 0.0,
):
    """
    Main function for generating motion sequences with customizable configurations.

    Parameters:
        size (int): Number of sequences to generate.
        output_path (str): Path to the folder where the generated sequences will be stored.
        configs (list[str], optional): List of MotionConfigs to use. Defaults to
            ["standard", "expSlow", "expFast", "hinUndHer"].
        seed (int, optional): Randomness seed. Defaults to 1.
        anchors (Optional[list[str]], optional): Anchors of the four-segment chain. If None,
            all segments are chosen as anchors. Defaults to None.
        mot_art (bool, optional): Whether to simulate motion artifacts (nonrigidly attached IMUs). Defaults to False.
        dyn_sim (bool, optional): Whether to perform a dynamic or only a kinematic forward simulation. Defaults to False.
        sampling_rates (list[float], optional): Sampling rates to simulate. Defaults to
            [40, 60, 80, 100, 120, 140, 160, 180, 200].
        T (float, optional): Maximum trial length in seconds. Higher sampling rates will be truncated
            to match the length corresponding to the lowest sampling rate (e.g., 40 Hz * 150s = 6000 samples).
            Defaults to 150.0.
        dof_configuration (Optional[list[str]], optional): List of DOF (Degrees of Freedom) configurations
            for the joints in the format ['111', '121', ...]. If None, all combinations of
            1D, 2D, and 3D joints are considered. Defaults to ['111'].
        embc_rom_limitation (bool, optional): If enabled then for each `MotionConfig` adds a second `MotionConfig` object
            where the global rotation is limited to stay within [-20°, 20°] from the initial random global orientation of
            the kinematic chain.
        save_rcmg_to_json (str, optional): Save a deserialised RCMG representation in a json file with name `save_rcmg_to_json`

    Returns:
        None: This function generates motion sequences and saves them to the specified output path.
    """  # noqa: E501
    sys = ring.System.create(sys_str)

    syss = []
    segs = [s for s in sys.findall_segments() if s != sys.find_body_to_world(name=True)]
    if dof_configuration is None:
        dof_combinations = list(itertools.product(*([[1, 2, 3]] * len(segs))))
    else:
        dof_combinations = [tuple((int(s) for s in comb)) for comb in dof_configuration]

    for dofs in dof_combinations:
        _sys = sys
        for seg, dof in zip(segs, dofs):
            _sys = _change_joint_type(_sys, seg, dof)
        syss.extend(randomize_anchors(_sys, anchors))

    if mot_art:
        dyn_sim = True

    if config_preset is not None:
        if config_preset == "diverse":
            configs = _build_diverse_configs(T)
        else:
            raise ValueError(f"Unknown config_preset: {config_preset}")
    else:
        configs = [replace(ring.MotionConfig.from_register(c), T=T) for c in configs]
        if embc_rom_limitation:
            _replace_rom = lambda mconfig, add: _add_rom(mconfig) if add else mconfig
            configs = [_replace_rom(c, add) for c in configs for add in [False, True]]
        if include_standstills_prob > 0.0:
            configs = [
                replace(c, include_standstills_prob=include_standstills_prob)
                for c in configs
            ]

    rcmg = ring.RCMG(
        syss,
        configs,
        add_X_imus=True,
        dynamic_simulation=dyn_sim,
        imu_motion_artifacts=mot_art,
        imu_motion_artifacts_kwargs=dict(
            prob_rigid=0.5,
            pos_min_max=0.05,
            all_imus_either_rigid_or_flex=True,
            disable_warning=True,
        ),
        randomize_joint_params=True,
        randomize_motion_artifacts=True,
        randomize_positions=True,
        randomize_hz=True,
        randomize_hz_kwargs=dict(sampling_rates=sampling_rates, add_dt=False),
        cor=True,
        finalize_fn=finalize_fn_factory(sys),
        sys_ml=sys,
    )

    if save_rcmg_to_json is not None:
        rcmg.serialise_to_json(
            ring.utils.parse_path(save_rcmg_to_json, extension="json")
        )

    rcmg.to_folder(output_path, size, seed, overwrite=False)


if __name__ == "__main__":
    fire.Fire(main)
