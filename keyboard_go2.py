"""
GO2 keyboard locomotion test

Environment:
    Isaac Sim 5.1
    Isaac Lab v2.3.0
    Unitree GO2
    model_7850.pt

Keyboard:
    W : forward
    S : backward
    A : left
    D : right
    Q : rotate left
    E : rotate right
    SPACE : stop
"""

# ============================================================
# 0. Launch Isaac Sim first
# ============================================================

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="GO2 keyboard locomotion")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ============================================================
# 1. Imports after Isaac Sim starts
# ============================================================

import time

import carb
import gymnasium as gym
import omni.appwindow
import torch
import torch.nn as nn

import isaaclab_tasks  # registers Isaac Lab gym environments

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg_PLAY,
)


# ============================================================
# 2. Checkpoint
# ============================================================

CHECKPOINT_PATH = (
    "logs/rsl_rl/unitree_go2_rough/"
    "2024-04-06_02-37-07/model_7850.pt"
)


# ============================================================
# 3. Keyboard velocity command
#
# [ vx, vy, yaw_rate ]
# ============================================================

base_command = [0.0, 0.0, 0.0]

VX_FORWARD = 0.6
VX_BACKWARD = -0.4

VY_LEFT = 0.4
VY_RIGHT = -0.4

YAW_LEFT = 0.6
YAW_RIGHT = -0.6


def keyboard_callback(event, *args, **kwargs):
    """Handle keyboard input."""

    global base_command

    if event.type == carb.input.KeyboardEventType.KEY_PRESS:

        if event.input.name == "W":
            base_command = [VX_FORWARD, 0.0, 0.0]

        elif event.input.name == "S":
            base_command = [VX_BACKWARD, 0.0, 0.0]

        elif event.input.name == "A":
            base_command = [0.0, VY_LEFT, 0.0]

        elif event.input.name == "D":
            base_command = [0.0, VY_RIGHT, 0.0]

        elif event.input.name == "Q":
            base_command = [0.0, 0.0, YAW_LEFT]

        elif event.input.name == "E":
            base_command = [0.0, 0.0, YAW_RIGHT]

        elif event.input.name == "SPACE":
            base_command = [0.0, 0.0, 0.0]

        print(
            f"[KEYBOARD] "
            f"vx={base_command[0]:+.2f}, "
            f"vy={base_command[1]:+.2f}, "
            f"yaw={base_command[2]:+.2f}"
        )

    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:

        if event.input.name in ["W", "S", "A", "D", "Q", "E"]:
            base_command = [0.0, 0.0, 0.0]

    return True


# ============================================================
# 4. Legacy checkpoint loader
#
# model_7850.pt:
#
# 235
#  ↓
# 512
#  ↓ ELU
# 256
#  ↓ ELU
# 128
#  ↓ ELU
# 12
#
# ============================================================

def load_policy(checkpoint_path, device):

    print(f"[INFO] Loading policy:")
    print(f"       {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint["model_state_dict"]

    # Read dimensions directly from checkpoint
    obs_dim = state_dict["actor.0.weight"].shape[1]
    action_dim = state_dict["actor.6.weight"].shape[0]

    print(f"[INFO] Observation dimension : {obs_dim}")
    print(f"[INFO] Action dimension      : {action_dim}")

    actor = nn.Sequential(
        nn.Linear(obs_dim, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, action_dim),
    )

    # checkpoint keys:
    # actor.0.weight
    # actor.0.bias
    # ...
    #
    # Sequential expects:
    # 0.weight
    # 0.bias
    # ...

    actor_state_dict = {
        key[len("actor."):]: value
        for key, value in state_dict.items()
        if key.startswith("actor.")
    }

    actor.load_state_dict(actor_state_dict)

    actor.to(device)
    actor.eval()

    print("[INFO] Policy loaded successfully")

    return actor, obs_dim, action_dim


# ============================================================
# 5. Extract policy observation
# ============================================================

def get_policy_observation(obs):

    # Depending on wrapper/version,
    # observation may be a dict or directly a tensor.

    if isinstance(obs, dict):
        return obs["policy"]

    try:
        if "policy" in obs:
            return obs["policy"]
    except Exception:
        pass

    return obs


# ============================================================
# 6. Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Environment configuration
    # --------------------------------------------------------

    env_cfg = UnitreeGo2RoughEnvCfg_PLAY()

    # One GO2 only
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5

    # --------------------------------------------------------
    # Use FLAT terrain
    #
    # Important:
    # Keep height scanner.
    #
    # Policy expects:
    #
    # 235 observations
    #
    # including 187 height scan values.
    #
    # On flat ground these height values simply become flat.
    # --------------------------------------------------------

    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.max_init_terrain_level = None

    # No terrain curriculum
    env_cfg.curriculum.terrain_levels = None

    # --------------------------------------------------------
    # Disable random velocity command generation
    #
    # We will overwrite command using keyboard.
    # --------------------------------------------------------

    env_cfg.commands.base_velocity.resampling_time_range = (
        1.0e9,
        1.0e9,
    )

    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.debug_vis = False

    # --------------------------------------------------------
    # Disable unnecessary randomization
    # --------------------------------------------------------

    env_cfg.observations.policy.enable_corruption = False

    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None

    # Make reset deterministic
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    env_cfg.events.reset_robot_joints.params[
        "position_range"
    ] = (1.0, 1.0)

    # --------------------------------------------------------
    # Create Isaac Lab environment
    # --------------------------------------------------------

    print("[INFO] Creating GO2 environment...")

    env = gym.make(
        "Isaac-Velocity-Rough-Unitree-Go2-v0",
        cfg=env_cfg,
    )

    # Same wrapper used by RSL-RL
    env = RslRlVecEnvWrapper(env)

    device = env.unwrapped.device

    print(f"[INFO] Environment device: {device}")

    # Camera
    env.unwrapped.sim.set_camera_view(
        eye=[2.5, 2.5, 1.5],
        target=[0.0, 0.0, 0.35],
    )

    # --------------------------------------------------------
    # Load locomotion policy
    # --------------------------------------------------------

    actor, expected_obs_dim, expected_action_dim = load_policy(
        CHECKPOINT_PATH,
        device,
    )

    # --------------------------------------------------------
    # Obtain command term
    # --------------------------------------------------------

    command_term = env.unwrapped.command_manager.get_term(
        "base_velocity"
    )

    # --------------------------------------------------------
    # Keyboard subscription
    # --------------------------------------------------------

    input_interface = carb.input.acquire_input_interface()

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()

    keyboard_subscription = (
        input_interface.subscribe_to_keyboard_events(
            keyboard,
            keyboard_callback,
        )
    )

    # Keep subscription alive
    _ = keyboard_subscription

    # --------------------------------------------------------
    # Initial observation
    # --------------------------------------------------------

    obs = env.get_observations()

    policy_obs = get_policy_observation(obs)

    print()
    print("======================================")
    print(" GO2 KEYBOARD CONTROL READY")
    print("======================================")
    print(f" Observation shape : {tuple(policy_obs.shape)}")
    print(f" Expected obs dim  : {expected_obs_dim}")
    print(f" Expected actions  : {expected_action_dim}")
    print()
    print(" W     : forward")
    print(" S     : backward")
    print(" A     : left")
    print(" D     : right")
    print(" Q     : rotate left")
    print(" E     : rotate right")
    print(" SPACE : stop")
    print("======================================")
    print()

    if policy_obs.shape[-1] != expected_obs_dim:
        raise RuntimeError(
            f"Observation mismatch: "
            f"environment={policy_obs.shape[-1]}, "
            f"policy={expected_obs_dim}"
        )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    dt = env.unwrapped.step_dt

    while simulation_app.is_running():

        start_time = time.time()

        # Keyboard command:
        #
        # [vx, vy, yaw_rate]
        command = torch.tensor(
            [base_command],
            dtype=torch.float32,
            device=device,
        )

        # Update Isaac Lab command manager
        command_term.vel_command_b[:] = command

        with torch.inference_mode():

            policy_obs = get_policy_observation(obs)

            # ------------------------------------------------
            # Observation structure:
            #
            # 0:3    base linear velocity
            # 3:6    base angular velocity
            # 6:9    projected gravity
            # 9:12   velocity command
            #
            # Explicitly insert keyboard command here to avoid
            # one-step delay / random command contamination.
            # ------------------------------------------------

            policy_input = policy_obs.clone()

            policy_input[:, 9:12] = command

            # Policy inference
            actions = actor(policy_input)

            # Apply 12-dimensional joint action
            obs, _, _, _ = env.step(actions)

        # Real-time-ish execution
        sleep_time = dt - (time.time() - start_time)

        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":

    try:
        main()

    finally:
        simulation_app.close()
