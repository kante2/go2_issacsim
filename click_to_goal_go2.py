"""
GO2 Click-to-Goal Navigation
============================

Environment
-----------
Isaac Sim 5.1
Isaac Lab v2.3.0
Unitree GO2
Locomotion policy: model_7850.pt

Control
-------
LEFT CLICK on viewport ground:
    Set navigation goal

SPACE:
    Cancel current goal / stop

Architecture
------------
Mouse click
    ↓
Viewport request_query()
    ↓
Goal world position (x, y)
    ↓
Point-to-goal controller
    ↓
[vx, 0, yaw_rate]
    ↓
GO2 locomotion policy
    ↓
12 joint actions
    ↓
GO2
"""

# ============================================================
# 0. Isaac Sim launch
# ============================================================

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="GO2 click-to-goal controller"
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ============================================================
# 1. Imports AFTER Isaac Sim startup
# ============================================================

import math
import time

import carb
import gymnasium as gym
import omni.appwindow
import omni.usd
import torch
import torch.nn as nn

from pxr import Gf, UsdGeom

from omni.ui import scene as sc

from omni.kit.viewport.utility import (
    get_active_viewport_and_window,
)

import isaaclab_tasks

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg_PLAY,
)


# ============================================================
# 2. GO2 locomotion checkpoint
# ============================================================

CHECKPOINT_PATH = (
    "logs/rsl_rl/unitree_go2_rough/"
    "2024-04-06_02-37-07/"
    "model_7850.pt"
)


# ============================================================
# 3. Navigation controller parameters
# ============================================================

# Goal 도달 판정 거리
GOAL_TOLERANCE = 0.25  # [m]

# 최대 전진 속도
MAX_VX = 0.60  # [m/s]

# 최대 회전 속도
MAX_YAW_RATE = 0.80  # [rad/s]

# P-controller gains
KP_LINEAR = 0.8
KP_YAW = 1.5

# 목표 방향과 로봇 방향의 차이가 이 값보다 크면
# 전진하지 않고 제자리 회전부터 수행
TURN_IN_PLACE_THRESHOLD = math.radians(35.0)


# ============================================================
# 4. Global navigation state
# ============================================================

goal_xy = None
goal_active = False


# ============================================================
# 5. Utility functions
# ============================================================

def wrap_to_pi(angle):
    """
    Normalize angle to [-pi, pi].
    """
    return (
        angle + math.pi
    ) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(q):
    """
    Isaac Lab quaternion:
        [w, x, y, z]

    quaternion → yaw
    """

    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    siny_cosp = 2.0 * (
        w * z + x * y
    )

    cosy_cosp = 1.0 - 2.0 * (
        y * y + z * z
    )

    return math.atan2(
        siny_cosp,
        cosy_cosp,
    )


# ============================================================
# 6. Policy loader
#
# checkpoint:
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

def load_policy(
    checkpoint_path,
    device,
):

    print()
    print("========================================")
    print(" Loading GO2 locomotion policy")
    print("========================================")

    print(
        f"[POLICY] checkpoint = "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    obs_dim = (
        state_dict[
            "actor.0.weight"
        ].shape[1]
    )

    action_dim = (
        state_dict[
            "actor.6.weight"
        ].shape[0]
    )

    print(
        f"[POLICY] obs dim    = "
        f"{obs_dim}"
    )

    print(
        f"[POLICY] action dim = "
        f"{action_dim}"
    )

    actor = nn.Sequential(

        nn.Linear(
            obs_dim,
            512,
        ),

        nn.ELU(),

        nn.Linear(
            512,
            256,
        ),

        nn.ELU(),

        nn.Linear(
            256,
            128,
        ),

        nn.ELU(),

        nn.Linear(
            128,
            action_dim,
        ),
    )

    # checkpoint:
    #
    # actor.0.weight
    # actor.0.bias
    # ...
    #
    # nn.Sequential:
    #
    # 0.weight
    # 0.bias
    # ...

    actor_state_dict = {

        key[len("actor."):]: value

        for key, value
        in state_dict.items()

        if key.startswith("actor.")
    }

    actor.load_state_dict(
        actor_state_dict
    )

    actor.to(device)

    actor.eval()

    print(
        "[POLICY] loaded successfully"
    )

    print(
        "========================================"
    )
    print()

    return (
        actor,
        obs_dim,
        action_dim,
    )


# ============================================================
# 7. Observation helper
# ============================================================

def get_policy_observation(obs):

    if isinstance(obs, dict):

        return obs["policy"]

    try:

        if "policy" in obs:

            return obs["policy"]

    except Exception:

        pass

    return obs


# ============================================================
# 8. Goal marker
# ============================================================

def create_goal_marker():
    """
    Create red sphere representing goal position.
    """

    stage = (
        omni.usd
        .get_context()
        .get_stage()
    )

    sphere = UsdGeom.Sphere.Define(
        stage,
        "/World/GoalMarker",
    )

    sphere.CreateRadiusAttr(
        0.10
    )

    sphere.CreateDisplayColorAttr(
        [
            Gf.Vec3f(
                1.0,
                0.0,
                0.0,
            )
        ]
    )

    sphere.CreateVisibilityAttr().Set(
        UsdGeom.Tokens.invisible
    )

    xform_api = (
        UsdGeom.XformCommonAPI(
            sphere.GetPrim()
        )
    )

    return (
        sphere,
        xform_api,
    )


def update_goal_marker(
    sphere,
    xform_api,
    x,
    y,
    z,
):
    """
    Move red goal marker to clicked position.
    """

    # ground보다 조금 위에 sphere 표시
    marker_z = max(
        float(z) + 0.10,
        0.10,
    )

    xform_api.SetTranslate(
        Gf.Vec3d(
            float(x),
            float(y),
            marker_z,
        )
    )

    sphere.CreateVisibilityAttr().Set(
        UsdGeom.Tokens.inherited
    )


def hide_goal_marker(sphere):

    sphere.CreateVisibilityAttr().Set(
        UsdGeom.Tokens.invisible
    )


# ============================================================
# 9. Viewport mouse click
# ============================================================

def setup_viewport_click(
    goal_marker,
    goal_marker_xform,
):
    """
    Install an omni.ui.scene Screen over the active viewport.

    Workflow:

        ClickGesture
            ↓
        NDC mouse position
            ↓
        Viewport texture pixel
            ↓
        request_query()
            ↓
        world position
    """

    global goal_xy
    global goal_active

    # --------------------------------------------------------
    # Get active viewport
    # --------------------------------------------------------

    (
        viewport_api,
        viewport_window,
    ) = get_active_viewport_and_window()

    if (
        viewport_api is None
        or viewport_window is None
    ):

        raise RuntimeError(
            "Active Viewport를 찾을 수 없습니다."
        )

    print()
    print(
        "========================================"
    )
    print(
        " Viewport Click Handler"
    )
    print(
        "========================================"
    )

    print(
        "[CLICK] Active viewport found"
    )

    print(
        f"[CLICK] viewport resolution = "
        f"{viewport_api.resolution}"
    )

    # --------------------------------------------------------
    # Renderer query callback
    # --------------------------------------------------------

    def query_completed(
        prim_path,
        world_position,
        *args,
    ):

        global goal_xy
        global goal_active

        print()
        print(
            "----------------------------------------"
        )
        print(
            "[PICK RESULT]"
        )

        print(
            f" prim = {prim_path}"
        )

        print(
            f" pos  = {world_position}"
        )

        # 아무 geometry도 못 찍은 경우
        if (
            not prim_path
            or world_position is None
        ):

            print(
                "[PICK] No geometry detected."
            )

            print(
                "----------------------------------------"
            )

            return

        # ----------------------------------------------------
        # World coordinates
        # ----------------------------------------------------

        gx = float(
            world_position[0]
        )

        gy = float(
            world_position[1]
        )

        gz = float(
            world_position[2]
        )

        goal_xy = (
            gx,
            gy,
        )

        goal_active = True

        # ----------------------------------------------------
        # Goal marker
        # ----------------------------------------------------

        update_goal_marker(
            goal_marker,
            goal_marker_xform,
            gx,
            gy,
            gz,
        )

        print()
        print(
            "========================================"
        )

        print(
            " NEW GOAL"
        )

        print(
            f" x = {gx:.3f}"
        )

        print(
            f" y = {gy:.3f}"
        )

        print(
            f" z = {gz:.3f}"
        )

        print(
            "========================================"
        )

        print()

    # --------------------------------------------------------
    # Screen click callback
    # --------------------------------------------------------

    def on_viewport_clicked(sender):

        print()
        print(
            "[MOUSE] LEFT CLICK detected"
        )

        # omni.ui.scene mouse coordinate
        mouse_ndc = (
            sender
            .gesture_payload
            .mouse
        )

        print(
            f"[MOUSE] NDC = "
            f"("
            f"{float(mouse_ndc[0]):+.3f}, "
            f"{float(mouse_ndc[1]):+.3f}"
            f")"
        )

        # ----------------------------------------------------
        # NDC → rendered viewport texture pixel
        # ----------------------------------------------------

        (
            pixel,
            in_viewport,
        ) = (
            viewport_api
            .map_ndc_to_texture_pixel(
                mouse_ndc
            )
        )

        print(
            f"[MOUSE] pixel = {pixel}"
        )

        print(
            f"[MOUSE] in viewport = "
            f"{in_viewport}"
        )

        if not in_viewport:

            print(
                "[MOUSE] Click outside "
                "rendered viewport."
            )

            return

        # ----------------------------------------------------
        # Pixel → world position
        # ----------------------------------------------------

        viewport_api.request_query(
            pixel,
            query_completed,
            query_name="go2_goal_query",
        )

    # --------------------------------------------------------
    # SceneView overlay
    # --------------------------------------------------------

    frame = (
        viewport_window
        .get_frame(
            "go2_click_to_goal"
        )
    )

    with frame:

        scene_view = sc.SceneView(
            aspect_ratio_policy=(
                sc.AspectRatioPolicy.STRETCH
            )
        )

        # Viewport camera matrix와 SceneView 연동
        viewport_api.add_scene_view(
            scene_view
        )

        with scene_view.scene:

            click_gesture = (
                sc.ClickGesture(
                    on_ended_fn=(
                        on_viewport_clicked
                    ),
                    mouse_button=0,
                    name="go2_goal_click",
                )
            )

            # IMPORTANT:
            #
            # Screen은 viewport 전체를 덮는
            # invisible interaction surface.
            #
            screen = sc.Screen(
                gesture=click_gesture
            )

    print(
        "[CLICK] SceneView installed"
    )

    print(
        "[CLICK] ClickGesture installed"
    )

    print(
        "========================================"
    )
    print()

    # IMPORTANT:
    # Python GC로 제거되지 않도록
    # references 반환
    return {

        "viewport_api":
            viewport_api,

        "viewport_window":
            viewport_window,

        "frame":
            frame,

        "scene_view":
            scene_view,

        "screen":
            screen,

        "gesture":
            click_gesture,
    }


# ============================================================
# 10. SPACE key emergency stop
# ============================================================

def keyboard_callback(event):

    global goal_active
    global goal_xy

    if (
        event.type
        == carb.input.KeyboardEventType.KEY_PRESS
    ):

        if event.input.name == "SPACE":

            goal_active = False
            goal_xy = None

            print()
            print(
                "========================================"
            )
            print(
                " GOAL CANCELLED"
            )
            print(
                "========================================"
            )
            print()

    return True


# ============================================================
# 11. Point-to-goal controller
# ============================================================

def compute_velocity_command(
    robot,
):
    """
    Current robot pose + goal position
        ↓
    velocity command

    output:
        vx
        vy
        yaw_rate
    """

    global goal_active
    global goal_xy

    # --------------------------------------------------------
    # No active goal
    # --------------------------------------------------------

    if (
        not goal_active
        or goal_xy is None
    ):

        return (
            0.0,
            0.0,
            0.0,
        )

    # --------------------------------------------------------
    # Robot position
    # --------------------------------------------------------

    position = (
        robot
        .data
        .root_pos_w[0]
        .detach()
        .cpu()
        .tolist()
    )

    # --------------------------------------------------------
    # Robot orientation
    # --------------------------------------------------------

    quat = (
        robot
        .data
        .root_quat_w[0]
        .detach()
        .cpu()
        .tolist()
    )

    robot_x = position[0]
    robot_y = position[1]

    robot_yaw = (
        quaternion_to_yaw(
            quat
        )
    )

    # --------------------------------------------------------
    # Position error
    # --------------------------------------------------------

    dx = (
        goal_xy[0]
        - robot_x
    )

    dy = (
        goal_xy[1]
        - robot_y
    )

    distance = math.sqrt(
        dx * dx
        + dy * dy
    )

    # --------------------------------------------------------
    # Desired heading
    # --------------------------------------------------------

    desired_yaw = math.atan2(
        dy,
        dx,
    )

    heading_error = wrap_to_pi(
        desired_yaw
        - robot_yaw
    )

    # --------------------------------------------------------
    # Goal reached
    # --------------------------------------------------------

    if (
        distance
        < GOAL_TOLERANCE
    ):

        goal_active = False

        print()
        print(
            "========================================"
        )

        print(
            " GOAL REACHED"
        )

        print(
            f" robot = "
            f"("
            f"{robot_x:.3f}, "
            f"{robot_y:.3f}"
            f")"
        )

        print(
            f" remaining error = "
            f"{distance:.3f} m"
        )

        print(
            "========================================"
        )
        print()

        return (
            0.0,
            0.0,
            0.0,
        )

    # --------------------------------------------------------
    # Yaw controller
    # --------------------------------------------------------

    yaw_rate = (
        KP_YAW
        * heading_error
    )

    yaw_rate = max(
        -MAX_YAW_RATE,
        min(
            MAX_YAW_RATE,
            yaw_rate,
        ),
    )

    # --------------------------------------------------------
    # Translation controller
    # --------------------------------------------------------

    if (
        abs(heading_error)
        > TURN_IN_PLACE_THRESHOLD
    ):

        # 방향이 많이 틀어진 경우:
        # 먼저 제자리 회전
        vx = 0.0

    else:

        # 기본 distance P controller
        vx = (
            KP_LINEAR
            * distance
        )

        vx = min(
            MAX_VX,
            vx,
        )

        # heading error가 커질수록
        # forward speed 감소
        vx *= max(
            0.0,
            math.cos(
                heading_error
            ),
        )

    # --------------------------------------------------------
    # Current controller:
    #
    # vx      = forward
    # vy      = 0
    # yawrate = heading control
    # --------------------------------------------------------

    vy = 0.0

    return (
        vx,
        vy,
        yaw_rate,
    )


# ============================================================
# 12. Main
# ============================================================

def main():

    global goal_xy
    global goal_active

    # --------------------------------------------------------
    # GO2 environment configuration
    # --------------------------------------------------------

    env_cfg = (
        UnitreeGo2RoughEnvCfg_PLAY()
    )

    # One robot only
    env_cfg.scene.num_envs = 1

    env_cfg.scene.env_spacing = 2.5

    # --------------------------------------------------------
    # Flat terrain
    # --------------------------------------------------------

    env_cfg.scene.terrain.terrain_type = (
        "plane"
    )

    env_cfg.scene.terrain.terrain_generator = (
        None
    )

    env_cfg.scene.terrain.max_init_terrain_level = (
        None
    )

    # no terrain curriculum
    env_cfg.curriculum.terrain_levels = (
        None
    )

    # --------------------------------------------------------
    # Disable random velocity command
    # --------------------------------------------------------

    env_cfg.commands.base_velocity.resampling_time_range = (
        1.0e9,
        1.0e9,
    )

    env_cfg.commands.base_velocity.rel_standing_envs = (
        0.0
    )

    env_cfg.commands.base_velocity.rel_heading_envs = (
        0.0
    )

    env_cfg.commands.base_velocity.heading_command = (
        False
    )

    env_cfg.commands.base_velocity.debug_vis = (
        False
    )

    env_cfg.commands.base_velocity.ranges.lin_vel_x = (
        0.0,
        0.0,
    )

    env_cfg.commands.base_velocity.ranges.lin_vel_y = (
        0.0,
        0.0,
    )

    env_cfg.commands.base_velocity.ranges.ang_vel_z = (
        0.0,
        0.0,
    )

    # --------------------------------------------------------
    # Disable observation noise
    # --------------------------------------------------------

    env_cfg.observations.policy.enable_corruption = (
        False
    )

    # --------------------------------------------------------
    # Disable disturbance
    # --------------------------------------------------------

    env_cfg.events.push_robot = None

    env_cfg.events.base_external_force_torque = (
        None
    )

    # --------------------------------------------------------
    # Deterministic reset
    # --------------------------------------------------------

    env_cfg.events.reset_base.params[
        "pose_range"
    ] = {

        "x": (
            0.0,
            0.0,
        ),

        "y": (
            0.0,
            0.0,
        ),

        "yaw": (
            0.0,
            0.0,
        ),
    }

    env_cfg.events.reset_base.params[
        "velocity_range"
    ] = {

        "x": (
            0.0,
            0.0,
        ),

        "y": (
            0.0,
            0.0,
        ),

        "z": (
            0.0,
            0.0,
        ),

        "roll": (
            0.0,
            0.0,
        ),

        "pitch": (
            0.0,
            0.0,
        ),

        "yaw": (
            0.0,
            0.0,
        ),
    }

    env_cfg.events.reset_robot_joints.params[
        "position_range"
    ] = (
        1.0,
        1.0,
    )

    # timeout 사실상 제거
    env_cfg.episode_length_s = (
        1.0e9
    )

    # --------------------------------------------------------
    # Environment creation
    # --------------------------------------------------------

    print()
    print(
        "[ENV] Creating GO2 environment..."
    )

    env = gym.make(
        "Isaac-Velocity-Rough-Unitree-Go2-v0",
        cfg=env_cfg,
    )

    # Same wrapper structure used for RSL-RL
    env = RslRlVecEnvWrapper(
        env
    )

    device = (
        env
        .unwrapped
        .device
    )

    print(
        f"[ENV] device = "
        f"{device}"
    )

    # --------------------------------------------------------
    # Robot handle
    # --------------------------------------------------------

    robot = (
        env
        .unwrapped
        .scene[
            "robot"
        ]
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    env.unwrapped.sim.set_camera_view(

        eye=[
            4.0,
            4.0,
            3.0,
        ],

        target=[
            0.0,
            0.0,
            0.2,
        ],
    )

    # --------------------------------------------------------
    # Goal marker
    # --------------------------------------------------------

    (
        goal_marker,
        goal_marker_xform,
    ) = create_goal_marker()

    # --------------------------------------------------------
    # Load GO2 locomotion policy
    # --------------------------------------------------------

    (
        actor,
        expected_obs_dim,
        expected_action_dim,
    ) = load_policy(
        CHECKPOINT_PATH,
        device,
    )

    # --------------------------------------------------------
    # Isaac Lab command manager
    # --------------------------------------------------------

    command_term = (
        env
        .unwrapped
        .command_manager
        .get_term(
            "base_velocity"
        )
    )

    # --------------------------------------------------------
    # Keyboard
    # --------------------------------------------------------

    input_interface = (
        carb
        .input
        .acquire_input_interface()
    )

    app_window = (
        omni
        .appwindow
        .get_default_app_window()
    )

    keyboard = (
        app_window
        .get_keyboard()
    )

    keyboard_subscription = (
        input_interface
        .subscribe_to_keyboard_events(
            keyboard,
            keyboard_callback,
        )
    )

    # Keep ref alive
    _ = keyboard_subscription

    # --------------------------------------------------------
    # Viewport mouse click
    # --------------------------------------------------------

    click_handler = (
        setup_viewport_click(
            goal_marker,
            goal_marker_xform,
        )
    )

    # --------------------------------------------------------
    # Initial observation
    # --------------------------------------------------------

    obs = (
        env
        .get_observations()
    )

    policy_obs = (
        get_policy_observation(
            obs
        )
    )

    print()
    print(
        "========================================"
    )

    print(
        " GO2 CLICK-TO-GOAL READY"
    )

    print(
        "========================================"
    )

    print(
        f" Observation shape = "
        f"{tuple(policy_obs.shape)}"
    )

    print(
        f" Expected obs      = "
        f"{expected_obs_dim}"
    )

    print(
        f" Expected actions  = "
        f"{expected_action_dim}"
    )

    print()
    print(
        " LEFT CLICK : set goal"
    )

    print(
        " SPACE      : cancel / stop"
    )

    print()
    print(
        " Click directly inside the 3D Viewport."
    )

    print(
        "========================================"
    )

    print()

    # --------------------------------------------------------
    # Dimension verification
    # --------------------------------------------------------

    if (
        policy_obs.shape[-1]
        != expected_obs_dim
    ):

        raise RuntimeError(

            "Observation dimension mismatch: "

            f"environment="
            f"{policy_obs.shape[-1]}, "

            f"policy="
            f"{expected_obs_dim}"
        )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    dt = (
        env
        .unwrapped
        .step_dt
    )

    status_counter = 0

    try:

        while simulation_app.is_running():

            start_time = (
                time.time()
            )

            # ================================================
            # Navigation controller
            # ================================================

            (
                vx,
                vy,
                yaw_rate,
            ) = compute_velocity_command(
                robot
            )

            command = torch.tensor(

                [
                    [
                        vx,
                        vy,
                        yaw_rate,
                    ]
                ],

                dtype=torch.float32,

                device=device,
            )

            # -----------------------------------------------
            # Update Isaac command manager
            # -----------------------------------------------

            command_term.vel_command_b[:] = (
                command
            )

            # ================================================
            # Locomotion policy
            # ================================================

            with torch.inference_mode():

                policy_obs = (
                    get_policy_observation(
                        obs
                    )
                )

                policy_input = (
                    policy_obs.clone()
                )

                # -------------------------------------------
                # Policy observation:
                #
                # 0:3    base_lin_vel
                # 3:6    base_ang_vel
                # 6:9    projected_gravity
                # 9:12   velocity command
                #
                # Explicitly replace command section.
                # -------------------------------------------

                policy_input[
                    :,
                    9:12
                ] = command

                # -------------------------------------------
                # Policy:
                #
                # 235 observations
                #       ↓
                # neural network
                #       ↓
                # 12 joint actions
                # -------------------------------------------

                actions = actor(
                    policy_input
                )

                # -------------------------------------------
                # Step environment
                # -------------------------------------------

                (
                    obs,
                    _,
                    _,
                    _,
                ) = env.step(
                    actions
                )

            # ================================================
            # Navigation status log
            # ================================================

            status_counter += 1

            if (
                goal_active
                and goal_xy is not None
                and status_counter % 25 == 0
            ):

                position = (
                    robot
                    .data
                    .root_pos_w[0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                dx = (
                    goal_xy[0]
                    - position[0]
                )

                dy = (
                    goal_xy[1]
                    - position[1]
                )

                distance = math.sqrt(
                    dx * dx
                    + dy * dy
                )

                print(
                    f"[NAV] "
                    f"robot=("
                    f"{position[0]:+.2f}, "
                    f"{position[1]:+.2f}) | "
                    f"goal=("
                    f"{goal_xy[0]:+.2f}, "
                    f"{goal_xy[1]:+.2f}) | "
                    f"dist={distance:.2f} | "
                    f"cmd=["
                    f"{vx:+.2f}, "
                    f"{vy:+.2f}, "
                    f"{yaw_rate:+.2f}"
                    f"]"
                )

            # ================================================
            # Real-time-ish pacing
            # ================================================

            elapsed = (
                time.time()
                - start_time
            )

            sleep_time = (
                dt - elapsed
            )

            if sleep_time > 0:

                time.sleep(
                    sleep_time
                )

    finally:

        # ----------------------------------------------------
        # Viewport cleanup
        # ----------------------------------------------------

        try:

            viewport_api = (
                click_handler[
                    "viewport_api"
                ]
            )

            scene_view = (
                click_handler[
                    "scene_view"
                ]
            )

            viewport_api.remove_scene_view(
                scene_view
            )

            scene_view.destroy()

        except Exception as e:

            print(
                f"[WARN] Viewport cleanup: "
                f"{e}"
            )

        env.close()


# ============================================================
# 13. Entry point
# ============================================================

if __name__ == "__main__":

    try:

        main()

    finally:

        simulation_app.close()
