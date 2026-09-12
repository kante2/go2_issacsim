"""
GO2 Click-to-Goal + ROS2 / RViz
================================

Isaac Sim 5.1
Isaac Lab v2.3.0
Unitree GO2
Locomotion policy: model_7850.pt

Isaac Viewport:
    LEFT CLICK : set goal
    SPACE      : cancel goal

ROS2 topics:
    /odom          nav_msgs/Odometry
    /goal_pose     geometry_msgs/PoseStamped
    /path          nav_msgs/Path       (actual driven path)
    /planned_path  nav_msgs/Path       (current straight-line reference)
    /tf            tf2_msgs/TFMessage

RViz Fixed Frame:
    map
"""

# ============================================================
# 0. Launch Isaac Sim
# ============================================================

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="GO2 click-to-goal + RViz"
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

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg_PLAY,
)


# ============================================================
# 2. ROS2 imports
# ============================================================

try:
    import rclpy

    from rclpy.node import Node
    from rclpy.qos import QoSProfile

    from geometry_msgs.msg import (
        PoseStamped,
        TransformStamped,
    )

    from nav_msgs.msg import (
        Odometry,
        Path,
    )

    from tf2_msgs.msg import TFMessage

except ImportError as e:

    raise RuntimeError(
        "\n"
        "ROS2 Python modules could not be imported.\n"
        "Run this script with Isaac Sim's bundled Humble rclpy "
        "environment configured.\n"
        f"Original error: {e}"
    )


# ============================================================
# 3. Configuration
# ============================================================

CHECKPOINT_PATH = (
    "logs/rsl_rl/unitree_go2_rough/"
    "2024-04-06_02-37-07/"
    "model_7850.pt"
)

MAP_FRAME = "map"
BASE_FRAME = "base_link"

GOAL_TOLERANCE = 0.25

MAX_VX = 0.60
MAX_YAW_RATE = 0.80

KP_LINEAR = 0.8
KP_YAW = 1.5

TURN_IN_PLACE_THRESHOLD = math.radians(35.0)

# Actual path point spacing
PATH_POINT_MIN_DISTANCE = 0.03

# Prevent unlimited memory usage
MAX_PATH_POINTS = 5000

# Planned path sampling
PLANNED_PATH_RESOLUTION = 0.05


# ============================================================
# 4. Navigation state
# ============================================================

goal_xy = None
goal_active = False

ros_bridge = None


# ============================================================
# 5. Math
# ============================================================

def wrap_to_pi(angle):
    return (
        angle + math.pi
    ) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(q):
    """
    Isaac Lab quaternion:
        [w, x, y, z]
    """

    w, x, y, z = q

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


def yaw_to_ros_quaternion(yaw):
    """
    Return ROS quaternion:
        x, y, z, w
    """

    return (
        0.0,
        0.0,
        math.sin(yaw / 2.0),
        math.cos(yaw / 2.0),
    )


def tensor_to_list(tensor):
    return (
        tensor
        .detach()
        .cpu()
        .tolist()
    )


# ============================================================
# 6. ROS2 / RViz publisher
# ============================================================

class Go2RvizBridge(Node):

    def __init__(self):

        super().__init__(
            "go2_isaac_rviz_bridge"
        )

        qos = QoSProfile(
            depth=10
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            "/odom",
            qos,
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            "/goal_pose",
            qos,
        )

        self.path_pub = self.create_publisher(
            Path,
            "/path",
            qos,
        )

        self.planned_path_pub = (
            self.create_publisher(
                Path,
                "/planned_path",
                qos,
            )
        )

        # Using TFMessage directly, same style as go2_omniverse
        self.tf_pub = self.create_publisher(
            TFMessage,
            "/tf",
            qos,
        )

        self.actual_path = Path()
        self.actual_path.header.frame_id = (
            MAP_FRAME
        )

        self.last_path_xy = None

        print()
        print("========================================")
        print(" ROS2 / RViz publishers ready")
        print("========================================")
        print(" /odom")
        print(" /goal_pose")
        print(" /path")
        print(" /planned_path")
        print(" /tf")
        print("========================================")
        print()


    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    def publish_goal(
        self,
        x,
        y,
        z=0.0,
    ):

        msg = PoseStamped()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.header.frame_id = (
            MAP_FRAME
        )

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)

        msg.pose.orientation.w = 1.0

        self.goal_pub.publish(
            msg
        )


    # --------------------------------------------------------
    # Odometry + TF
    # --------------------------------------------------------

    def publish_robot_state(
        self,
        robot,
    ):

        stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        pos = tensor_to_list(
            robot.data.root_pos_w[0]
        )

        # Isaac:
        # [w, x, y, z]
        quat = tensor_to_list(
            robot.data.root_quat_w[0]
        )

        try:

            lin_vel = tensor_to_list(
                robot.data.root_lin_vel_b[0]
            )

            ang_vel = tensor_to_list(
                robot.data.root_ang_vel_b[0]
            )

        except Exception:

            lin_vel = [
                0.0,
                0.0,
                0.0,
            ]

            ang_vel = [
                0.0,
                0.0,
                0.0,
            ]

        # ====================================================
        # TF: map -> base_link
        # ====================================================

        transform = TransformStamped()

        transform.header.stamp = stamp
        transform.header.frame_id = MAP_FRAME

        transform.child_frame_id = (
            BASE_FRAME
        )

        transform.transform.translation.x = (
            float(pos[0])
        )

        transform.transform.translation.y = (
            float(pos[1])
        )

        transform.transform.translation.z = (
            float(pos[2])
        )

        transform.transform.rotation.x = (
            float(quat[1])
        )

        transform.transform.rotation.y = (
            float(quat[2])
        )

        transform.transform.rotation.z = (
            float(quat[3])
        )

        transform.transform.rotation.w = (
            float(quat[0])
        )

        tf_msg = TFMessage(
            transforms=[transform]
        )

        self.tf_pub.publish(
            tf_msg
        )

        # ====================================================
        # Odometry
        # ====================================================

        odom = Odometry()

        odom.header.stamp = stamp
        odom.header.frame_id = MAP_FRAME

        odom.child_frame_id = (
            BASE_FRAME
        )

        odom.pose.pose.position.x = (
            float(pos[0])
        )

        odom.pose.pose.position.y = (
            float(pos[1])
        )

        odom.pose.pose.position.z = (
            float(pos[2])
        )

        odom.pose.pose.orientation.x = (
            float(quat[1])
        )

        odom.pose.pose.orientation.y = (
            float(quat[2])
        )

        odom.pose.pose.orientation.z = (
            float(quat[3])
        )

        odom.pose.pose.orientation.w = (
            float(quat[0])
        )

        odom.twist.twist.linear.x = (
            float(lin_vel[0])
        )

        odom.twist.twist.linear.y = (
            float(lin_vel[1])
        )

        odom.twist.twist.linear.z = (
            float(lin_vel[2])
        )

        odom.twist.twist.angular.x = (
            float(ang_vel[0])
        )

        odom.twist.twist.angular.y = (
            float(ang_vel[1])
        )

        odom.twist.twist.angular.z = (
            float(ang_vel[2])
        )

        self.odom_pub.publish(
            odom
        )

        # ====================================================
        # Actual path
        # ====================================================

        self.update_actual_path(
            pos,
            quat,
            stamp,
        )


    # --------------------------------------------------------
    # Actual driven trajectory
    # --------------------------------------------------------

    def reset_actual_path(
        self,
        pos=None,
        quat=None,
    ):

        self.actual_path = Path()

        self.actual_path.header.frame_id = (
            MAP_FRAME
        )

        self.last_path_xy = None

        if (
            pos is not None
            and quat is not None
        ):

            stamp = (
                self.get_clock()
                .now()
                .to_msg()
            )

            self._append_actual_pose(
                pos,
                quat,
                stamp,
            )

            self.path_pub.publish(
                self.actual_path
            )


    def _append_actual_pose(
        self,
        pos,
        quat,
        stamp,
    ):

        pose = PoseStamped()

        pose.header.stamp = stamp
        pose.header.frame_id = MAP_FRAME

        pose.pose.position.x = (
            float(pos[0])
        )

        pose.pose.position.y = (
            float(pos[1])
        )

        pose.pose.position.z = (
            float(pos[2])
        )

        pose.pose.orientation.x = (
            float(quat[1])
        )

        pose.pose.orientation.y = (
            float(quat[2])
        )

        pose.pose.orientation.z = (
            float(quat[3])
        )

        pose.pose.orientation.w = (
            float(quat[0])
        )

        self.actual_path.poses.append(
            pose
        )

        if (
            len(self.actual_path.poses)
            > MAX_PATH_POINTS
        ):

            self.actual_path.poses = (
                self.actual_path.poses[
                    -MAX_PATH_POINTS:
                ]
            )

        self.last_path_xy = (
            float(pos[0]),
            float(pos[1]),
        )


    def update_actual_path(
        self,
        pos,
        quat,
        stamp,
    ):

        current_xy = (
            float(pos[0]),
            float(pos[1]),
        )

        if self.last_path_xy is None:

            self._append_actual_pose(
                pos,
                quat,
                stamp,
            )

            self.actual_path.header.stamp = (
                stamp
            )

            self.path_pub.publish(
                self.actual_path
            )

            return

        dx = (
            current_xy[0]
            - self.last_path_xy[0]
        )

        dy = (
            current_xy[1]
            - self.last_path_xy[1]
        )

        moved = math.hypot(
            dx,
            dy,
        )

        if moved >= PATH_POINT_MIN_DISTANCE:

            self._append_actual_pose(
                pos,
                quat,
                stamp,
            )

            self.actual_path.header.stamp = (
                stamp
            )

            self.path_pub.publish(
                self.actual_path
            )


    # --------------------------------------------------------
    # Planned / reference trajectory
    # --------------------------------------------------------

    def publish_planned_path(
        self,
        start_pos,
        goal_pos,
    ):
        """
        Current controller has no global planner yet.

        Therefore planned_path is currently a straight line:
            robot -> clicked goal

        Later MPPI trajectory can replace this function directly.
        """

        sx = float(start_pos[0])
        sy = float(start_pos[1])
        sz = float(start_pos[2])

        gx = float(goal_pos[0])
        gy = float(goal_pos[1])

        dx = gx - sx
        dy = gy - sy

        distance = math.hypot(
            dx,
            dy,
        )

        heading = math.atan2(
            dy,
            dx,
        )

        qx, qy, qz, qw = (
            yaw_to_ros_quaternion(
                heading
            )
        )

        num_points = max(
            2,
            int(
                distance
                / PLANNED_PATH_RESOLUTION
            ) + 1,
        )

        path = Path()

        path.header.frame_id = MAP_FRAME

        path.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        for i in range(num_points):

            alpha = (
                i
                / float(num_points - 1)
            )

            pose = PoseStamped()

            pose.header.frame_id = MAP_FRAME
            pose.header.stamp = (
                path.header.stamp
            )

            pose.pose.position.x = (
                sx
                + alpha * dx
            )

            pose.pose.position.y = (
                sy
                + alpha * dy
            )

            # Keep path around body height in RViz
            pose.pose.position.z = sz

            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            path.poses.append(
                pose
            )

        self.planned_path_pub.publish(
            path
        )

        print(
            f"[ROS2] planned path published: "
            f"{num_points} poses, "
            f"{distance:.2f} m"
        )


# ============================================================
# 7. Policy
# ============================================================

def load_policy(
    checkpoint_path,
    device,
):

    print(
        f"[POLICY] Loading: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = (
        checkpoint[
            "model_state_dict"
        ]
    )

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
        f"[POLICY] {obs_dim} -> "
        f"{action_dim}"
    )

    return (
        actor,
        obs_dim,
        action_dim,
    )


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
# 8. Isaac goal marker
# ============================================================

def create_goal_marker():

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


# ============================================================
# 9. Click handler
# ============================================================

def setup_viewport_click(
    goal_marker,
    goal_marker_xform,
    robot,
    ros_node,
):

    (
        viewport_api,
        viewport_window,
    ) = get_active_viewport_and_window()

    if (
        viewport_api is None
        or viewport_window is None
    ):

        raise RuntimeError(
            "Active Viewport not found."
        )

    print(
        f"[CLICK] Viewport = "
        f"{viewport_api.resolution}"
    )


    def query_completed(
        prim_path,
        world_position,
        *args,
    ):

        global goal_xy
        global goal_active

        if (
            not prim_path
            or world_position is None
        ):

            print(
                "[PICK] No geometry."
            )
            return

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

        update_goal_marker(
            goal_marker,
            goal_marker_xform,
            gx,
            gy,
            gz,
        )

        # ====================================================
        # ROS: clicked goal
        # ====================================================

        ros_node.publish_goal(
            gx,
            gy,
            gz,
        )

        robot_pos = tensor_to_list(
            robot.data.root_pos_w[0]
        )

        robot_quat = tensor_to_list(
            robot.data.root_quat_w[0]
        )

        # Actual trajectory starts fresh for each new goal
        ros_node.reset_actual_path(
            robot_pos,
            robot_quat,
        )

        # Reference trajectory to clicked target
        ros_node.publish_planned_path(
            robot_pos,
            (
                gx,
                gy,
                gz,
            ),
        )

        print()
        print(
            "========================================"
        )
        print(" NEW GOAL")
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
            f" prim = {prim_path}"
        )
        print(
            "========================================"
        )
        print()


    def on_viewport_clicked(sender):

        mouse_ndc = (
            sender
            .gesture_payload
            .mouse
        )

        (
            pixel,
            in_viewport,
        ) = (
            viewport_api
            .map_ndc_to_texture_pixel(
                mouse_ndc
            )
        )

        if not in_viewport:

            return

        viewport_api.request_query(
            pixel,
            query_completed,
            query_name=(
                "go2_rviz_goal_query"
            ),
        )


    frame = (
        viewport_window
        .get_frame(
            "go2_rviz_click"
        )
    )

    with frame:

        scene_view = sc.SceneView(
            aspect_ratio_policy=(
                sc.AspectRatioPolicy.STRETCH
            )
        )

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

            screen = sc.Screen(
                gesture=click_gesture
            )

    print(
        "[CLICK] Click handler ready"
    )

    return {
        "viewport_api": viewport_api,
        "scene_view": scene_view,
        "frame": frame,
        "screen": screen,
        "gesture": click_gesture,
    }


# ============================================================
# 10. Keyboard emergency stop
# ============================================================

def keyboard_callback(event):

    global goal_active

    if (
        event.type
        == carb.input.KeyboardEventType.KEY_PRESS
    ):

        if event.input.name == "SPACE":

            goal_active = False

            print()
            print(
                "[NAV] GOAL CANCELLED / STOP"
            )
            print()

    return True


# ============================================================
# 11. Point-to-goal controller
# ============================================================

def compute_velocity_command(
    robot,
):

    global goal_xy
    global goal_active

    if (
        not goal_active
        or goal_xy is None
    ):

        return (
            0.0,
            0.0,
            0.0,
        )

    position = tensor_to_list(
        robot.data.root_pos_w[0]
    )

    quat = tensor_to_list(
        robot.data.root_quat_w[0]
    )

    robot_x = position[0]
    robot_y = position[1]

    robot_yaw = (
        quaternion_to_yaw(
            quat
        )
    )

    dx = (
        goal_xy[0]
        - robot_x
    )

    dy = (
        goal_xy[1]
        - robot_y
    )

    distance = math.hypot(
        dx,
        dy,
    )

    desired_yaw = math.atan2(
        dy,
        dx,
    )

    heading_error = wrap_to_pi(
        desired_yaw
        - robot_yaw
    )

    # --------------------------------------------------------
    # Reached goal
    # --------------------------------------------------------

    if distance < GOAL_TOLERANCE:

        goal_active = False

        print()
        print(
            "========================================"
        )
        print(" GOAL REACHED")
        print(
            f" final error = "
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
    # Yaw command
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
    # Forward command
    # --------------------------------------------------------

    if (
        abs(heading_error)
        > TURN_IN_PLACE_THRESHOLD
    ):

        vx = 0.0

    else:

        vx = min(
            MAX_VX,
            KP_LINEAR * distance,
        )

        vx *= max(
            0.0,
            math.cos(
                heading_error
            ),
        )

    return (
        vx,
        0.0,
        yaw_rate,
    )


# ============================================================
# 12. Main
# ============================================================

def main():

    global ros_bridge

    # ========================================================
    # Environment
    # ========================================================

    env_cfg = (
        UnitreeGo2RoughEnvCfg_PLAY()
    )

    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5

    # flat ground
    env_cfg.scene.terrain.terrain_type = (
        "plane"
    )

    env_cfg.scene.terrain.terrain_generator = (
        None
    )

    env_cfg.scene.terrain.max_init_terrain_level = (
        None
    )

    env_cfg.curriculum.terrain_levels = (
        None
    )

    # disable random commands
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

    env_cfg.observations.policy.enable_corruption = (
        False
    )

    env_cfg.events.push_robot = None

    env_cfg.events.base_external_force_torque = (
        None
    )

    # deterministic reset
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

    env_cfg.episode_length_s = (
        1.0e9
    )

    print(
        "[ENV] Creating GO2..."
    )

    env = gym.make(
        "Isaac-Velocity-Rough-Unitree-Go2-v0",
        cfg=env_cfg,
    )

    env = RslRlVecEnvWrapper(
        env
    )

    device = (
        env.unwrapped.device
    )

    robot = (
        env
        .unwrapped
        .scene[
            "robot"
        ]
    )

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

    # ========================================================
    # ROS2
    # ========================================================

    rclpy.init(
        args=None
    )

    ros_bridge = (
        Go2RvizBridge()
    )

    # ========================================================
    # Goal marker
    # ========================================================

    (
        goal_marker,
        goal_marker_xform,
    ) = create_goal_marker()

    # ========================================================
    # Policy
    # ========================================================

    (
        actor,
        expected_obs_dim,
        expected_action_dim,
    ) = load_policy(
        CHECKPOINT_PATH,
        device,
    )

    # ========================================================
    # Command manager
    # ========================================================

    command_term = (
        env
        .unwrapped
        .command_manager
        .get_term(
            "base_velocity"
        )
    )

    # ========================================================
    # Keyboard
    # ========================================================

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

    _ = keyboard_subscription

    # ========================================================
    # Viewport click
    # ========================================================

    click_handler = (
        setup_viewport_click(
            goal_marker,
            goal_marker_xform,
            robot,
            ros_bridge,
        )
    )

    # ========================================================
    # Initial observation
    # ========================================================

    obs = (
        env
        .get_observations()
    )

    policy_obs = (
        get_policy_observation(
            obs
        )
    )

    if (
        policy_obs.shape[-1]
        != expected_obs_dim
    ):

        raise RuntimeError(
            f"Observation mismatch: "
            f"{policy_obs.shape[-1]} "
            f"!= {expected_obs_dim}"
        )

    print()
    print(
        "========================================"
    )
    print(
        " GO2 CLICK + RVIZ READY"
    )
    print(
        "========================================"
    )
    print(
        "LEFT CLICK : new goal"
    )
    print(
        "SPACE      : stop"
    )
    print()
    print(
        "RViz Fixed Frame = map"
    )
    print()
    print(
        "/planned_path = reference"
    )
    print(
        "/path         = actual trajectory"
    )
    print(
        "========================================"
    )
    print()

    dt = (
        env.unwrapped.step_dt
    )

    status_counter = 0

    try:

        while simulation_app.is_running():

            start_time = (
                time.time()
            )

            # =================================================
            # Goal navigation
            # =================================================

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

            command_term.vel_command_b[:] = (
                command
            )

            # =================================================
            # GO2 locomotion policy
            # =================================================

            with torch.inference_mode():

                policy_obs = (
                    get_policy_observation(
                        obs
                    )
                )

                policy_input = (
                    policy_obs.clone()
                )

                # command observation
                policy_input[
                    :,
                    9:12
                ] = command

                actions = actor(
                    policy_input
                )

                (
                    obs,
                    _,
                    _,
                    _,
                ) = env.step(
                    actions
                )

            # =================================================
            # ROS2
            # =================================================

            ros_bridge.publish_robot_state(
                robot
            )

            # process ROS callbacks / discovery
            rclpy.spin_once(
                ros_bridge,
                timeout_sec=0.0,
            )

            # =================================================
            # Console navigation status
            # =================================================

            status_counter += 1

            if (
                goal_active
                and goal_xy is not None
                and status_counter % 25 == 0
            ):

                position = tensor_to_list(
                    robot.data.root_pos_w[0]
                )

                distance = math.hypot(

                    goal_xy[0]
                    - position[0],

                    goal_xy[1]
                    - position[1],
                )

                print(
                    f"[NAV] "
                    f"pos=("
                    f"{position[0]:+.2f}, "
                    f"{position[1]:+.2f}) "
                    f"goal=("
                    f"{goal_xy[0]:+.2f}, "
                    f"{goal_xy[1]:+.2f}) "
                    f"dist={distance:.2f} "
                    f"cmd=("
                    f"{vx:+.2f}, "
                    f"{vy:+.2f}, "
                    f"{yaw_rate:+.2f})"
                )

            # =================================================
            # Real-time pacing
            # =================================================

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

        # =====================================================
        # Cleanup viewport
        # =====================================================

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
                f"[WARN] viewport cleanup: "
                f"{e}"
            )

        # =====================================================
        # Cleanup ROS2
        # =====================================================

        try:

            ros_bridge.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()

        except Exception:
            pass

        env.close()


# ============================================================
# 13. Entry
# ============================================================

if __name__ == "__main__":

    try:
        main()

    finally:
        simulation_app.close()
