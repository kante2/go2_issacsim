"""Minimal Unitree GO2 spawn test for Isaac Sim 5.1 + Isaac Lab 2.3."""

import argparse

from isaaclab.app import AppLauncher


# ---------------------------------------------------------
# 1. Isaac Sim 실행 옵션
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Spawn Unitree GO2")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ---------------------------------------------------------
# 2. Isaac Sim이 켜진 뒤 Isaac Lab 모듈 import
# ---------------------------------------------------------
import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import UNITREE_GO2_CFG


def design_scene():
    """Ground + Light + GO2 생성."""

    # 바닥 생성
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    # 조명 생성
    light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/Light", light_cfg)

    # GO2 configuration 복사
    go2_cfg = UNITREE_GO2_CFG.copy()

    # GO2가 생성될 USD Prim 위치
    go2_cfg.prim_path = "/World/Robot"

    # 실제 Articulation 생성
    go2 = Articulation(cfg=go2_cfg)

    return go2


def main():
    # -----------------------------------------------------
    # 3. Physics simulation 생성
    # -----------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
    )

    sim = SimulationContext(sim_cfg)

    # 카메라 위치
    sim.set_camera_view(
        eye=[2.5, 2.5, 1.8],
        target=[0.0, 0.0, 0.4],
    )

    # -----------------------------------------------------
    # 4. Scene 생성
    # -----------------------------------------------------
    go2 = design_scene()

    # Physics 시작
    sim.reset()

    print("\n====================================")
    print("GO2 SPAWN SUCCESS")
    print("Number of joints :", go2.num_joints)
    print("Joint names      :", go2.joint_names)
    print("====================================\n")

    # -----------------------------------------------------
    # 5. GO2 초기 자세 설정
    # -----------------------------------------------------
    root_state = go2.data.default_root_state.clone()

    go2.write_root_pose_to_sim(root_state[:, :7])
    go2.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = go2.data.default_joint_pos.clone()
    joint_vel = go2.data.default_joint_vel.clone()

    go2.write_joint_state_to_sim(joint_pos, joint_vel)

    go2.reset()

    sim_dt = sim.get_physics_dt()

    # -----------------------------------------------------
    # 6. Simulation Loop
    # -----------------------------------------------------
    while simulation_app.is_running():

        # 기본 관절 위치 유지
        go2.set_joint_position_target(go2.data.default_joint_pos)

        # 명령을 PhysX에 전달
        go2.write_data_to_sim()

        # Physics 1 step
        sim.step()

        # Isaac Lab 내부 상태 업데이트
        go2.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
