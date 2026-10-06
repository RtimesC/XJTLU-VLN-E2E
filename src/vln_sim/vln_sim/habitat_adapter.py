"""Real Habitat-Sim adapter wrapping habitat_sim.Simulator."""

import os
import time
import math
from typing import Optional
import numpy as np

try:
    import magnum as mn
    from magnum import Vector3
except ImportError:  # pragma: no cover - only exercised in Habitat-Sim envs
    mn = None
    Vector3 = None

from .bridge_core import BaseSimAdapter, SimAgentPose, SimObservation, integrate_differential_drive

try:
    import habitat_sim
except ImportError:
    habitat_sim = None


class HabitatSimAdapter(BaseSimAdapter):
    """Adapter interfacing directly with Facebook AI Habitat-Sim."""

    AGENT_RADIUS_M = 0.38625
    AGENT_HEIGHT_M = 0.5

    def __init__(
        self,
        scene_path: str,
        width: int = 640,
        height: int = 480,
        hfov: float = 90.0,
        sensor_height: float = 0.45,
    ):
        if habitat_sim is None:
            raise RuntimeError(
                "habitat_sim is not installed in the active environment. "
                "Please install habitat-sim (via conda install habitat-sim withbullet -c aihabitat) "
                "or use MockSceneAdapter."
            )
        if (width, height) != (640, 480):
            raise ValueError("XJTLU camera resolution is fixed at 640x480")
        if abs(float(hfov) - 90.0) > 1e-9:
            raise ValueError("XJTLU camera HFOV is fixed at 90 degrees")
        if abs(float(sensor_height) - 0.45) > 1e-9:
            raise ValueError("XJTLU camera height is fixed at 0.45 m")

        self.scene_path = scene_path
        self.width = width
        self.height = height
        self.hfov = hfov
        self.sensor_height = sensor_height

        self._sim = None
        self._spot_robot = None
        self._step_counter = 0
        self.pose = SimAgentPose()
        self._init_sim()

    def _init_sim(self):
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path
        backend_cfg.enable_physics = False

        # Visual camera sensor (matching physical car camera height 0.45m)
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.uuid = "color_sensor"
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.resolution = [self.height, self.width]
        camera_sensor_spec.position = [0.0, self.sensor_height, 0.0]
        camera_sensor_spec.hfov = self.hfov

        # Visualization-only chase camera. The policy still receives only color_sensor.
        third_person_spec = habitat_sim.CameraSensorSpec()
        third_person_spec.uuid = "third_person_sensor"
        third_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        third_person_spec.resolution = [self.height, self.width]
        # Close, low chase view so the rover body and wheels are legible.
        third_person_spec.position = [0.0, 1.35, 2.15]
        third_person_spec.orientation = [math.radians(-22.0), 0.0, 0.0]
        third_person_spec.hfov = self.hfov

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        # Immutable low-profile vehicle footprint; do not inherit Habitat's
        # human-sized defaults.
        agent_cfg.radius = self.AGENT_RADIUS_M
        agent_cfg.height = self.AGENT_HEIGHT_M
        agent_cfg.sensor_specifications = [camera_sensor_spec, third_person_spec]

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self._sim = habitat_sim.Simulator(cfg)
        self._create_visual_spot()

    def _create_visual_spot(self) -> None:
        """Load Habitat-Lab's official Spot articulated robot for chase view."""
        if mn is None:
            raise RuntimeError("Habitat-Sim Magnum bindings are unavailable")
        urdf_path = os.environ.get(
            "XJTLU_SPOT_URDF",
            "/home/sousuke/Desktop/habitat-lab/data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf",
        )
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(
                "SpotRobot asset not found: " + urdf_path +
                ". Download the official hab_spot_arm dataset first."
            )
        self._spot_robot = self._sim.get_articulated_object_manager().add_articulated_object_from_urdf(
            urdf_path,
            fixed_base=True,
            maintain_link_order=True,
        )
        self._position_visual_spot(visible=False)

    def _position_visual_spot(self, visible: bool) -> None:
        """Move the articulated Spot model with the simulated agent."""
        if self._spot_robot is None:
            return
        if not visible:
            self._spot_robot.translation = mn.Vector3(0.0, -1000.0, 0.0)
            return
        self._spot_robot.translation = mn.Vector3(self.pose.x, self.pose.z, self.pose.y)
        self._spot_robot.rotation = mn.Quaternion.rotation(
            mn.Rad(self.pose.yaw), mn.Vector3.y_axis()
        )

    def _capture_views(self) -> tuple[np.ndarray, np.ndarray]:
        # The policy front camera never sees the visual car. The second render
        # supplies only the chase-camera frame used by the video compositor.
        self._position_visual_spot(visible=False)
        front = self._sim.get_sensor_observations()["color_sensor"][:, :, :3].copy()
        self._position_visual_spot(visible=True)
        chase = self._sim.get_sensor_observations()["third_person_sensor"][:, :, :3].copy()
        return front, chase

    def reset(self, init_pose: Optional[SimAgentPose] = None) -> SimObservation:
        self.pose = init_pose if init_pose is not None else SimAgentPose()
        self._step_counter = 0
        obs = self._sim.reset()
        if init_pose is not None and self._sim is not None:
            agent = self._sim.get_agent(0)
            state = agent.get_state()
            state.position = np.array([self.pose.x, self.pose.z, self.pose.y])
            half_yaw = self.pose.yaw / 2.0
            if hasattr(np, 'quaternion'):
                state.rotation = np.quaternion(np.cos(half_yaw), 0, np.sin(half_yaw), 0)
            agent.set_state(state)
            obs = self._sim.get_sensor_observations()
        rgb, third_person_rgb = self._capture_views()
        return SimObservation(
            rgb=rgb,
            timestamp_sec=time.time(),
            step_index=0,
            third_person_rgb=third_person_rgb,
        )

    def is_navigable(self, pose: SimAgentPose) -> bool:
        """Check the vehicle base position against Habitat's navmesh."""
        if Vector3 is None:
            raise RuntimeError("Habitat-Sim Magnum bindings are unavailable")
        point = Vector3(float(pose.x), float(pose.z), float(pose.y))
        return bool(self._sim.pathfinder.is_navigable(point))

    def sample_navigable_pose(self, yaw: float = 0.0) -> SimAgentPose:
        """Sample a valid low-profile vehicle pose from the loaded navmesh."""
        point = self._sim.pathfinder.get_random_navigable_point()
        return SimAgentPose(x=float(point[0]), y=float(point[2]), z=float(point[1]), yaw=float(yaw))


    def step(self, linear_velocity: float, angular_velocity: float, dt: float) -> SimObservation:
        self.pose = integrate_differential_drive(self.pose, linear_velocity, angular_velocity, dt)
        self._step_counter += 1

        # Move agent in Habitat
        agent = self._sim.get_agent(0)
        state = agent.get_state()
        state.position = np.array([self.pose.x, self.pose.z, self.pose.y])
        # Convert yaw to quaternion
        half_yaw = self.pose.yaw / 2.0
        state.rotation = np.quaternion(np.cos(half_yaw), 0, np.sin(half_yaw), 0)
        agent.set_state(state)

        rgb, third_person_rgb = self._capture_views()
        return SimObservation(
            rgb=rgb,
            timestamp_sec=time.time(),
            step_index=self._step_counter,
            third_person_rgb=third_person_rgb,
        )

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None
