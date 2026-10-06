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
        self._visual_car_parts = []
        self._visual_wheels = []
        self._wheel_spin_rad = 0.0
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
        self._create_visual_car()

    def _create_visual_car(self) -> None:
        """Create a non-colliding, wheeled rover for the chase view."""
        if mn is None:
            raise RuntimeError("Habitat-Sim Magnum bindings are unavailable")
        templates = self._sim.get_object_template_manager()
        objects = self._sim.get_rigid_object_manager()
        # The rover is deliberately visual-only. The simulator's configured
        # agent footprint remains authoritative for navigation and collision.
        parts = (
            ("chassis", "cubeSolid", (0.0, 0.16, 0.0), (0.34, 0.12, 0.29), None),
            ("cover", "cubeSolid", (0.0, 0.33, 0.02), (0.30, 0.045, 0.25), None),
            ("camera_mount", "cubeSolid", (0.0, 0.43, -0.23), (0.08, 0.055, 0.08), None),
            ("camera_lens", "cylinderSolid_rings_1_segments_12_halfLen_1_useTexCoords_false_useTangents_false_capEnds_true", (0.0, 0.43, -0.32), (0.045, 0.045, 0.045), "lens"),
        )
        for name, primitive, offset, scale, _ in parts:
            template = templates.get_template_by_handle(primitive)
            template.scale = mn.Vector3(*scale)
            template.is_collidable = False
            handle = f"xjtlu_visual_car_{name}"
            templates.register_template(template, handle)
            obj = objects.add_object_by_template_handle(handle)
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            obj.collidable = False
            self._visual_car_parts.append((obj, mn.Vector3(*offset)))
        wheel_template = "cylinderSolid_rings_1_segments_12_halfLen_1_useTexCoords_false_useTangents_false_capEnds_true"
        wheel_specs = (
            ("wheel_left_front", (-0.34, 0.10, -0.20)),
            ("wheel_right_front", (0.34, 0.10, -0.20)),
            ("wheel_left_rear", (-0.34, 0.10, 0.20)),
            ("wheel_right_rear", (0.34, 0.10, 0.20)),
        )
        for name, offset in wheel_specs:
            template = templates.get_template_by_handle(wheel_template)
            template.scale = mn.Vector3(0.14, 0.085, 0.14)
            template.is_collidable = False
            handle = f"xjtlu_visual_car_{name}"
            templates.register_template(template, handle)
            obj = objects.add_object_by_template_handle(handle)
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            obj.collidable = False
            self._visual_wheels.append((obj, mn.Vector3(*offset)))
        self._position_visual_car(visible=False)

    def _position_visual_car(self, visible: bool) -> None:
        """Move the car to the agent pose or park it outside the scene."""
        if not self._visual_car_parts:
            return
        rotation = mn.Quaternion.rotation(mn.Rad(self.pose.yaw), mn.Vector3.y_axis())
        base = mn.Vector3(self.pose.x, self.pose.z, self.pose.y)
        for obj, offset in self._visual_car_parts:
            obj.translation = base + rotation.transform_vector(offset) if visible else mn.Vector3(0.0, -1000.0, 0.0)
            obj.rotation = rotation
        wheel_base_rotation = mn.Quaternion.rotation(mn.Rad(math.pi / 2.0), mn.Vector3.z_axis())
        wheel_spin = mn.Quaternion.rotation(mn.Rad(self._wheel_spin_rad), mn.Vector3.x_axis())
        for obj, offset in self._visual_wheels:
            obj.translation = base + rotation.transform_vector(offset) if visible else mn.Vector3(0.0, -1000.0, 0.0)
            obj.rotation = rotation * wheel_base_rotation * wheel_spin

    def _capture_views(self) -> tuple[np.ndarray, np.ndarray]:
        # The policy front camera never sees the visual car. The second render
        # supplies only the chase-camera frame used by the video compositor.
        self._position_visual_car(visible=False)
        front = self._sim.get_sensor_observations()["color_sensor"][:, :, :3].copy()
        self._position_visual_car(visible=True)
        chase = self._sim.get_sensor_observations()["third_person_sensor"][:, :, :3].copy()
        return front, chase

    def reset(self, init_pose: Optional[SimAgentPose] = None) -> SimObservation:
        self.pose = init_pose if init_pose is not None else SimAgentPose()
        self._step_counter = 0
        self._wheel_spin_rad = 0.0
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
        # Wheel radius is 0.10 m. The sign is chosen so forward motion turns
        # the visible wheels in the rolling direction.
        self._wheel_spin_rad -= float(linear_velocity) * float(dt) / 0.10

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
