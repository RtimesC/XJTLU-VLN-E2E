#!/usr/bin/env python3
"""Run closed-loop simulation of SpatialReasoningPolicy with real Habitat 3D rendering.

Supports both:
- Real Habitat-Sim 3D environment (loading real .glb scenes with photorealistic textures)
- MockSceneAdapter fallback for headless Mac/CI environments without habitat-sim.
"""

import argparse
import math
import os
import sys
from typing import List, Tuple
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/vln_core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/vln_policy")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/vln_sim")))

from vln_core.action_adapter import ActionAdapter
from vln_core.episode_manager import EpisodeManager, EpisodeManagerConfig
from vln_core.episode_scenario import EpisodeScenario, SuccessRegion
from vln_core.evaluator import EpisodeResult, VlnEvaluator
from vln_core.safety_filter import SafetyFilter, SafetyFilterConfig
from vln_policy.spatial_reasoning_policy import (
    ReasoningState,
    SpatialReasoningConfig,
    SpatialReasoningPolicy,
)
from vln_sim.bridge_core import SimAgentPose
from vln_sim.mock_scene_adapter import MockSceneAdapter
from vln_sim.habitat_adapter import HabitatSimAdapter, habitat_sim


def render_hud_overlay(
    rgb: np.ndarray,
    step: int,
    sim_time: float,
    state_str: str,
    linear_v: float,
    angular_w: float,
    stop_prob: float,
    pose: SimAgentPose,
    instruction: str,
    goal_dist: float,
    sectors: list,
    info: dict,
    backend_name: str,
) -> np.ndarray:
    """Renders a crisp, high-definition HUD overlay on top of the first-person frame."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # Top header bar (translucent dark)
    header_overlay = bgr.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, 54), (15, 15, 18), -1)
    cv2.addWeighted(header_overlay, 0.70, bgr, 0.30, 0, bgr)

    cv2.putText(
        bgr,
        f"XJTLU VLN-E2E | Backend: {backend_name} | Camera: Front (h=0.45m, 90 HFOV)",
        (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 200, 100),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        f'Task: "{instruction}"',
        (12, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 240, 255),
        1,
        cv2.LINE_AA,
    )

    # Bottom-left telemetry display
    box_overlay = bgr.copy()
    cv2.rectangle(box_overlay, (10, h - 145), (295, h - 10), (15, 15, 18), -1)
    cv2.addWeighted(box_overlay, 0.70, bgr, 0.30, 0, bgr)
    cv2.rectangle(bgr, (10, h - 145), (295, h - 10), (80, 80, 85), 1)

    state_color = (0, 220, 255)
    if state_str == "APPROACH":
        state_color = (80, 255, 80)
    elif state_str in ("VERIFY", "STOP"):
        state_color = (50, 120, 255)
    elif state_str == "ORIENT":
        state_color = (255, 160, 50)

    cv2.putText(bgr, f"STATE: {state_str}", (18, h - 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2, cv2.LINE_AA)
    cv2.putText(bgr, f"Time: {sim_time:4.1f}s | Step: {step:3d}", (18, h - 98), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"Cmd: v={linear_v:4.2f}m/s, w={angular_w:+4.2f}r/s", (18, h - 76), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"Pose: ({pose.x:.2f}m, {pose.y:.2f}m, {math.degrees(pose.yaw):+.1f} deg)", (18, h - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"Goal Dist: {goal_dist:.2f}m | p_stop: {stop_prob:.2f}", (18, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 230, 120), 1, cv2.LINE_AA)

    # 8-Sector Memory Mini-Radar (Top Right)
    radar_center = (w - 65, 80)
    radar_r = 38
    radar_overlay = bgr.copy()
    cv2.circle(radar_overlay, radar_center, radar_r + 8, (15, 15, 18), -1)
    cv2.addWeighted(radar_overlay, 0.70, bgr, 0.30, 0, bgr)
    cv2.circle(bgr, radar_center, radar_r + 8, (100, 100, 105), 1)
    cv2.putText(bgr, "SECTOR RADAR", (w - 110, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    for s in sectors:
        angle_deg = s.center_angle_deg - 90.0
        rad = math.radians(angle_deg)
        x_end = int(radar_center[0] + radar_r * math.cos(rad))
        y_end = int(radar_center[1] + radar_r * math.sin(rad))

        prob = min(max(s.target_probability, 0.0), 1.0)
        c_val = int(prob * 8.0 * 255)
        c_val = min(max(c_val, 40), 255)
        sec_color = (50, c_val, 255 - c_val)

        cv2.line(bgr, radar_center, (x_end, y_end), sec_color, 2)
        cv2.circle(bgr, (x_end, y_end), 3, sec_color, -1)

    cv2.arrowedLine(
        bgr,
        (radar_center[0], radar_center[1] + 8),
        (radar_center[0], radar_center[1] - 12),
        (255, 255, 255),
        2,
        tipLength=0.35,
    )

    if info.get("target_detected", False):
        cv2.putText(
            bgr,
            f"TARGET DETECTED ({info.get('target_confidence', 0.0):.2f})",
            (w // 2 - 120, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return bgr


def compose_video_frame(front_hud_bgr: np.ndarray, third_person_rgb: np.ndarray) -> np.ndarray:
    """Compose front-view HUD and visualization-only third-person view side by side."""
    third_bgr = cv2.cvtColor(third_person_rgb, cv2.COLOR_RGB2BGR)
    third_bgr = cv2.resize(third_bgr, (front_hud_bgr.shape[1], front_hud_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(third_bgr, (0, 0), (third_bgr.shape[1] - 1, third_bgr.shape[0] - 1), (80, 220, 255), 2)
    cv2.rectangle(third_bgr, (0, 0), (250, 32), (15, 15, 18), -1)
    cv2.putText(third_bgr, "THIRD-PERSON (VISUALIZATION)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 255), 1, cv2.LINE_AA)
    return np.hstack([front_hud_bgr, third_bgr])


def run_simulation(
    episode_id: str = "spatial_sim_01",
    instruction: str = "navigate straight through the corridor to find the doorway",
    init_yaw_rad: float = 0.25,
    dt: float = 0.1,
    max_steps: int = 250,
    scene_path: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    output_video_path: str = "recordings/spatial_reasoning_sim.mp4",
):
    print("=" * 105)
    print("         XJTLU VLN-E2E - Limited-FOV Spatial Reasoning Simulation (3D HUD Video)")
    print("=" * 105)
    print(f" Episode ID   : {episode_id}")
    print(f" Instruction  : \"{instruction}\"")
    print(f" Camera Mount : Front-facing, Height = 0.45m, HFOV = 90.0°")
    print(f" Output Video : {output_video_path}")
    print(f" Timestep dt  : {dt}s | Max Steps: {max_steps}")
    print("-" * 105)

    # Automatically choose between real Habitat 3D simulator and Mock fallback
    has_habitat = habitat_sim is not None and os.path.exists(scene_path)
    if has_habitat:
        backend_name = "Habitat-Sim 3D"
        print(f"[*] Initializing real 3D Habitat renderer with scene: {scene_path}")
        sim = HabitatSimAdapter(
            scene_path=scene_path,
            width=640,
            height=480,
            hfov=90.0,
            sensor_height=0.45,
        )
    else:
        backend_name = "Mock Synthetic"
        reason = "habitat_sim not installed" if habitat_sim is None else f"scene file not found ({scene_path})"
        print(f"[*] Fallback to MockSceneAdapter ({reason}).")
        sim = MockSceneAdapter(width=640, height=480)

    print(f"[*] Active Backend: {backend_name}")
    print(f"{'Step':>5} | {'Sim Time':>8} | {'State':^16} | {'Pose (x, y, yaw)':^23} | {'Raw (v, w)':^16} | {'Safe (v, w)':^16} | {'p_stop':>6}")
    print("-" * 105)

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 10.0, (1280, 480))

    obs = sim.reset(SimAgentPose(x=0.0, y=0.0, yaw=init_yaw_rad))

    policy = SpatialReasoningPolicy(SpatialReasoningConfig(max_episode_steps=max_steps))
    policy.reset(episode_id=episode_id)

    adapter = ActionAdapter()
    adapter.reset_episode(episode_id=episode_id)

    safety = SafetyFilter(
        SafetyFilterConfig(
            max_linear_velocity=0.6,
            max_angular_velocity=0.8,
            max_linear_accel=1.5,
            max_angular_accel=2.0,
        )
    )

    manager = EpisodeManager(EpisodeManagerConfig(max_duration_sec=35.0, max_steps=max_steps))
    manager.start_episode(episode_id=episode_id, instruction=instruction, monotonic_now=0.0)

    goal_x, goal_y = 3.0, 0.0
    scenario = EpisodeScenario(
        scenario_id="corridor_doorway_01",
        instruction=instruction,
        start_x=0.0,
        start_y=0.0,
        start_yaw=init_yaw_rad,
        success_region=SuccessRegion(center_x=goal_x, center_y=goal_y, radius=0.8),
    )

    trajectory: List[Tuple[float, float]] = [(sim.pose.x, sim.pose.y)]
    velocity_commands: List[Tuple[float, float]] = []
    inference_latencies_ms: List[float] = []
    collision_count = 0

    sim_time = 0.0
    completed = False
    termination_reason = "max_steps"

    for step in range(1, max_steps + 1):
        goal_dist = math.hypot(sim.pose.x - goal_x, sim.pose.y - goal_y)

        # 1. Perception & Spatial Reasoning
        act, info = policy.step(obs.rgb, instruction=instruction, episode_id=episode_id)
        inference_latencies_ms.append(act.inference_latency_ms)

        # 2. Action Adaptation
        twist, is_stop_triggered, _ = adapter.process_action(act)

        # 3. Safety Filter
        safe_twist, safety_status = safety.filter_command(
            twist,
            episode_id=act.episode_id,
            action_sequence_id=act.sequence_id,
            current_time_monotonic=sim_time,
        )
        velocity_commands.append((safe_twist.linear_x, safe_twist.angular_z))

        # 4. Episode Manager Governance
        finished, reason = manager.update_action(
            sequence_id=act.sequence_id,
            stop_probability=act.stop_probability,
            is_latched_stopped=adapter.is_stopped,
            monotonic_now=sim_time,
        )

        trajectory.append((sim.pose.x, sim.pose.y))
        if getattr(obs, "is_collision", False):
            collision_count += 1

        state_str = info.get("state", policy.state.value)

        # 5. Render HUD Overlay frame to video
        hud_frame = render_hud_overlay(
            rgb=obs.rgb,
            step=step,
            sim_time=sim_time,
            state_str=state_str,
            linear_v=safe_twist.linear_x,
            angular_w=safe_twist.angular_z,
            stop_prob=act.stop_probability,
            pose=sim.pose,
            instruction=instruction,
            goal_dist=goal_dist,
            sectors=policy.memory.sectors,
            info=info,
            backend_name=backend_name,
        )
        third_person_rgb = obs.third_person_rgb
        if third_person_rgb is None:
            third_person_rgb = np.zeros_like(obs.rgb)
        video_writer.write(compose_video_frame(hud_frame, third_person_rgb))

        pose_str = f"({sim.pose.x:5.2f}, {sim.pose.y:5.2f}, {math.degrees(sim.pose.yaw):+5.1f}°)"
        raw_str = f"({act.linear_velocity:4.2f}, {act.angular_velocity:+4.2f})"
        safe_str = f"({safe_twist.linear_x:4.2f}, {safe_twist.angular_z:+4.2f})"

        if step <= 5 or step % 10 == 0 or finished or policy.state in (ReasoningState.VERIFY, ReasoningState.STOP):
            print(f"{step:5d} | {sim_time:7.2f}s | {state_str:^16} | {pose_str:^23} | {raw_str:^16} | {safe_str:^16} | {act.stop_probability:6.2f}")

        if finished or policy.state == ReasoningState.STOP:
            completed = True
            termination_reason = "success_stopped" if policy.state == ReasoningState.STOP else reason
            print("-" * 105)
            print(f">>> EPISODE COMPLETED at step {step} (t={sim_time:.2f}s, reason={termination_reason})!")
            break

        obs = sim.step(safe_twist.linear_x, safe_twist.angular_z, dt)
        sim_time += dt

    video_writer.release()
    print(f"\n[+] Video saved to: {os.path.abspath(output_video_path)}")

    result = EpisodeResult(
        scenario_id=scenario.scenario_id,
        episode_id=episode_id,
        completed=completed,
        termination_reason=termination_reason,
        final_x=sim.pose.x,
        final_y=sim.pose.y,
        final_yaw=sim.pose.yaw,
        total_steps=step,
        elapsed_time_sec=sim_time,
        trajectory=trajectory,
        collision_count=collision_count,
        inference_latencies_ms=inference_latencies_ms,
        velocity_commands=velocity_commands,
    )

    evaluator = VlnEvaluator()
    metrics = evaluator.evaluate_episode(result, scenario)

    print("\n" + "=" * 55)
    print("             EVALUATION METRICS REPORT")
    print("=" * 55)
    print(f" Success (SR)          : {'YES' if metrics.success else 'NO'}")
    print(f" SPL                   : {metrics.spl:.4f}")
    print(f" Path Length           : {metrics.path_length_m:.2f} m")
    print(f" Shortest Path         : {metrics.shortest_path_m:.2f} m")
    print(f" Final Distance to Goal: {metrics.stop_distance_m:.2f} m")
    print(f" Collision Count       : {metrics.collision_count}")
    print(f" Inference Latency p95 : {metrics.inference_latency_p95_ms:.2f} ms")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="data/scene_datasets/habitat-test-scenes/skokloster-castle.glb")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--output", type=str, default="recordings/spatial_reasoning_sim.mp4")
    args = parser.parse_args()

    run_simulation(max_steps=args.steps, scene_path=args.scene, output_video_path=args.output)
