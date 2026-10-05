#!/usr/bin/env python3
"""Run closed-loop simulation of SpatialReasoningPolicy.

Demonstrates end-to-end perception -> spatial reasoning -> safety -> evaluator pipeline:
1. Environment generates limited-FOV low-height (0.45m) RGB frames.
2. SpatialReasoningPolicy leverages 8-sector topological memory, language priors,
   and visual grounding to infer target direction and navigate.
3. ActionAdapter converts policy action to stamped Twist format and latches stop condition.
4. SafetyFilter enforces vehicle velocity bounds and acceleration constraints.
5. EpisodeManager governs episode lifecycle and latched completion.
6. Evaluator computes path efficiency, SPL, and success metrics.
"""

import math
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/vln_core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/vln_policy")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/vln_sim")))

from vln_core.action_adapter import ActionAdapter
from vln_core.episode_manager import EpisodeManager, EpisodeManagerConfig
from vln_core.episode_scenario import EpisodeScenario, SuccessRegion
from vln_core.evaluator import EpisodeResult, VlnEvaluator
from vln_core.safety_filter import SafetyFilter, SafetyFilterConfig
from vln_policy.spatial_reasoning_policy import ReasoningState, SpatialReasoningConfig, SpatialReasoningPolicy
from vln_sim.bridge_core import SimAgentPose
from vln_sim.mock_scene_adapter import MockSceneAdapter


def run_simulation(
    episode_id: str = "spatial_sim_01",
    instruction: str = "navigate straight through the corridor to find the doorway",
    init_yaw_rad: float = 0.25,
    dt: float = 0.1,
    max_steps: int = 150,
):
    print("=" * 105)
    print("         XJTLU VLN-E2E - Limited-FOV Spatial Reasoning Simulation")
    print("=" * 105)
    print(f" Episode ID   : {episode_id}")
    print(f" Instruction  : \"{instruction}\"")
    print(f" Camera Mount : Front-facing, Height = 0.45m, HFOV = 90.0°")
    print(f" Initial Pose : x=0.00m, y=0.00m, yaw={init_yaw_rad:+.2f} rad ({math.degrees(init_yaw_rad):+.1f}°)")
    print(f" Timestep dt  : {dt}s")
    print("-" * 105)
    print(f"{'Step':>5} | {'Sim Time':>8} | {'State':^16} | {'Pose (x, y, yaw)':^23} | {'Raw (v, w)':^16} | {'Safe (v, w)':^16} | {'p_stop':>6}")
    print("-" * 105)

    sim = MockSceneAdapter(width=640, height=480)
    obs = sim.reset(SimAgentPose(x=0.0, y=0.0, yaw=init_yaw_rad))

    policy = SpatialReasoningPolicy(SpatialReasoningConfig())
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

    manager = EpisodeManager(EpisodeManagerConfig(max_duration_sec=30.0, max_steps=max_steps))
    manager.start_episode(episode_id=episode_id, instruction=instruction, monotonic_now=0.0)

    scenario = EpisodeScenario(
        scenario_id="corridor_doorway_01",
        instruction=instruction,
        start_x=0.0,
        start_y=0.0,
        start_yaw=init_yaw_rad,
        success_region=SuccessRegion(center_x=3.0, center_y=0.0, radius=0.8),
    )


    trajectory: List[Tuple[float, float]] = [(sim.pose.x, sim.pose.y)]
    velocity_commands: List[Tuple[float, float]] = []
    inference_latencies_ms: List[float] = []
    collision_count = 0

    sim_time = 0.0
    completed = False
    termination_reason = "max_steps"

    for step in range(1, max_steps + 1):
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


        pose_str = f"({sim.pose.x:5.2f}, {sim.pose.y:5.2f}, {math.degrees(sim.pose.yaw):+5.1f}°)"
        raw_str = f"({act.linear_velocity:4.2f}, {act.angular_velocity:+4.2f})"
        safe_str = f"({safe_twist.linear_x:4.2f}, {safe_twist.angular_z:+4.2f})"
        state_str = info.get("state", policy.state.value)

        if step <= 5 or step % 5 == 0 or finished or policy.state in (ReasoningState.VERIFY, ReasoningState.STOP):
            print(f"{step:5d} | {sim_time:7.2f}s | {state_str:^16} | {pose_str:^23} | {raw_str:^16} | {safe_str:^16} | {act.stop_probability:6.2f}")

        if finished or policy.state == ReasoningState.STOP:
            completed = True
            termination_reason = "success_stopped" if policy.state == ReasoningState.STOP else reason
            print("-" * 105)
            print(f">>> EPISODE COMPLETED at step {step} (t={sim_time:.2f}s, reason={termination_reason})!")
            break

        obs = sim.step(safe_twist.linear_x, safe_twist.angular_z, dt)
        sim_time += dt


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
    run_simulation()
