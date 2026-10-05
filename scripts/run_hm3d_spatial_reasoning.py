#!/usr/bin/env python3
"""Run the spatial reasoning policy on a real HM3D PointNav smoke episode.

The HM3D dataset supplies the scene, start pose, goal position, and geodesic
distance. The policy still receives only the front RGB camera; the evaluator
uses the episode goal and simulator-provided geodesic distance.
"""

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/vln_core"))
sys.path.insert(0, str(ROOT / "src/vln_policy"))
sys.path.insert(0, str(ROOT / "src/vln_sim"))

from vln_core.episode_scenario import EpisodeScenario, SuccessRegion
from vln_sim.bridge_core import SimAgentPose
from run_spatial_reasoning_sim_loop import run_simulation


def load_episode(path: str, episode_index: int) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    episodes = data.get("episodes", [])
    if not episodes:
        raise ValueError(f"No episodes found in {path}")
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(f"episode index {episode_index} outside [0, {len(episodes) - 1}]")
    return episodes[episode_index]


def dataset_quaternion_to_sim_yaw(rotation: Any) -> float:
    """Convert HM3D [x, y, z, w] yaw to the runner's +x-forward convention."""
    if len(rotation) != 4:
        raise ValueError(f"Expected quaternion [x, y, z, w], got {rotation}")
    theta = 2.0 * math.atan2(float(rotation[1]), float(rotation[3]))
    return math.atan2(math.sin(-math.pi / 2.0 - theta), math.cos(-math.pi / 2.0 - theta))


def build_scenario(episode: Dict[str, Any], radius: float) -> tuple[SimAgentPose, EpisodeScenario]:
    start = [float(v) for v in episode["start_position"]]
    goal = [float(v) for v in episode["goals"][0]["position"]]
    start_pose = SimAgentPose(
        x=start[0],
        y=start[2],
        z=start[1],
        yaw=dataset_quaternion_to_sim_yaw(episode["start_rotation"]),
    )
    scenario = EpisodeScenario(
        scenario_id=f"hm3d_smoke_{episode['episode_id']}",
        instruction="navigate to the HM3D episode goal",
        start_x=start[0],
        start_y=start[2],
        start_yaw=start_pose.yaw,
        success_region=SuccessRegion(center_x=goal[0], center_y=goal[2], radius=radius),
        max_steps=500,
        tags=["hm3d", "pointnav_smoke"],
        geodesic_distance_m=float(episode.get("info", {}).get("geodesic_distance"))
        if episode.get("info", {}).get("geodesic_distance") is not None
        else None,
    )
    return start_pose, scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="/home/sousuke/Desktop/habitat-lab/data/datasets/pointnav/hm3d_smoke/v1/val/val.json.gz",
    )
    parser.add_argument(
        "--hm3d-root",
        default="/home/sousuke/Desktop/habitat-lab/data/versioned_data/hm3d-0.2",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--goal-radius", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", default="recordings/hm3d_spatial_reasoning_ep0.mp4")
    args = parser.parse_args()

    episode = load_episode(args.dataset, args.episode_index)
    start_pose, scenario = build_scenario(episode, args.goal_radius)
    scene_path = os.path.join(args.hm3d_root, episode["scene_id"])
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"HM3D scene not found: {scene_path}")

    print(f"HM3D episode: {scenario.scenario_id}")
    print(f"Scene: {scene_path}")
    print(f"Start (x, z, y): ({start_pose.x:.3f}, {start_pose.y:.3f}, {start_pose.z:.3f})")
    print(
        "Goal (x, z): "
        f"({scenario.success_region.center_x:.3f}, {scenario.success_region.center_y:.3f}), "
        f"radius={scenario.success_region.radius:.2f}m"
    )
    print(f"Dataset geodesic distance: {scenario.geodesic_distance_m}")

    run_simulation(
        episode_id=scenario.scenario_id,
        instruction=scenario.instruction,
        max_steps=args.steps,
        scene_path=scene_path,
        output_video_path=args.output,
        init_pose=start_pose,
        scenario=scenario,
    )


if __name__ == "__main__":
    main()
