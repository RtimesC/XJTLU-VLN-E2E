#!/usr/bin/env python3
"""Run the paper-1 ESM/OSG/MAP protocol on a real HM3D scene.

This runner intentionally uses the dependency-free geometry backend by default.
SAM2+CLIP can be supplied through a future OpenSetBackend implementation without
changing the episode or record schema. It is a mechanism protocol, not an
original R2R/R4R/REVERIE numerical reproduction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for package in ("vln_core", "vln_policy", "vln_sim"):
    sys.path.insert(0, str(ROOT / "src" / package))

from vln_core.experiment_protocol import ExperimentRecord, write_jsonl
from vln_policy.gaussian_map import GaussianMap, GaussianMapConfig
from vln_policy.multi_level_action import MultiLevelActionPredictor
from vln_policy.open_set_grouping import FrozenSamClipBackend, GeometricGroupingBackend, OpenSetSemanticGrouper
from vln_policy.paper1_pipeline import Paper1Pipeline
from vln_sim.bridge_core import SimAgentPose
from vln_sim.habitat_adapter import HabitatSimAdapter


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _camera_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg / 2.0))
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _pose_matrix(pose: SimAgentPose, sensor_height: float) -> np.ndarray:
    c, s = np.cos(pose.yaw), np.sin(pose.yaw)
    return np.array(
        [[c, 0.0, s, pose.x], [0.0, 1.0, 0.0, pose.z + sensor_height], [-s, 0.0, c, pose.y], [0, 0, 0, 1]],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction", default="navigate straight through the corridor")
    parser.add_argument("--action-mode", choices=("discrete", "continuous"), default="discrete")
    parser.add_argument("--semantic-backend", choices=("geometry", "sam2clip"), default="geometry")
    parser.add_argument("--sam2-config", default="/tmp/sam2-src/sam2/configs/sam2/sam2_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", default="/home/sousuke/models/sam2/sam2_hiera_tiny.pt")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-cache", default="/home/sousuke/models/huggingface")
    parser.add_argument("--semantic-query", action="append", default=None)
    parser.add_argument("--output", default="artifacts/paper1_hm3d/record.jsonl")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    np.random.seed(args.seed)
    semantic_queries = tuple(args.semantic_query or ("door", "chair", "table", "sofa", "bed", "wall", "floor"))

    sim = HabitatSimAdapter(args.scene)
    try:
        start = sim.sample_navigable_pose(yaw=0.0)
        obs = sim.reset(start)
        width, height, hfov, sensor_height = 640, 480, 90.0, 0.45
        intrinsics = _camera_intrinsics(width, height, hfov)
        if args.semantic_backend == "sam2clip":
            backend = FrozenSamClipBackend.from_pretrained(
                args.sam2_config, args.sam2_checkpoint, args.clip_model,
                device="cuda" if __import__("torch").cuda.is_available() else "cpu",
                cache_dir=args.model_cache,
            )
        else:
            backend = GeometricGroupingBackend()
        pipeline = Paper1Pipeline(GaussianMap(GaussianMapConfig()), OpenSetSemanticGrouper(backend), MultiLevelActionPredictor())
        raw_actions, safe_actions, executed_actions, trajectory = [], [], [], []
        collisions = 0
        for step in range(args.steps):
            pose_matrix = _pose_matrix(sim.pose, sensor_height)
            rgb, depth = obs.rgb, obs.depth
            if depth is None:
                raise RuntimeError("Habitat adapter did not provide depth_sensor observations")
            result = pipeline.step(rgb, depth, intrinsics, pose_matrix, args.instruction, step, semantic_queries)
            v, w, stop = pipeline.predictor.to_continuous(result.action)
            raw = {"action": result.action.name, "v": v, "omega": w, "stop": stop}
            raw_actions.append(raw)
            if args.action_mode == "discrete":
                v, w = (0.0, 0.4) if result.action.name == "turn-left" else (0.0, -0.4) if result.action.name == "turn-right" else (0.25, 0.0)
                if result.action.name == "stop":
                    v, w = 0.0, 0.0
            safe_actions.append({"v": v, "omega": w, "stop": stop})
            obs = sim.step(v, w, 0.1)
            executed_actions.append({"v": v, "omega": w, "stop": stop, "collision": obs.is_collision})
            collisions += int(obs.is_collision)
            trajectory.append([sim.pose.x, sim.pose.y, sim.pose.yaw])
            if stop >= 1.0:
                break
        record = ExperimentRecord(
            commit=_commit(),
            scene_id=os.path.basename(args.scene),
            episode_id=f"paper1_hm3d_{args.seed}",
            seed=args.seed,
            camera_config={"width": width, "height": height, "hfov": hfov, "sensor_height": sensor_height},
            agent_config={"radius": sim.AGENT_RADIUS_M, "height": sim.AGENT_HEIGHT_M, "dt": 0.1},
            map_config={"voxel_size": pipeline.gaussian_map.config.voxel_size, "semantic_backend": args.semantic_backend},
            policy_config={"pipeline": "ESM_OSG_MAP", "action_mode": args.action_mode, "instruction": args.instruction},
            raw_actions=raw_actions,
            safe_actions=safe_actions,
            executed_actions=executed_actions,
            trajectory=trajectory,
            metrics={
                "map_size": float(len(pipeline.gaussian_map)),
                "collisions": float(collisions),
                "semantic_group_count": float(sum(p.semantic_group != "unknown" for p in pipeline.gaussian_map.primitives)),
            },
            failure_tags=["semantic_grouping_failure"] if args.semantic_backend == "geometry" else [],
        )
        write_jsonl([record], args.output)
        print(json.dumps({"output": str(Path(args.output).resolve()), "map_size": len(pipeline.gaussian_map), "collisions": collisions}))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
