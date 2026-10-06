"""Lightweight online 3D Gaussian scene map for the paper-1 protocol.

The implementation intentionally separates geometry accumulation from any
renderer or learned model.  It can therefore run in the fast Mac tests while
using the same RGB-D and pose contract as the Habitat adapter on Linux.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass
class GaussianPrimitive:
    position: np.ndarray
    scale: float = 0.04
    color: np.ndarray = None
    feature: Optional[np.ndarray] = None
    semantic_group: str = "unknown"
    confidence: float = 1.0
    last_observed_step: int = 0
    observations: int = 1

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32).reshape(3)
        if self.color is None:
            self.color = np.zeros(3, dtype=np.float32)
        self.color = np.asarray(self.color, dtype=np.float32).reshape(3)
        if self.feature is not None:
            self.feature = np.asarray(self.feature, dtype=np.float32)


@dataclass(frozen=True)
class GaussianMapConfig:
    voxel_size: float = 0.08
    merge_distance: float = 0.12
    sample_stride: int = 8
    max_primitives: int = 100_000


def backproject_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    stride: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project sampled RGB-D pixels into world coordinates.

    ``intrinsics`` is a 3x3 matrix and ``camera_pose`` maps camera points to
    world points. Depth is expressed in metres. Invalid or non-positive depth
    samples are dropped.
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
    camera_pose = np.asarray(camera_pose, dtype=np.float32).reshape(4, 4)
    if rgb.ndim != 3 or depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError("rgb and depth must share their first two dimensions")
    stride = max(1, int(stride))
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[v, u]
    valid = np.isfinite(z) & (z > 1e-5)
    z = z[valid]
    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    xyz_camera = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)
    xyz_h = np.concatenate((xyz_camera, np.ones((len(xyz_camera), 1), dtype=np.float32)), axis=1)
    xyz_world = (camera_pose @ xyz_h.T).T[:, :3]
    colors = rgb[v.astype(np.int32), u.astype(np.int32), :3].astype(np.float32) / 255.0
    return xyz_world.astype(np.float32), colors


class GaussianMap:
    """Incremental voxel-merged Gaussian map with deterministic serialization."""

    def __init__(self, config: GaussianMapConfig = GaussianMapConfig()) -> None:
        if config.voxel_size <= 0 or config.merge_distance <= 0:
            raise ValueError("voxel_size and merge_distance must be positive")
        self.config = config
        self._primitives: dict[tuple[int, int, int], GaussianPrimitive] = {}

    @property
    def primitives(self) -> tuple[GaussianPrimitive, ...]:
        return tuple(self._primitives.values())

    def __len__(self) -> int:
        return len(self._primitives)

    def update(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_pose: np.ndarray,
        step: int,
        features: Optional[np.ndarray] = None,
        semantic_groups: Optional[Iterable[str]] = None,
    ) -> int:
        positions, colors = backproject_rgbd(
            rgb, depth, intrinsics, camera_pose, self.config.sample_stride
        )
        groups = list(semantic_groups) if semantic_groups is not None else ["unknown"] * len(positions)
        if len(groups) != len(positions):
            raise ValueError("semantic_groups must match the number of projected points")
        for i, (position, color) in enumerate(zip(positions, colors)):
            key = tuple(np.floor(position / self.config.voxel_size).astype(np.int64).tolist())
            old = self._primitives.get(key)
            if old is None:
                feature = None if features is None else np.asarray(features[i], dtype=np.float32)
                self._primitives[key] = GaussianPrimitive(
                    position=position,
                    color=color,
                    feature=feature,
                    semantic_group=groups[i],
                    last_observed_step=int(step),
                )
            else:
                n = float(old.observations)
                old.position = (old.position * n + position) / (n + 1.0)
                old.color = (old.color * n + color) / (n + 1.0)
                if features is not None:
                    f = np.asarray(features[i], dtype=np.float32)
                    old.feature = f if old.feature is None else (old.feature * n + f) / (n + 1.0)
                if groups[i] != "unknown":
                    old.semantic_group = groups[i]
                old.observations += 1
                old.last_observed_step = int(step)
        if len(self) > self.config.max_primitives:
            keys = sorted(self._primitives, key=lambda k: self._primitives[k].last_observed_step)
            for key in keys[: len(self) - self.config.max_primitives]:
                del self._primitives[key]
        return len(positions)

    def query_radius(self, center: np.ndarray, radius: float) -> list[GaussianPrimitive]:
        center = np.asarray(center, dtype=np.float32).reshape(3)
        return [p for p in self.primitives if float(np.linalg.norm(p.position - center)) <= radius]

    def coverage(self) -> float:
        if not self._primitives:
            return 0.0
        return float(np.mean([p.confidence for p in self.primitives]))

    def save(self, path: str | Path) -> None:
        payload = []
        for primitive in self.primitives:
            item = asdict(primitive)
            item["position"] = primitive.position.tolist()
            item["color"] = primitive.color.tolist()
            item["feature"] = None if primitive.feature is None else primitive.feature.tolist()
            payload.append(item)
        Path(path).write_text(json.dumps({"config": asdict(self.config), "primitives": payload}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GaussianMap":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(GaussianMapConfig(**data["config"]))
        for item in data["primitives"]:
            primitive = GaussianPrimitive(**item)
            key = tuple(np.floor(primitive.position / result.config.voxel_size).astype(np.int64).tolist())
            result._primitives[key] = primitive
        return result
