"""Composed ESM -> OSG -> MAP pipeline used by the HM3D protocol runner."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .gaussian_map import GaussianMap
from .multi_level_action import DiscreteNavigationAction, MultiLevelActionPredictor
from .open_set_grouping import OpenSetSemanticGrouper


@dataclass(frozen=True)
class Paper1Step:
    action: DiscreteNavigationAction
    map_size: int
    projected_points: int
    semantic_groups: dict[str, int]


class Paper1Pipeline:
    """Dependency-light pipeline with explicit geometry/semantic switches."""

    def __init__(self, gaussian_map: GaussianMap, grouper: OpenSetSemanticGrouper, predictor: MultiLevelActionPredictor):
        self.gaussian_map = gaussian_map
        self.grouper = grouper
        self.predictor = predictor

    def step(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_pose: np.ndarray,
        instruction: str,
        step: int,
        semantic_queries: tuple[str, ...] = (),
    ) -> Paper1Step:
        projected = self.gaussian_map.update(rgb, depth, intrinsics, camera_pose, step)
        groups = self.grouper.group(rgb, self.gaussian_map, semantic_queries)
        action = self.predictor.predict(instruction, self.gaussian_map)
        return Paper1Step(action, len(self.gaussian_map), projected, groups)
