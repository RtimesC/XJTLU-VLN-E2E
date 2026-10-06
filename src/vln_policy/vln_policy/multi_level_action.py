"""Interpretable scene/view/instance action interface for paper-1 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .gaussian_map import GaussianMap


@dataclass(frozen=True)
class MultiLevelFeatures:
    scene_score: float
    view_score: float
    instance_score: float
    target_direction: float
    free_space: float


@dataclass(frozen=True)
class DiscreteNavigationAction:
    name: str
    confidence: float
    features: MultiLevelFeatures


class MultiLevelActionPredictor:
    """Small deterministic MAP reference used before learned training."""

    def extract(self, instruction: str, gaussian_map: GaussianMap) -> MultiLevelFeatures:
        text = instruction.lower()
        left = bool(re.search(r"\b(left|turn left)\b|左", text))
        right = bool(re.search(r"\b(right|turn right)\b|右", text))
        target_direction = -1.0 if left else 1.0 if right else 0.0
        scene_score = min(1.0, len(gaussian_map) / 500.0)
        view_score = min(1.0, len(gaussian_map.query_radius((0, 0, 0), 2.0)) / 100.0)
        instance_score = 0.0 if not gaussian_map.primitives else sum(
            p.semantic_group != "unknown" for p in gaussian_map.primitives
        ) / len(gaussian_map.primitives)
        return MultiLevelFeatures(scene_score, view_score, instance_score, target_direction, 1.0)

    def predict(self, instruction: str, gaussian_map: GaussianMap) -> DiscreteNavigationAction:
        features = self.extract(instruction, gaussian_map)
        if "stop" in instruction.lower() or "到达" in instruction:
            name = "stop"
        elif features.target_direction < 0:
            name = "turn-left"
        elif features.target_direction > 0:
            name = "turn-right"
        else:
            name = "forward"
        confidence = min(1.0, 0.4 + 0.2 * features.scene_score + 0.2 * features.view_score + 0.2 * features.instance_score)
        return DiscreteNavigationAction(name, confidence, features)

    @staticmethod
    def to_continuous(action: DiscreteNavigationAction) -> tuple[float, float, float]:
        if action.name == "turn-left":
            return 0.0, 0.4, 0.0
        if action.name == "turn-right":
            return 0.0, -0.4, 0.0
        if action.name == "stop":
            return 0.0, 0.0, 1.0
        return 0.25, 0.0, 0.0
