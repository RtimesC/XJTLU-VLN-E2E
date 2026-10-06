"""Open-set semantic grouping interfaces for the Gaussian map.

SAM2 and CLIP are optional runtime backends. The protocol remains usable with
the deterministic ``RegionObservation`` provider for geometry-only tests and
for environments where the heavyweight models are not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from .gaussian_map import GaussianMap


@dataclass(frozen=True)
class RegionObservation:
    mask: np.ndarray
    label: str = "unknown"
    feature: np.ndarray | None = None
    confidence: float = 1.0


class OpenSetBackend(Protocol):
    def observe(self, rgb: np.ndarray, text_queries: Iterable[str]) -> list[RegionObservation]:
        ...


class GeometricGroupingBackend:
    """Dependency-free backend used only for deterministic smoke tests."""

    def observe(self, rgb: np.ndarray, text_queries: Iterable[str]) -> list[RegionObservation]:
        del text_queries
        h, w = rgb.shape[:2]
        mask = np.ones((h, w), dtype=bool)
        return [RegionObservation(mask=mask, label="unknown", confidence=0.0)]


class FrozenSamClipBackend:
    """Optional adapter boundary for frozen SAM2 + CLIP inference.

    The actual model loading is deliberately explicit so a missing checkpoint
    cannot silently turn a paper experiment into a geometry-only result.
    """

    def __init__(self, sam2_model: object, clip_model: object) -> None:
        self.sam2_model = sam2_model
        self.clip_model = clip_model

    def observe(self, rgb: np.ndarray, text_queries: Iterable[str]) -> list[RegionObservation]:
        raise NotImplementedError(
            "Bind the installed SAM2 and CLIP inference APIs here; use "
            "GeometricGroupingBackend only for the geometry baseline."
        )


class OpenSetSemanticGrouper:
    def __init__(self, backend: OpenSetBackend) -> None:
        self.backend = backend

    def group(
        self,
        rgb: np.ndarray,
        gaussian_map: GaussianMap,
        text_queries: Iterable[str] = (),
    ) -> dict[str, int]:
        observations = self.backend.observe(rgb, text_queries)
        counts: dict[str, int] = {}
        primitives = gaussian_map.primitives
        if not primitives:
            return counts
        # The first implementation records stable group statistics. A future
        # camera-projection index can replace this without changing the API.
        for region in observations:
            label = region.label or "unknown"
            counts[label] = counts.get(label, 0) + int(np.count_nonzero(region.mask))
        return counts
