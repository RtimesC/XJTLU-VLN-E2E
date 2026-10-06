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

    def __init__(self, sam2_model: object, clip_processor: object, clip_model: object, device: str = "cpu") -> None:
        self.sam2_model = sam2_model
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        sam2_config: str,
        sam2_checkpoint: str,
        clip_model_name_or_path: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        cache_dir: str | None = None,
    ) -> "FrozenSamClipBackend":
        """Load official SAM2 and frozen Transformers CLIP lazily on Linux."""
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from transformers import CLIPModel, CLIPProcessor

        sam_model = build_sam2(sam2_config, sam2_checkpoint, device=device, mode="eval")
        # Fewer points keeps the 10 Hz navigation loop bounded while retaining
        # instance-level masks for the protocol experiment.
        mask_generator = SAM2AutomaticMaskGenerator(
            sam_model, points_per_side=16, points_per_batch=32, pred_iou_thresh=0.75,
            stability_score_thresh=0.85, min_mask_region_area=64,
        )
        processor = CLIPProcessor.from_pretrained(clip_model_name_or_path, cache_dir=cache_dir)
        clip_model = CLIPModel.from_pretrained(clip_model_name_or_path, cache_dir=cache_dir)
        clip_model.to(device).eval()
        return cls(mask_generator, processor, clip_model, device)

    def observe(self, rgb: np.ndarray, text_queries: Iterable[str]) -> list[RegionObservation]:
        from PIL import Image
        import torch

        queries = tuple(text_queries)
        masks = self.sam2_model.generate(np.asarray(rgb, dtype=np.uint8))
        if not masks:
            return []
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        results: list[RegionObservation] = []
        for mask_item in masks:
            mask = np.asarray(mask_item["segmentation"], dtype=bool)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            crop = np.asarray(image)[y0:y1, x0:x1].copy()
            local_mask = mask[y0:y1, x0:x1]
            crop[~local_mask] = 0
            if queries:
                inputs = self.clip_processor(
                    text=list(queries), images=Image.fromarray(crop), return_tensors="pt", padding=True
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    outputs = self.clip_model(**inputs)
                    scores = outputs.logits_per_image[0].softmax(dim=-1)
                    index = int(torch.argmax(scores).item())
                    feature = outputs.image_embeds[0].detach().float().cpu().numpy()
                    confidence = float(scores[index].item())
                label = queries[index]
            else:
                label, feature, confidence = "unknown", None, float(mask_item.get("predicted_iou", 0.0))
            results.append(RegionObservation(mask, label, feature, confidence))
        return results


class OpenSetSemanticGrouper:
    def __init__(self, backend: OpenSetBackend) -> None:
        self.backend = backend

    def group(
        self,
        rgb: np.ndarray,
        gaussian_map: GaussianMap,
        text_queries: Iterable[str] = (),
        intrinsics: np.ndarray | None = None,
        camera_pose: np.ndarray | None = None,
    ) -> dict[str, int]:
        observations = self.backend.observe(rgb, text_queries)
        counts: dict[str, int] = {}
        primitives = gaussian_map.primitives
        if not primitives:
            return counts
        for region in observations:
            label = region.label or "unknown"
            counts[label] = counts.get(label, 0) + int(np.count_nonzero(region.mask))
        if intrinsics is not None and camera_pose is not None:
            self._assign_projected_primitives(gaussian_map, observations, intrinsics, camera_pose)
        return counts

    @staticmethod
    def _assign_projected_primitives(
        gaussian_map: GaussianMap,
        observations: list[RegionObservation],
        intrinsics: np.ndarray,
        camera_pose: np.ndarray,
    ) -> None:
        if not observations:
            return
        inv_pose = np.linalg.inv(np.asarray(camera_pose, dtype=np.float32).reshape(4, 4))
        k = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
        h, w = observations[0].mask.shape
        for primitive in gaussian_map.primitives:
            p = inv_pose @ np.array([*primitive.position, 1.0], dtype=np.float32)
            if p[2] <= 1e-5:
                continue
            uv = k @ p[:3]
            x, y = int(round(float(uv[0] / uv[2]))), int(round(float(uv[1] / uv[2])))
            if not (0 <= x < w and 0 <= y < h):
                continue
            candidates = [region for region in observations if region.mask[y, x]]
            if not candidates:
                continue
            region = max(candidates, key=lambda item: item.confidence)
            primitive.semantic_group = region.label
            primitive.confidence = float(region.confidence)
            if region.feature is not None:
                primitive.feature = region.feature
