"""Reproducible HM3D experiment records for paper-1 protocol runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


FAILURE_TAGS = frozenset(
    {
        "geometry_failure",
        "semantic_grouping_failure",
        "viewpoint_consistency_failure",
        "language_grounding_failure",
        "map_memory_failure",
        "continuous_transfer_failure",
        "collision_or_recovery_failure",
        "latency_failure",
    }
)


@dataclass
class ExperimentRecord:
    commit: str
    scene_id: str
    episode_id: str
    seed: int
    camera_config: dict[str, Any]
    agent_config: dict[str, Any]
    map_config: dict[str, Any]
    policy_config: dict[str, Any]
    raw_actions: list[dict[str, Any]] = field(default_factory=list)
    safe_actions: list[dict[str, Any]] = field(default_factory=list)
    executed_actions: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[list[float]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    failure_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        unknown = set(self.failure_tags) - FAILURE_TAGS
        if unknown:
            raise ValueError(f"Unknown failure tags: {sorted(unknown)}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def write_jsonl(records: Iterable[ExperimentRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(record.to_json() for record in records) + "\n", encoding="utf-8")
