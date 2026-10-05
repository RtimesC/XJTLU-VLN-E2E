"""Fast tests for real-scene runner guardrails."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/vln_core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/vln_sim")))

from run_spatial_reasoning_sim_loop import select_simulator, validate_scenario
from vln_core.episode_scenario import EpisodeScenario, SuccessRegion


def make_scenario(**overrides):
    values = dict(
        scenario_id="guard_test",
        instruction="test",
        start_x=0.0,
        start_y=0.0,
        start_yaw=0.0,
        success_region=SuccessRegion(center_x=1.0, center_y=0.0, radius=0.5),
    )
    values.update(overrides)
    return EpisodeScenario(**values)


def test_real_scene_requires_explicit_path():
    with pytest.raises(ValueError, match="--scene"):
        select_simulator(None, allow_mock=False)


def test_missing_real_scene_cannot_silently_fallback():
    with pytest.raises(FileNotFoundError, match="Real scene not found"):
        select_simulator("/definitely/missing/scene.glb", allow_mock=False)


def test_mock_requires_explicit_opt_in():
    sim, backend = select_simulator(None, allow_mock=True)
    try:
        assert backend == "Mock Synthetic"
    finally:
        sim.close()


def test_invalid_episode_goal_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        validate_scenario(make_scenario(success_region=SuccessRegion(math.nan, 0.0, 0.5)))


def test_non_positive_goal_radius_is_rejected():
    with pytest.raises(ValueError, match="radius"):
        validate_scenario(make_scenario(success_region=SuccessRegion(1.0, 0.0, 0.0)))
