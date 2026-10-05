"""Tests for explicit navmesh pose guardrails."""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/vln_core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/vln_sim")))

from run_spatial_reasoning_sim_loop import validate_habitat_start
from vln_sim.bridge_core import SimAgentPose
from vln_sim.habitat_adapter import HabitatSimAdapter


def test_invalid_habitat_pose_fails_without_sampling(monkeypatch):
    sim = object.__new__(HabitatSimAdapter)
    monkeypatch.setattr(sim, "is_navigable", lambda pose: False)
    with pytest.raises(ValueError, match="not navigable"):
        validate_habitat_start(sim, SimAgentPose(), sample_valid_pose=False)


def test_invalid_habitat_pose_can_be_sampled(monkeypatch):
    sim = object.__new__(HabitatSimAdapter)
    monkeypatch.setattr(sim, "is_navigable", lambda pose: False)
    expected = SimAgentPose(x=1.0, y=2.0, z=0.1)
    monkeypatch.setattr(sim, "sample_navigable_pose", lambda yaw: expected)
    assert validate_habitat_start(sim, SimAgentPose(yaw=0.4), sample_valid_pose=True) is expected
