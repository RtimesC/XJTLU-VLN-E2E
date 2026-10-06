"""Fast tests for the paper-1 protocol modules."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/vln_policy")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/vln_core")))

from vln_core.experiment_protocol import ExperimentRecord, write_jsonl
from vln_policy.gaussian_map import GaussianMap, GaussianMapConfig, backproject_rgbd
from vln_policy.multi_level_action import MultiLevelActionPredictor
from vln_policy.open_set_grouping import GeometricGroupingBackend, OpenSetSemanticGrouper


def test_backproject_rgbd_identity_pose():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    depth = np.ones((2, 2), dtype=np.float32)
    points, colors = backproject_rgbd(rgb, depth, np.diag([1.0, 1.0, 1.0]), np.eye(4), stride=1)
    assert points.shape == (4, 3)
    assert np.allclose(points[:, 2], 1.0)
    assert np.allclose(colors[0], [1.0, 0.0, 0.0])


def test_gaussian_map_merges_and_roundtrips(tmp_path):
    m = GaussianMap(GaussianMapConfig(voxel_size=0.5, sample_stride=1))
    rgb = np.full((2, 2, 3), 128, dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    k = np.diag([1.0, 1.0, 1.0])
    m.update(rgb, depth, k, np.eye(4), step=1)
    m.update(rgb, depth, k, np.eye(4), step=2)
    assert len(m) == 4
    path = tmp_path / "map.json"
    m.save(path)
    loaded = GaussianMap.load(path)
    assert len(loaded) == len(m)
    assert loaded.primitives[0].observations == 2


def test_osg_geometry_backend_is_explicitly_unknown():
    m = GaussianMap()
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)
    m.update(rgb, depth, np.eye(3), np.eye(4), step=1)
    result = OpenSetSemanticGrouper(GeometricGroupingBackend()).group(rgb, m)
    assert "unknown" in result


def test_map_predictor_has_discrete_and_continuous_outputs():
    action = MultiLevelActionPredictor().predict("turn left through the doorway", GaussianMap())
    assert action.name == "turn-left"
    assert MultiLevelActionPredictor.to_continuous(action) == (0.0, 0.4, 0.0)


def test_experiment_record_rejects_unknown_failure_tag(tmp_path):
    with pytest.raises(ValueError, match="Unknown failure tags"):
        ExperimentRecord("abc", "scene", "ep", 1, {}, {}, {}, {}, failure_tags=["bad"])
    record = ExperimentRecord("abc", "scene", "ep", 1, {}, {}, {}, {}, failure_tags=["geometry_failure"])
    path = tmp_path / "records.jsonl"
    write_jsonl([record], path)
    assert json.loads(path.read_text())["episode_id"] == "ep"
