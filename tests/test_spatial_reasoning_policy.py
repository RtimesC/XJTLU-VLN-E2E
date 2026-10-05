"""Unit tests for SpatialReasoningPolicy, SectorMemory, and LimitedFovVisualGrounder."""

import cv2
import numpy as np
import pytest

from vln_policy.spatial_reasoning_policy import (
    LimitedFovVisualGrounder,
    ReasoningState,
    SpatialReasoningConfig,
    SpatialReasoningPolicy,
    SpatialSectorMemory,
)


def create_synthetic_doorway_image(
    door_x: int = 320,
    door_y: int = 140,
    door_w: int = 120,
    door_h: int = 240,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Creates a synthetic room image containing a doorway frame."""
    # Textured room wall
    img = np.ones((height, width, 3), dtype=np.uint8) * 180
    # Add wall gradient/texture
    for y in range(height):
        img[y, :, :] = np.clip(img[y, :, :] - int(y * 0.1), 0, 255)

    # Door frame (dark rectangle representing open doorway)
    cv2.rectangle(
        img,
        (door_x - door_w // 2, door_y),
        (door_x + door_w // 2, door_y + door_h),
        (30, 30, 30),
        -1,
    )
    # Door borders
    cv2.rectangle(
        img,
        (door_x - door_w // 2, door_y),
        (door_x + door_w // 2, door_y + door_h),
        (240, 240, 240),
        3,
    )
    return img


def test_spatial_sector_memory_initialization_and_priors():
    memory = SpatialSectorMemory(num_sectors=8)
    assert len(memory.sectors) == 8
    # Initial uniform probabilities
    for s in memory.sectors:
        assert pytest.approx(s.target_probability, rel=1e-3) == 1.0 / 8.0

    # Inject 'turn left' prompt
    memory.apply_instruction_prior("go into the room and turn left")
    # Front-left (id=1) and Left (id=2) should have higher probability than Front (id=0) or Right (id=6)
    assert memory.sectors[1].target_probability > memory.sectors[0].target_probability
    assert memory.sectors[2].target_probability > memory.sectors[6].target_probability

    # Reset restores uniform
    memory.reset()
    for s in memory.sectors:
        assert pytest.approx(s.target_probability, rel=1e-3) == 1.0 / 8.0


def test_spatial_sector_memory_rotation():
    memory = SpatialSectorMemory(num_sectors=8)
    # Set front sector to have high probability
    memory.sectors[0].target_probability = 0.8
    memory._normalize_probabilities()

    # Robot turns 45 degrees left (approx 0.785 rad, +yaw)
    memory.update_rotation(np.pi / 4.0)
    # After turning left, the target in front is now in sector 7 (Front-Right, -45 deg)
    assert memory.sectors[7].target_probability > 0.4



def test_limited_fov_visual_grounder_detection():
    config = SpatialReasoningConfig()
    grounder = LimitedFovVisualGrounder(config)

    # Blank image (no target)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res_blank = grounder.analyze(blank)
    assert not res_blank.target_detected
    assert res_blank.target_confidence == 0.0

    # Image with clear doorway in center
    door_img = create_synthetic_doorway_image(door_x=320, door_y=120, door_w=120, door_h=260)
    res_door = grounder.analyze(door_img)
    assert res_door.target_detected
    assert abs(res_door.target_center_x_norm) < 0.20
    assert res_door.target_area_ratio > 0.05
    assert res_door.target_confidence >= config.grounding_confidence_threshold


def test_spatial_reasoning_policy_lifecycle():
    config = SpatialReasoningConfig(
        max_search_steps=10,
        arrival_confirm_frames=2,
    )
    policy = SpatialReasoningPolicy(config=config)
    policy.reset("test_ep_001")

    assert policy.state == ReasoningState.ORIENT

    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    # In blank environment with left instruction, runs ORIENT then INFERRED_EXPLORE
    for _ in range(8):
        action, info = policy.step(blank, instruction="turn left to exit")
        assert action.linear_velocity >= 0.0
        assert action.stop_probability == 0.0

    assert policy.state == ReasoningState.INFERRED_EXPLORE

    # Feed an image with doorway in center
    target_img = create_synthetic_doorway_image(door_x=320, door_y=140, door_w=100, door_h=220)
    action, info = policy.step(target_img)
    assert policy.state == ReasoningState.APPROACH
    assert action.linear_velocity == config.approach_linear_velocity

    # Feed large doorway representing arrival
    arrival_img = create_synthetic_doorway_image(door_x=320, door_y=40, door_w=280, door_h=400)
    action, info = policy.step(arrival_img)
    assert policy.state == ReasoningState.VERIFY

    action, info = policy.step(arrival_img)
    assert policy.state == ReasoningState.STOP
    assert action.stop_probability == 1.0
    assert action.linear_velocity == 0.0
    assert action.angular_velocity == 0.0


def test_spatial_reasoning_policy_timeout():
    config = SpatialReasoningConfig(max_episode_steps=5)
    policy = SpatialReasoningPolicy(config=config)
    policy.reset("timeout_ep")

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        action, info = policy.step(blank)

    assert policy.state == ReasoningState.FAILED
    assert action.stop_probability == 1.0
