"""Spatial Intelligence and Reasoning Policy for limited-FOV low-chassis VLN.

This policy addresses key physical challenges of the XJTLU low-chassis car:
1. Limited Field-of-View (90 deg HFOV) and low camera mounting height (0.45m).
2. Frequent near-field visual occlusions (furniture legs, partitions, corners).
3. Need for spatial working memory and direction inference from linguistic cues,
   rather than reactive random spinning when targets are temporarily out of sight.

Architecture:
- SpatialSectorMemory: Egocentric 8-sector topological memory tracking traversability,
  exploration recency, and landmark/target relevance.
- LimitedFovReasoner: Extracts occlusion ratio, free-space corridor hints, and maps
  natural language spatial cues (left, right, straight, etc.) to sector belief.
- SpatialReasoningPolicy: State machine (ORIENT -> INFERRED_EXPLORE -> APPROACH -> VERIFY -> STOP)
  emitting PolicyActionData with linear/angular velocity and stop confidence.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import re
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from vln_core.protocol import PolicyActionData, PolicyOutcome


class ReasoningState(str, Enum):
    ORIENT = "ORIENT"
    INFERRED_EXPLORE = "INFERRED_EXPLORE"
    APPROACH = "APPROACH"
    VERIFY = "VERIFY"
    STOP = "STOP"
    FAILED = "FAILED"


@dataclass
class SectorBelief:
    """Belief state for an egocentric angular sector."""
    sector_id: int               # 0: Front, 1: Front-Left, 2: Left, 3: Rear-Left, etc.
    center_angle_deg: float      # Relative to current vehicle heading (0 = straight ahead)
    exploration_count: int = 0
    traversability: float = 0.5  # [0.0 = blocked, 1.0 = clear corridor]
    target_probability: float = 0.125  # Uniform prior across 8 sectors
    last_seen_step: int = -1


@dataclass
class VisualAnalysisResult:
    """Visual feature summary extracted under limited-FOV conditions."""
    target_detected: bool
    target_center_x_norm: float  # [-1.0 (left), 1.0 (right)]
    target_area_ratio: float
    target_confidence: float
    occlusion_ratio: float       # [0.0 = clear view, 1.0 = heavily blocked by near obstacle]
    free_space_bias_norm: float  # [-1.0 to 1.0] indicating open navigable passage offset


@dataclass
class SpatialReasoningConfig:
    """Hyperparameters for SpatialReasoningPolicy."""
    num_sectors: int = 8                    # 8 sectors (45 deg each)
    max_search_steps: int = 40
    max_episode_steps: int = 250
    arrival_area_ratio_threshold: float = 0.18
    arrival_center_tolerance_norm: float = 0.15
    arrival_confirm_frames: int = 3
    grounding_confidence_threshold: float = 0.50

    # Low-height occlusion parameters
    occlusion_area_threshold: float = 0.40  # If lower third of image is untextured or solid obstacle

    # Motion parameters
    search_angular_velocity: float = 0.40    # rad/s
    approach_linear_velocity: float = 0.35   # m/s
    verify_linear_velocity: float = 0.15     # m/s
    explore_linear_velocity: float = 0.25    # m/s forward when exploring inferred corridor
    steer_gain: float = 0.50

    model_version: str = "spatial_reasoning_v1"


class SpatialSectorMemory:
    """Maintains egocentric 8-sector spatial working memory around the vehicle."""

    def __init__(self, num_sectors: int = 8):
        self.num_sectors = num_sectors
        self.sector_span_deg = 360.0 / num_sectors
        self.sectors: List[SectorBelief] = []
        self._init_sectors()

    def _init_sectors(self):
        self.sectors = []
        for i in range(self.num_sectors):
            # 0: 0 deg (Front)
            # 1: 45 deg (Front-Left)
            # 2: 90 deg (Left)
            # 3: 135 deg (Rear-Left)
            # 4: 180 deg (Rear)
            # 5: -135 deg (Rear-Right)
            # 6: -90 deg (Right)
            # 7: -45 deg (Front-Right)
            angle = (i * self.sector_span_deg)
            if angle > 180.0:
                angle -= 360.0
            self.sectors.append(
                SectorBelief(
                    sector_id=i,
                    center_angle_deg=angle,
                    target_probability=1.0 / self.num_sectors,
                )
            )

    def reset(self):
        self._init_sectors()

    def apply_instruction_prior(self, instruction: str):
        """Inject linguistic spatial priors into sector probabilities."""
        text = instruction.lower()
        left_cues = ["left", "turn left", "port", "左"]
        right_cues = ["right", "turn right", "starboard", "右"]
        straight_cues = ["straight", "forward", "ahead", "through", "直走", "向前"]

        boost_left = any(cue in text for cue in left_cues)
        boost_right = any(cue in text for cue in right_cues)
        boost_straight = any(cue in text for cue in straight_cues)

        weights = np.ones(self.num_sectors, dtype=float)
        if boost_left:
            weights[1] += 1.5  # Front-Left
            weights[2] += 2.0  # Left
        if boost_right:
            weights[6] += 2.0  # Right
            weights[7] += 1.5  # Front-Right
        if boost_straight:
            weights[0] += 2.0  # Front

        # Normalize probabilities
        norm_weights = weights / np.sum(weights)
        for i, s in enumerate(self.sectors):
            s.target_probability = float(norm_weights[i])

    def update_front_observation(
        self,
        step: int,
        traversability: float,
        target_seen: bool,
        target_conf: float,
        target_x_norm: float,
    ):
        """Update front sector (and adjacent sectors based on target offset)."""
        front = self.sectors[0]
        front.exploration_count += 1
        front.last_seen_step = step
        front.traversability = 0.7 * front.traversability + 0.3 * traversability

        if target_seen:
            if target_x_norm < -0.3:
                target_sector = 1  # Front-Left
            elif target_x_norm > 0.3:
                target_sector = 7  # Front-Right
            else:
                target_sector = 0  # Front

            for i, s in enumerate(self.sectors):
                if i == target_sector:
                    s.target_probability = max(s.target_probability, target_conf)
                else:
                    s.target_probability *= 0.5
            self._normalize_probabilities()
        else:
            front.target_probability *= 0.75
            self._normalize_probabilities()

    def update_rotation(self, delta_yaw_rad: float):
        """Shift sector memory beliefs when the vehicle rotates."""
        delta_deg = math.degrees(delta_yaw_rad)
        sector_shift = int(round(delta_deg / self.sector_span_deg))
        if sector_shift != 0:
            shift_idx = sector_shift % self.num_sectors
            probs = [s.target_probability for s in self.sectors]
            travs = [s.traversability for s in self.sectors]
            for i in range(self.num_sectors):
                src_idx = (i + shift_idx) % self.num_sectors
                self.sectors[i].target_probability = probs[src_idx]
                self.sectors[i].traversability = travs[src_idx]

    def _normalize_probabilities(self):
        total = sum(s.target_probability for s in self.sectors)
        if total > 1e-6:
            for s in self.sectors:
                s.target_probability /= total

    def select_inferred_heading(self) -> Tuple[int, float, float]:
        """Selects the best sector (sector_id, angle_deg, expected_utility)."""
        best_id = 0
        best_score = -1.0
        best_angle = 0.0

        for s in self.sectors:
            freshness = 1.0 / (1.0 + 0.2 * s.exploration_count)
            score = (
                0.60 * s.target_probability
                + 0.25 * s.traversability
                + 0.15 * freshness
            )
            if score > best_score:
                best_score = score
                best_id = s.sector_id
                best_angle = s.center_angle_deg

        return best_id, best_angle, best_score


class LimitedFovVisualGrounder:
    """Visual grounder tailored for low-chassis 0.45m camera observations."""

    def __init__(self, config: SpatialReasoningConfig):
        self.config = config

    def analyze(self, rgb: np.ndarray) -> VisualAnalysisResult:
        if rgb is None or rgb.size == 0:
            return VisualAnalysisResult(
                target_detected=False,
                target_center_x_norm=0.0,
                target_area_ratio=0.0,
                target_confidence=0.0,
                occlusion_ratio=0.0,
                free_space_bias_norm=0.0,
            )

        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        lower_region = gray[int(h * 0.65):, :]
        lower_variance = float(np.var(lower_region))
        occlusion_ratio = float(np.clip(1.0 - (lower_variance / 400.0), 0.0, 1.0))

        col_means = np.mean(gray[:int(h * 0.7), :], axis=0)
        open_center_idx = int(np.argmax(col_means))
        free_space_bias = (open_center_idx - (w / 2.0)) / (w / 2.0)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_target = None
        best_conf = 0.0

        img_area = float(w * h)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area_ratio = (bw * bh) / img_area
            aspect_ratio = bh / float(max(bw, 1))

            if (
                0.015 <= area_ratio <= 0.70
                and aspect_ratio >= 1.15
                and (bh / float(h)) >= 0.20
            ):
                center_x_norm = ((x + bw / 2.0) - (w / 2.0)) / (w / 2.0)
                conf = 0.5 + 0.3 * min(area_ratio / 0.15, 1.0) - 0.2 * abs(center_x_norm)
                if conf > best_conf and conf >= self.config.grounding_confidence_threshold:
                    best_conf = conf
                    best_target = (center_x_norm, area_ratio, conf)

        if best_target:
            return VisualAnalysisResult(
                target_detected=True,
                target_center_x_norm=best_target[0],
                target_area_ratio=best_target[1],
                target_confidence=best_target[2],
                occlusion_ratio=occlusion_ratio,
                free_space_bias_norm=free_space_bias,
            )
        else:
            return VisualAnalysisResult(
                target_detected=False,
                target_center_x_norm=0.0,
                target_area_ratio=0.0,
                target_confidence=0.0,
                occlusion_ratio=occlusion_ratio,
                free_space_bias_norm=free_space_bias,
            )


class SpatialReasoningPolicy:
    """Main policy integrating topological sector memory and spatial reasoning."""

    def __init__(self, config: Optional[SpatialReasoningConfig] = None):
        self.config = config or SpatialReasoningConfig()
        self.memory = SpatialSectorMemory(num_sectors=self.config.num_sectors)
        self.grounder = LimitedFovVisualGrounder(self.config)

        self.state = ReasoningState.ORIENT
        self.step_count = 0
        self.arrival_counter = 0
        self.current_episode_id = ""
        self.instruction_set = False

    def reset(self, episode_id: str = ""):
        self.memory.reset()
        self.state = ReasoningState.ORIENT
        self.step_count = 0
        self.arrival_counter = 0
        self.current_episode_id = episode_id
        self.instruction_set = False

    def _make_action(
        self,
        v: float,
        w: float,
        p_stop: float,
        start_mono: float,
        outcome: PolicyOutcome = PolicyOutcome.RUNNING,
        outcome_detail: str = "",
    ) -> PolicyActionData:
        elapsed_ms = (time.monotonic() - start_mono) * 1000.0
        return PolicyActionData(
            header_stamp_sec=time.time(),
            frame_id="base_link",
            episode_id=self.current_episode_id,
            sequence_id=self.step_count,
            linear_velocity=float(v),
            angular_velocity=float(w),
            stop_probability=float(p_stop),
            inference_latency_ms=float(elapsed_ms),
            valid=True,
            model_version=self.config.model_version,
            outcome=outcome,
            outcome_detail=outcome_detail,
        )

    def step(
        self,
        rgb: np.ndarray,
        instruction: str = "",
        episode_id: str = "",
    ) -> Tuple[PolicyActionData, dict]:
        """Execute one reasoning step from front RGB observation."""
        start_mono = time.monotonic()
        self.step_count += 1
        if episode_id and episode_id != self.current_episode_id:
            self.reset(episode_id)

        # 1. Parse linguistic cues once per episode
        if instruction and not self.instruction_set:
            self.memory.apply_instruction_prior(instruction)
            self.instruction_set = True

        # 2. Extract limited-FOV visual cues
        vis = self.grounder.analyze(rgb)

        # 3. Update spatial sector memory
        traversability = max(0.1, 1.0 - vis.occlusion_ratio)
        self.memory.update_front_observation(
            step=self.step_count,
            traversability=traversability,
            target_seen=vis.target_detected,
            target_conf=vis.target_confidence,
            target_x_norm=vis.target_center_x_norm,
        )

        # Global episode step timeout safeguard
        if self.step_count >= self.config.max_episode_steps:
            self.state = ReasoningState.FAILED
            action = self._make_action(
                v=0.0,
                w=0.0,
                p_stop=1.0,
                start_mono=start_mono,
                outcome=PolicyOutcome.FAILED,
                outcome_detail="TIMEOUT",
            )
            return action, self._make_info(vis, "TIMEOUT")

        # 4. State transition and control generation
        linear_vel = 0.0
        angular_vel = 0.0
        stop_prob = 0.0
        outcome = PolicyOutcome.RUNNING
        outcome_detail = ""

        if self.state in (ReasoningState.ORIENT, ReasoningState.INFERRED_EXPLORE):
            if vis.target_detected:
                self.state = ReasoningState.APPROACH
            elif self.state == ReasoningState.ORIENT:
                angular_vel = self.config.search_angular_velocity
                self.memory.update_rotation(angular_vel * 0.1)
                if self.step_count >= 8:
                    self.state = ReasoningState.INFERRED_EXPLORE
            elif self.state == ReasoningState.INFERRED_EXPLORE:
                best_id, best_angle, _ = self.memory.select_inferred_heading()
                if abs(best_angle) > 25.0:
                    turn_direction = 1.0 if best_angle > 0.0 else -1.0
                    angular_vel = turn_direction * self.config.search_angular_velocity
                    linear_vel = 0.05
                    self.memory.update_rotation(angular_vel * 0.1)
                else:
                    linear_vel = self.config.explore_linear_velocity
                    angular_vel = -self.config.steer_gain * vis.free_space_bias_norm

        if self.state == ReasoningState.APPROACH:
            if not vis.target_detected:
                self.state = ReasoningState.INFERRED_EXPLORE
            else:
                if (
                    vis.target_area_ratio >= self.config.arrival_area_ratio_threshold
                    and abs(vis.target_center_x_norm) <= self.config.arrival_center_tolerance_norm
                ):
                    self.state = ReasoningState.VERIFY
                    self.arrival_counter = 1
                else:
                    linear_vel = self.config.approach_linear_velocity
                    angular_vel = -self.config.steer_gain * vis.target_center_x_norm


        elif self.state == ReasoningState.VERIFY:
            if (
                vis.target_detected
                and vis.target_area_ratio >= self.config.arrival_area_ratio_threshold
            ):
                self.arrival_counter += 1
                linear_vel = self.config.verify_linear_velocity
                angular_vel = -0.2 * vis.target_center_x_norm
                if self.arrival_counter >= self.config.arrival_confirm_frames:
                    self.state = ReasoningState.STOP
                    linear_vel = 0.0
                    angular_vel = 0.0
                    stop_prob = 1.0
                    outcome = PolicyOutcome.STOP_REQUESTED
                    outcome_detail = "SUCCESS_ARRIVAL"
            else:
                self.arrival_counter = 0
                self.state = ReasoningState.APPROACH

        elif self.state == ReasoningState.STOP:
            linear_vel = 0.0
            angular_vel = 0.0
            stop_prob = 1.0
            outcome = PolicyOutcome.STOP_REQUESTED
            outcome_detail = "SUCCESS_ARRIVAL"


        action = self._make_action(
            v=linear_vel,
            w=angular_vel,
            p_stop=stop_prob,
            start_mono=start_mono,
            outcome=outcome,
            outcome_detail=outcome_detail,
        )
        return action, self._make_info(vis, self.state.value)

    def _make_info(self, vis: VisualAnalysisResult, state_str: str) -> dict:
        return {
            "state": state_str,
            "step": self.step_count,
            "target_detected": vis.target_detected,
            "target_confidence": vis.target_confidence,
            "occlusion_ratio": vis.occlusion_ratio,
            "model_version": self.config.model_version,
        }
