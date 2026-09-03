from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass(frozen=True)
class MotDetection:
    frame: int
    track_id: int
    label: Literal["player", "ball"]
    bbox_xywh: Tuple[float, float, float, float]
    confidence: Optional[float] = None
    visibility: Optional[float] = None


@dataclass(frozen=True)
class GroundTruthRole:
    video_id: str
    side: Literal["offense", "defense"]
    position: str
    track_id: Optional[int] = None
    source_label: str = ""
    track_state: Literal["VISIBLE", "NOT_VISIBLE", "UNKNOWN_GROUND_TRUTH"] = "VISIBLE"
    allowed_predictions: Optional[List[str]] = None
    source_row: int = 0


@dataclass(frozen=True)
class ActionAnnotation:
    video_id: str
    action: str
    actor_track_id: Optional[int]
    start_frame: Optional[int]
    end_frame: Optional[int] = None
    source_row: int = 0
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    dataset_order: int
    view_raw: Optional[str] = None
    play_type: Optional[str] = None
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrackSummary:
    track_id: int
    label: str
    frames_present: List[int]
    detections: List[MotDetection]
    first_frame: int
    last_frame: int
    num_boxes: int
    coverage_ratio: float
    median_bbox_height: float
    median_bbox_width: float
    median_footpoint: Optional[Tuple[float, float]] = None
    presnap_median_footpoint: Optional[Tuple[float, float]] = None
    # Immutable position-inference anchor. Prefer the earliest reliable formation
    # frame (normally frame 0); use a later pre-snap frame only when this track was
    # absent/occluded at the primary anchor. ID-switch detection must never rewrite it.
    formation_anchor_footpoint: Optional[Tuple[float, float]] = None
    formation_anchor_frame: Optional[int] = None
    presnap_motion: Optional[float] = None
    validity_score: float = 1.0
    validity_flags: List[str] = field(default_factory=list)


@dataclass
class RoleEvidence:
    track_id: int
    action_scores: Dict[str, float] = field(default_factory=dict)
    geometry_scores: Dict[str, float] = field(default_factory=dict)
    learned_scores: Dict[str, float] = field(default_factory=dict)
    paired_scores: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)


@dataclass
class PositionAssignment:
    slot_id: str
    side: Literal["offense", "defense"]
    position: str
    track_id: Optional[int]
    visibility: Literal["visible", "occluded_or_sparse", "out_of_view", "unknown"]
    confidence: float
    slot_state: Literal["ACTIVE_VISIBLE", "ACTIVE_NOT_VISIBLE", "INACTIVE_SLOT"] = "ACTIVE_VISIBLE"
    assigned_score: float = 0.0
    best_alternative_score: float = 0.0
    score_margin: float = 0.0
    alternative_position: Optional[str] = None
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    evidence: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    track_id_display: str = ""

    def __post_init__(self):
        if not self.track_id_display:
            if self.track_id is None or self.visibility == "out_of_view" or self.slot_state == "ACTIVE_NOT_VISIBLE":
                object.__setattr__(self, "track_id_display", "not_visible")
            else:
                object.__setattr__(self, "track_id_display", str(self.track_id))


@dataclass
class ViewPrediction:
    view: Literal["sideline", "endzone", "unknown"]
    confidence: float
    evidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class OffenseDirectionPrediction:
    direction: Literal["right", "left", "up", "down", "unknown"]
    confidence: float
    evidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class ViewInferenceResult:
    video_id: str
    view: Literal["sideline", "endzone", "unknown"]
    view_confidence: float
    offense_direction: Optional[str]
    offense_direction_confidence: float
    assignments: List[PositionAssignment]
    rejected_track_ids: List[int] = field(default_factory=list)
    suspected_id_switches: List[Dict[str, float]] = field(default_factory=list)
    personnel_hypothesis: Dict[str, int] = field(default_factory=dict)
    preliminary_personnel_hypothesis: Optional[Dict[str, int]] = None
    paired_personnel_prior: Optional[Dict[str, int]] = None
    confidence_calibrated: bool = False
    solver_pass: int = 1
    confidence: float = 1.0
    status: str = "AUTO_ACCEPTED"
    warnings: List[str] = field(default_factory=list)
    metadata_source: Optional[str] = None
    pair_resolution_margin: float = 0.0
    pair_diagnostics: Dict[str, Any] = field(default_factory=dict)
