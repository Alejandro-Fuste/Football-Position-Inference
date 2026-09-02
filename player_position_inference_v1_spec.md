# Player Position Inference V1 — Implementation Specification

**Target implementation environment:** Antigravity IDE  
**Language:** Python 3.11+  
**Project:** FilmBreakdownAI / TapeVision dataset tooling  
**Status:** Resolved V1 specification  
**Primary goal:** Build a new codebase that automatically infers American-football player positions for paired sideline/endzone play clips from MOT tracks, Key Actions annotations, limited PlayerTrack ground truth, and football constraints.

---

## 1. Purpose

The dataset contains approximately 4,000 video clips, but only a small subset (approximately 90 clips) currently has player-position annotations. The purpose of this codebase is to infer the missing player-position assignments automatically and produce output compatible with the existing PlayerTrack annotation workflow.

The system must not be designed as a pure visual 22-class classifier. V1 must use a **hybrid structured-inference architecture** combining:

1. MOT pre-snap geometry;
2. football action semantics;
3. football formation/personnel constraints;
4. view-specific learned role probabilities trained/calibrated from the labeled PlayerTrack examples;
5. paired sideline/endzone evidence fusion; and
6. global constrained assignment with confidence and human-review fallbacks.

The intended conceptual objective is:

> **Constrained probabilistic football formation reconstruction from noisy MOT trajectories, sparse action semantics, paired camera views, and limited supervised player-position annotations.**

The implementation must be modular so that field registration, stronger learned models, appearance/ReID, and automatic tracking repair can be added later without replacing the V1 pipeline.

---

## 2. Authoritative V1 Requirements

These requirements are resolved and must be treated as authoritative unless the repository's actual file schemas require adapter-level changes.

### 2.1 Dataset/view facts

- Each underlying football play normally has two video clips:
  - **sideline view**;
  - **endzone view**.
- The two clips are usually adjacent in dataset/video order.
- The sideline clip is usually first and the endzone clip usually follows it.
- `DatasetSummary.csv` contains a `view` column, but that column is not populated for every video.
- `DatasetSummary.csv` does **not** contain reliable personnel, formation, play-number, offense/defense-team, or explicit paired-video metadata that can solve this task.
- The system therefore needs:
  - view inference when `view` is blank;
  - pair inference when explicit pairing metadata is unavailable.

### 2.2 Cross-view identity facts

- Sideline and endzone views use **different MOT track IDs** for the same real-world player.
- There is no existing direct sideline-track-to-endzone-track mapping.
- Example conceptually:
  - sideline Center may be track `7`;
  - endzone Center may be track `15`.
- Sideline/endzone clips are not generally temporally synchronized.
- Key Actions are annotated independently for the two views.
- V1 does **not** need direct cross-view person ReID or frame synchronization.
- V1 should fuse evidence at the **role/personnel/canonical-slot level**, not by forcing track-to-track correspondence.

### 2.3 Position ground truth

- Existing PlayerTrack position annotations contain both sideline and endzone examples of the same underlying plays.
- The labeled examples must be used as:
  - ground truth for evaluation;
  - training/calibration data for view-specific role scorers.
- A separate output mapping is required for each video.

### 2.4 Endzone visibility

- Sideline generally includes all 22 football players.
- Endzone may omit receivers and defensive backs because of camera crop/zoom.
- Endzone may contain occlusion, including:
  - QB occluding Center;
  - offensive lineman occluding defensive lineman.
- Missing endzone players must still be represented in the output using canonical position slots with `track_id=not_visible` or an equivalent typed representation.
- V1 must distinguish:
  - `visible`;
  - `occluded_or_missing_boxes`;
  - `out_of_view` / `not_visible`.

### 2.5 MOT quality

MOT ZIP files contain two object labels:

- `player`;
- `ball`.

Tracking may contain:

- missing player boxes;
- extra detections;
- referees mislabeled/represented as player tracks;
- ID switches;
- ball tracks when the ball is tracked.

V1 policy:

- tolerate missing boxes;
- identify/reject obvious false-positive or referee-like tracks from football-player assignment;
- detect and report suspected ID switches;
- **do not rewrite or automatically repair MOT track IDs in V1**.
- pre-snap ID switches are uncommon, so stable pre-snap geometry should be the primary basis for role identity.

### 2.6 Required position taxonomy

V1 general position labels are:

#### Offense
- `QB`
- `RB`
- `FB`
- `WR`
- `TE`
- `LT`
- `LG`
- `C`
- `RG`
- `RT`

#### Defense
- `DE`
- `DT`
- `LB`
- `CB`
- `FS`
- `SS`

Do not introduce finer labels such as X/Z/slot, TE-Y/TE-H, EDGE/NT, MLB/WLB/SLB in V1 output. Internal grouping may use broader temporary groups such as `OL`, `DL`, `DB`, but the final resolved label must use the taxonomy above.

### 2.7 Duplicate roles

Multiple players can share the same general position label. Therefore general position labels cannot serve as unique keys.

Every expected roster role must use a canonical unique `slot_id`, e.g.:

```text
offense.QB_1
offense.RB_1
offense.WR_1
offense.WR_2
offense.WR_3
offense.TE_1

defense.DE_1
defense.DE_2
defense.DT_1
defense.LB_1
defense.LB_2
defense.CB_1
defense.CB_2
defense.FS_1
defense.SS_1
```

Slot numbering is a **stable identity within an inferred formation**, not a semantic claim such as X/Z. See Section 15 for deterministic ordering.

### 2.8 Paired-view disagreement

When the two views disagree:

- automatically resolve only when the stronger hypothesis exceeds the weaker hypothesis by a configurable confidence margin;
- otherwise mark the pair `PAIR_REVIEW_REQUIRED`;
- preserve both view-specific evidence and the reason for disagreement in the review report.

---

## 3. Input Artifacts

Antigravity must inspect the actual repository files before implementing adapters. Do not assume column indices based only on this document.

The initial golden fixture includes:

- `PlayerTrack_ID_Sheet(2).csv`
- `KeyActions_Sheet(2).csv`
- `JetSweep_1_cvat_mot.zip`

The project also uses a dataset-level `DatasetSummary.csv`.

### 3.1 PlayerTrack annotation input

Purpose:

- authoritative labeled player-position mapping for the limited annotated subset;
- target output shape/reference;
- supervised training/evaluation labels.

The loader must discover/validate the actual column schema. It should support the project's current naming conventions rather than silently renaming data.

At minimum, normalize each annotation into:

```python
@dataclass(frozen=True)
class GroundTruthRole:
    video_id: str
    side: Literal["offense", "defense"]
    position: str
    track_id: int | None
    source_row: int
```

If the source data already encodes duplicate role slots, preserve them. If it only contains repeated `WR`, `LB`, etc., generate canonical slot IDs deterministically after geometry/ordering information is available.

### 3.2 Key Actions input

Purpose:

- determine semantic actor anchors;
- identify snap timing / pre-snap window;
- contribute probabilistic role evidence;
- provide independent evidence for each view.

Normalize rows into a data structure such as:

```python
@dataclass(frozen=True)
class ActionAnnotation:
    video_id: str
    action: str
    actor_track_id: int | None
    start_frame: int | None
    end_frame: int | None
    source_row: int
```

If the actual file contains additional fields such as play name, video number, player position, target track, etc., preserve them in an `extra` mapping rather than discarding them.

### 3.3 MOT ZIP input

The loader must support the CVAT MOT ZIP structure used by the project.

At minimum, it must extract:

- frame index;
- track ID;
- bbox coordinates;
- object label/category (`player` vs `ball`);
- confidence/visibility fields if present.

Normalized record:

```python
@dataclass(frozen=True)
class MotDetection:
    frame: int
    track_id: int
    label: Literal["player", "ball"]
    bbox_xywh: tuple[float, float, float, float]
    confidence: float | None
    visibility: float | None
```

The code must not assume every frame has exactly 22 player boxes.

### 3.4 Dataset summary

Normalize at least:

```python
@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    dataset_order: int
    view_raw: str | None
    # preserve all other existing columns as extra metadata
```

`view_raw` may be blank.

---

## 4. Expected Outputs

V1 must generate **separate per-video position mappings** and a paired-play review artifact.

### 4.1 PlayerTrack-compatible CSV

The code must produce a CSV compatible with the current downstream PlayerTrack workflow.

Do not invent a completely incompatible replacement format.

If the existing PlayerTrack CSV has a fixed schema, write that schema. If additional inference metadata is needed, write a sidecar detailed CSV/JSON rather than breaking downstream compatibility.

### 4.2 Detailed inference JSON

Create a machine-readable sidecar for every video:

```json
{
  "schema_version": "1.0",
  "video_id": "JetSweep_1",
  "view": "sideline",
  "view_confidence": 0.98,
  "pair_id": "pair_000123",
  "pair_confidence": 0.91,
  "offense_direction": "...",
  "offense_direction_confidence": 0.87,
  "assignments": [
    {
      "slot_id": "offense.C_1",
      "side": "offense",
      "position": "C",
      "track_id": 7,
      "visibility": "visible",
      "confidence": 0.99,
      "evidence": {
        "action_semantics": 1.0,
        "geometry": 0.93,
        "learned_model": 0.88,
        "formation_constraints": 1.0,
        "paired_view": 0.95
      },
      "alternatives": []
    },
    {
      "slot_id": "offense.WR_3",
      "side": "offense",
      "position": "WR",
      "track_id": null,
      "track_id_display": "not_visible",
      "visibility": "out_of_view",
      "confidence": 0.92,
      "evidence": {
        "paired_view": 0.96,
        "personnel_constraint": 0.94
      }
    }
  ],
  "rejected_tracks": [],
  "suspected_id_switches": [],
  "warnings": [],
  "status": "AUTO_ACCEPTED"
}
```

Internally use `None`/`null` for missing track identity. Render the text `not_visible` only where required by CSV compatibility or display output.

### 4.3 Human review report

Generate Markdown with concise, auditable sections:

1. pair/view summary;
2. inferred personnel;
3. offense assignments;
4. defense assignments;
5. `not_visible` slots;
6. rejected/extra tracks;
7. suspected ID switches;
8. action anchors used;
9. paired-view disagreements;
10. low-confidence assignments;
11. final status/reasons.

The report must show **why** a role was assigned, not merely the final label.

---

## 5. High-Level Architecture

```text
Dataset discovery
    ↓
View metadata normalization + view classifier
    ↓
Sideline/endzone pair inference
    ↓
Per-view MOT + action loading
    ↓
Track quality preprocessing
    ↓
Snap/pre-snap window inference
    ↓
Action semantic anchors
    ↓
Stable pre-snap geometric features
    ↓
Offensive direction inference
    ↓
Offense/defense separation
    ↓
View-specific role probability scoring
    ↓
Paired-view role/personnel evidence fusion
    ↓
Global constrained formation solver
    ↓
not_visible completion for cropped endzone roles
    ↓
Confidence calibration + review policy
    ↓
PlayerTrack-compatible CSV + JSON + review report
```

### Critical architectural rule

Do **not** classify each track independently and accept per-track argmax labels. Position assignment must be solved jointly across the formation with football constraints.

---

## 6. Recommended Repository Structure

```text
player_position_inference/
├── pyproject.toml
├── README.md
├── src/
│   └── position_inference/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       │
│       ├── data/
│       │   ├── dataset_summary.py
│       │   ├── mot_loader.py
│       │   ├── action_loader.py
│       │   ├── playertrack_loader.py
│       │   ├── schemas.py
│       │   └── discovery.py
│       │
│       ├── pairing/
│       │   ├── view_classifier.py
│       │   ├── pair_builder.py
│       │   └── pair_confidence.py
│       │
│       ├── quality/
│       │   ├── track_stats.py
│       │   ├── player_validity.py
│       │   ├── false_positive_filter.py
│       │   └── id_switch_detector.py
│       │
│       ├── geometry/
│       │   ├── footpoints.py
│       │   ├── presnap_window.py
│       │   ├── normalization.py
│       │   ├── spatial_features.py
│       │   ├── direction.py
│       │   └── team_partition.py
│       │
│       ├── semantics/
│       │   ├── action_rules.py
│       │   ├── action_anchors.py
│       │   ├── personnel.py
│       │   └── formation_rules.py
│       │
│       ├── learning/
│       │   ├── feature_matrix.py
│       │   ├── train_role_models.py
│       │   ├── role_model.py
│       │   ├── calibration.py
│       │   └── model_io.py
│       │
│       ├── inference/
│       │   ├── candidate_scores.py
│       │   ├── offense_solver.py
│       │   ├── defense_solver.py
│       │   ├── assignment_solver.py
│       │   ├── paired_fusion.py
│       │   ├── missing_slots.py
│       │   └── confidence.py
│       │
│       ├── output/
│       │   ├── playertrack_writer.py
│       │   ├── json_writer.py
│       │   └── review_writer.py
│       │
│       └── evaluation/
│           ├── metrics.py
│           ├── evaluator.py
│           └── reports.py
│
├── config/
│   ├── action_role_rules.yaml
│   ├── position_taxonomy.yaml
│   ├── scoring_weights.yaml
│   ├── pairing.yaml
│   ├── confidence.yaml
│   └── model.yaml
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│       └── jetsweep_1/
│
└── scripts/
    ├── inspect_inputs.py
    ├── train_role_models.py
    └── evaluate_labeled_subset.py
```

Keep V1 implementation readable and testable. Do not prematurely merge this into the TapeVision neural-network modules.

---

## 7. Core Domain Data Models

Use typed dataclasses or Pydantic models.

### 7.1 Track summary

```python
@dataclass
class TrackSummary:
    track_id: int
    label: str
    frames_present: list[int]
    detections: list[MotDetection]
    first_frame: int
    last_frame: int
    num_boxes: int
    coverage_ratio: float
    median_bbox_height: float
    median_bbox_width: float
    median_footpoint: tuple[float, float] | None
    presnap_median_footpoint: tuple[float, float] | None
    presnap_motion: float | None
    validity_score: float
    validity_flags: list[str]
```

### 7.2 Role evidence

```python
@dataclass
class RoleEvidence:
    track_id: int
    side_probabilities: dict[str, float]
    role_probabilities: dict[str, float]
    action_scores: dict[str, float]
    geometry_scores: dict[str, float]
    learned_scores: dict[str, float]
    paired_scores: dict[str, float]
    flags: list[str]
```

### 7.3 Assignment

```python
@dataclass
class PositionAssignment:
    slot_id: str
    side: Literal["offense", "defense"]
    position: str
    track_id: int | None
    visibility: Literal[
        "visible",
        "occluded_or_sparse",
        "out_of_view",
        "unknown"
    ]
    confidence: float
    alternatives: list[tuple[str, float]]
    evidence: dict[str, float]
    flags: list[str]
```

### 7.4 View result

```python
@dataclass
class ViewInferenceResult:
    video_id: str
    view: Literal["sideline", "endzone", "unknown"]
    view_confidence: float
    offense_direction: str | None
    offense_direction_confidence: float
    assignments: list[PositionAssignment]
    rejected_track_ids: list[int]
    suspected_id_switches: list[dict]
    personnel_hypothesis: dict[str, int]
    confidence: float
    status: str
```

---

## 8. View Classification

### 8.1 Metadata first

If `DatasetSummary.csv.view` contains a recognized nonblank value, treat it as the primary signal but still run sanity validation.

### 8.2 Inference for missing values

Implement a pluggable view classifier.

V1 may initially use geometric/statistical features from MOT rather than raw-image CNN inference if videos/frames are not part of the repository input path.

Potential signals:

- number of unique visible player tracks during stable pre-snap frames;
- maximum simultaneous player count;
- horizontal vs vertical spread of player footpoints;
- proportion of players near frame boundaries;
- formation aspect ratio;
- player bbox scale distribution;
- expected endzone crop behavior (fewer wide players, more trench concentration).

If source video frames are available, allow a later visual classifier without changing the interface.

Return:

```python
ViewPrediction(
    view="sideline" | "endzone" | "unknown",
    confidence=float,
    evidence=dict
)
```

Low-confidence view classification must not silently become sideline/endzone.

---

## 9. Sideline/Endzone Pairing

Pairing must not depend on synchronized frames or shared track IDs.

### 9.1 Primary heuristic

Use dataset order:

- clip `i` sideline candidate;
- clip `i+1` endzone candidate.

### 9.2 Supporting signals

Use:

- populated `view` metadata;
- inferred view probabilities;
- compatible play/file naming stems when present;
- Key Actions compatibility:
  - both contain snap-related actions;
  - similar action vocabulary consistent with the same play;
- approximate clip ordering/grouping.

### 9.3 Pair confidence

Compute a score and retain evidence.

Do not pair when confidence is below configured minimum.

Status examples:

- `PAIR_CONFIRMED_METADATA`
- `PAIR_INFERRED_HIGH_CONFIDENCE`
- `PAIR_REVIEW_REQUIRED`
- `UNPAIRED`

### 9.4 No direct ReID requirement

Do not attempt appearance-based sideline/endzone identity matching in V1.

---

## 10. MOT Quality Preprocessing

### 10.1 Ball separation

All MOT rows labeled `ball` must be excluded from the pool of player-role candidates and stored separately.

Ball trajectories may later contribute to snap/possession evidence.

### 10.2 Player track validity

For every `player` track, calculate:

- number of frames present;
- pre-snap coverage;
- median bbox size;
- spatial continuity;
- speed/jump statistics;
- overlap/duplication behavior;
- whether track appears only briefly after the play begins;
- distance from formation cluster;
- whether action annotation references the track.

Create `player_validity_score`.

A track referenced by a high-confidence football action must not be rejected solely because of sparse boxes.

### 10.3 False positives/referees

Use conservative rejection.

Possible evidence:

- very short lifetime;
- low pre-snap coverage;
- geometry incompatible with either team;
- isolated position away from both formation clusters;
- duplicate overlapping trajectory;
- excessive simultaneous player count;
- no action/formation support.

Do not delete tracks from source data. Mark them as rejected for inference.

### 10.4 ID switch detection

V1 must flag likely switches using:

- sudden spatial discontinuity;
- abrupt bbox-scale change;
- role/formation identity contradiction;
- trajectory continuity with a different track;
- simultaneous disappearance/appearance nearby.

Primary role inference should rely on stable pre-snap history. Post-snap switches must not change the player's inferred football position.

Output suspected switches with frame range and confidence.

---

## 11. Snap and Stable Pre-Snap Window

### 11.1 Snap anchor priority

Use Key Actions first.

Strong snap-related actions include:

- `Ball Snap` / `Action_BallSnap` → Center actor;
- `Snap Receive` / `Action_SnapReceive` → QB actor.

The exact CSV spelling must be normalized through configurable aliases.

### 11.2 Stable window

Do not use a single frame.

Default strategy:

1. identify snap-start frame or closest reliable snap boundary;
2. examine a configurable interval before snap, e.g. 30 frames;
3. exclude the final few frames before snap if players have begun moving;
4. compute robust medians across the remaining stable frames;
5. allow pre-snap motion players to have a separate motion-aware representation.

Config example:

```yaml
presnap:
  lookback_frames: 30
  snap_exclusion_frames: 3
  minimum_stable_frames: 8
  aggregation: median
```

If Jet Motion occurs during the pre-snap interval, do not treat the moving WR's displacement as tracking noise. Preserve motion features and use the action semantic anchor.

---

## 12. Geometry and Normalization

### 12.1 Player contact point

For bbox `[x, y, w, h]`:

```python
foot_x = x + w / 2
foot_y = y + h
```

Use bottom-center as the default image-space player contact point, consistent with future field projection.

### 12.2 Robust track location

For stable pre-snap frames:

```python
median_footpoint = median(valid footpoints)
```

Also calculate variance, range, velocity, and coverage.

### 12.3 V1 local normalization

Field registration is **not required for V1**.

Use a Center-relative or formation-relative normalized coordinate system once Center is identified.

Candidate example:

```python
x_norm = (x_i - x_center) / scale

y_norm = (y_i - y_center) / scale
```

`scale` should be robust, e.g. median bbox height of valid nearby players or Center bbox height when reliable.

Do not make the exact formula unconfigurably hard-coded; evaluate it on labeled data.

### 12.4 Future field coordinates

Design the geometry API to accept optional field-projected coordinates later. The project's broader TapeVision output already anticipates field registration and bottom-center projection, but V1 must remain functional in image coordinates.

---

## 13. Action Semantics / Football Anchors

Action semantics are a first-class evidence source.

Implement `config/action_role_rules.yaml`.

Example structure:

```yaml
BallSnap:
  aliases: ["Ball Snap", "Action_BallSnap"]
  roles:
    C: 1.0
  mode: hard_anchor

SnapReceive:
  aliases: ["Snap Receive", "Action_SnapReceive"]
  roles:
    QB: 1.0
  mode: hard_anchor

Toss:
  aliases: ["Toss", "Action_Toss"]
  roles:
    QB: 0.95
  mode: strong

JetMotion:
  aliases: ["Jet Motion", "Action_JetMotion"]
  roles:
    WR: 0.90
    TE: 0.10
  mode: strong

BallCarry_JetSweep:
  play: JetSweep
  action_aliases: ["Ball Carry", "Action_BallCarry"]
  roles:
    WR: 0.85
    RB: 0.15
  mode: play_conditioned
```

### 13.1 Hard vs soft rules

Use hard anchors only for relationships that are definitionally reliable in this dataset.

Examples intended as hard/near-hard:

- Ball Snap actor → `C`;
- Snap Receive actor → `QB`.

Examples that should usually remain probabilistic:

- Jet Motion → often `WR`, sometimes another eligible player;
- blocking action → positional group evidence, not necessarily an exact role;
- route action → receiver/TE evidence;
- ball carry → play-conditioned skill-position evidence.

### 13.2 Tight end evidence

TE may align wide and can resemble WR geometrically.

Use combined evidence:

- normalized alignment;
- proximity to tackle before motion when available;
- blocking actions when present;
- route actions;
- learned bbox morphology/scale features;
- paired personnel constraints.

Apparent bbox height/body size is **auxiliary only**, because perspective can dominate raw pixel size.

---

## 14. Offensive Direction Inference

V1 must infer offensive direction.

Use an evidence ensemble rather than one rule.

Potential signals:

1. Center → QB relative alignment;
2. offensive line plane / LOS orientation;
3. backfield location relative to line;
4. early post-snap offensive movement;
5. ball snap movement when reliable;
6. paired-view consistency.

The solver must produce:

```python
OffenseDirectionPrediction(
    direction=<view-relative enum>,
    confidence=float,
    evidence=dict
)
```

Do not use screen-left/screen-right naively to label LT/RT. LT/LG/C/RG/RT are from the offense's perspective.

Low-confidence direction should propagate into lower LT/LG/RG/RT confidence and possibly human review.

---

## 15. Offense/Defense Separation

This is required in V1.

The system must separate candidate player tracks into offensive and defensive groups before fine-grained role assignment.

### 15.1 Strong offensive anchors

When available:

- Center from Ball Snap;
- QB from Snap Receive;
- motion/ball-carry actors;
- offensive blocking actions.

These become seed tracks.

### 15.2 Geometric partition

Use pre-snap formation geometry around the inferred LOS/Center to score each player for offense vs defense.

Possible features:

- side of LOS relative to Center/QB;
- cluster membership;
- distance to offensive line;
- stance/alignment depth;
- nearest-neighbor relationships;
- action semantic evidence.

### 15.3 Cardinality

Sideline usually should resolve to 11 offense + 11 defense, after rejecting extras.

Endzone may contain fewer visible players. Do **not** force 22 visible tracks in endzone.

The canonical formation must still represent the expected 11 offensive and 11 defensive slots when sufficiently supported by paired-view evidence.

---

## 16. Offensive Position Inference

Use a hierarchical structured approach.

### 16.1 Center

Primary anchor:

- Ball Snap actor → Center.

Fallback when Ball Snap annotation is missing/unusable:

- identify offensive-line cluster;
- score central OL candidate using geometry relative to QB/ball/line;
- paired-view evidence;
- learned role scorer.

### 16.2 Offensive line

Once Center is anchored:

1. identify four nearest credible OL neighbors aligned on/near the LOS;
2. solve ordered sequence around Center;
3. infer offensive direction/perspective;
4. assign:
   - LT;
   - LG;
   - C;
   - RG;
   - RT.

The five-line structure must be solved jointly. Do not independently label five tracks and hope they form a legal line.

### 16.3 QB

Primary anchor:

- Snap Receive actor.

Supporting:

- position behind Center;
- Toss/Throw/Fake/Handoff actions;
- backfield geometry;
- learned role model.

### 16.4 RB/FB

Use:

- depth behind LOS;
- position relative to QB;
- offset/alignment;
- motion;
- play-conditioned actions;
- personnel constraints.

`FB` is optional and must not be forced when unsupported.

### 16.5 TE/WR

Use:

- lateral displacement from Center/tackle box;
- LOS depth;
- proximity to tackle;
- action semantics;
- learned morphology features;
- personnel hypothesis;
- paired-view evidence.

Wide-aligned TE vs WR ambiguity must be represented in alternatives/confidence rather than resolved by raw height alone.

---

## 17. Defensive Position Inference

Use hierarchical classification.

### 17.1 Group stage

First infer broad groups:

- first level / defensive front;
- second level / linebackers;
- third level / defensive backs.

### 17.2 Fine stage

Then resolve:

- front → `DE` vs `DT`;
- second level → `LB`;
- defensive backs → `CB`, `FS`, `SS`.

### 17.3 Features

Use:

- depth relative to inferred LOS;
- lateral offset from Center;
- relation to offensive tackles/TEs;
- relation to WR alignment;
- deep-vs-box alignment;
- relative ordering within defensive group;
- paired-view visibility and personnel evidence.

### 17.4 Endzone omissions

Endzone often lacks wide CBs/safeties/receivers.

Use paired sideline personnel to create `not_visible` defensive slots when supported.

Do not hallucinate a visible track assignment for an absent player.

---

## 18. Learned Role Scorers

V1 should support learned tabular role scoring, not a required deep visual network.

### 18.1 Separate models by view

Train/calibrate separate scorers:

- `sideline_role_model`;
- `endzone_role_model`.

Because labeled PlayerTrack examples include both views, do not force one geometry model across both camera regimes.

### 18.2 Candidate features

At minimum evaluate:

- normalized x/y;
- distance from Center;
- signed lateral offset from Center;
- depth from inferred LOS;
- distance from QB;
- nearest-neighbor distances;
- lateral rank;
- depth rank;
- bbox height normalized by local scale;
- bbox width normalized by local scale;
- bbox aspect ratio;
- pre-snap coverage;
- pre-snap motion magnitude;
- action semantic flags;
- broad side/group probability;
- view;
- play type if known reliably from input context.

### 18.3 Baseline algorithms

Implement a clean baseline interface that supports:

1. multinomial logistic regression;
2. random forest / histogram gradient boosting;
3. optional XGBoost/LightGBM only if dependency policy permits.

Prefer scikit-learn baseline first for reproducibility and low dependency burden.

### 18.4 Output

The learned model produces probabilities, not final assignments:

```python
P(position | track_features, view)
```

These probabilities feed the constrained solver.

### 18.5 Leakage prevention

Split evaluation by **underlying football play pair**, not individual track rows, so paired views or players from the same play do not leak across train/test folds.

---

## 19. Paired-View Evidence Fusion

This is mandatory in V1.

### 19.1 Fusion level

Fuse at:

- inferred personnel counts;
- canonical slot hypotheses;
- role probability priors;
- view classification confidence;
- action-semantic anchors;
- broad formation structure.

Do not fuse by matching frames.

Do not require track-ID correspondence.

### 19.2 Complementary view weighting

Default conceptual weighting:

- **sideline** stronger for:
  - complete 22-player visibility;
  - WR/CB/safety spacing;
  - total personnel counts;
  - wide formation structure.
- **endzone** stronger for:
  - OL ordering/spacing;
  - DL/front alignment;
  - QB/backfield depth;
  - interior TE/tackle relationships.

Weights must be configurable and calibratable from labeled pairs.

### 19.3 Missing-player reasoning

If sideline strongly establishes a complete formation but endzone shows fewer valid players:

- infer expected canonical slots;
- solve visible endzone tracks against those slots;
- leave unsupported slots as `not_visible` instead of forcing extra/poor tracks into them.

### 19.4 Pair disagreements

Compute paired consistency score.

If two high-confidence hypotheses conflict and neither exceeds the configured confidence margin:

- do not silently choose;
- mark `PAIR_REVIEW_REQUIRED`.

---

## 20. Global Assignment Solver

### 20.1 Objective

For candidate track-to-slot assignment `A`, maximize a weighted score such as:

```text
S(A) =
    w_action * S_action
  + w_geometry * S_geometry
  + w_model * S_learned
  + w_formation * S_formation
  + w_pair * S_paired_view
  + w_quality * S_track_quality
```

Weights are config-driven.

### 20.2 Solver recommendation

Use **OR-Tools CP-SAT** if practical because V1 has:

- flexible role counts;
- hard unique-role constraints;
- optional FB/TE/WR counts;
- missing/not_visible endzone slots;
- rejected tracks;
- pair-level constraints.

A Hungarian assignment baseline is acceptable only if the implementation still correctly handles variable slots and missing assignments. Do not distort the problem merely to fit Hungarian matching.

### 20.3 Core hard constraints

Offense:

- exactly one `C`;
- exactly one `LT`;
- exactly one `LG`;
- exactly one `RG`;
- exactly one `RT`;
- normally exactly one primary `QB`;
- 11 canonical offensive player slots total when complete personnel can be inferred;
- each visible track maps to at most one slot;
- each canonical slot maps to at most one visible track.

Defense:

- 11 canonical defensive slots total when complete personnel can be inferred;
- each visible track maps to at most one slot;
- duplicate general labels allowed through unique slot IDs.

Endzone:

- canonical slots may map to `not_visible`;
- `not_visible` has a penalty/cost but must be preferable to assigning an obvious false positive.

### 20.4 Soft constraints

Examples:

- OL approximate alignment;
- guard/tackle adjacency around Center;
- QB behind Center;
- WRs generally outside tackle box;
- CBs often associated with wide receivers;
- safeties deeper than front/LB groups;
- TE blocking evidence;
- paired-view personnel agreement.

Soft constraints must not make rare legal formations impossible.

---

## 21. Canonical Slot Numbering

Because general labels repeat, define deterministic slot ordering.

V1 rule:

- unique roles use `_1` (`QB_1`, `C_1`, etc.);
- repeated roles are numbered by stable offense-perspective lateral order when visible geometry supports it;
- if a repeated role is `not_visible`, assign remaining slot number based on paired-view canonical personnel ordering;
- never infer X/Z/nickel/etc. from the number.

Example:

```text
offense.WR_1 = leftmost WR from offense perspective
offense.WR_2 = next WR
offense.WR_3 = next WR
```

Document the exact ordering convention in generated metadata.

If offensive direction is unresolved, slot numbering may be provisional and must carry a warning.

---

## 22. Confidence Model and Review States

### 22.1 Assignment confidence

Do not expose raw solver objective values directly as probabilities.

Create calibrated confidence using labeled data where possible.

Factors:

- semantic anchor strength;
- gap between best and second-best role/slot;
- solver objective margin between best and alternative valid formations;
- track validity;
- direction confidence;
- paired-view agreement;
- visibility quality;
- learned model calibration.

### 22.2 Suggested default states

Configurable thresholds:

```yaml
confidence:
  auto_accept: 0.90
  review_recommended: 0.70
  pair_resolution_margin: 0.12
```

Interpretation:

- `>= auto_accept`: eligible for auto-accept if no hard warnings;
- between review threshold and auto threshold: `REVIEW_RECOMMENDED`;
- below review threshold: `HUMAN_REQUIRED`.

### 22.3 Hard review triggers

Even if numeric confidence is high, force review for:

- unresolved sideline/endzone pairing;
- incompatible hard action anchors;
- no credible Center or QB after fallback;
- high-confidence pair personnel disagreement below resolution margin;
- impossible formation constraints;
- suspected pre-snap ID switch involving a hard anchor;
- unresolved offensive direction affecting OL labels;
- too many/reliably too few sideline football-player tracks to construct a credible formation.

---

## 23. Configuration Files

### 23.1 `position_taxonomy.yaml`

Define:

- valid offense/defense labels;
- group hierarchy;
- hard unique roles;
- allowed role-count ranges.

### 23.2 `action_role_rules.yaml`

Define:

- action aliases;
- role probabilities;
- hard vs soft anchor type;
- optional play conditioning.

### 23.3 `scoring_weights.yaml`

Define source weights by view and inference stage.

### 23.4 `pairing.yaml`

Define:

- adjacency prior;
- view-order prior;
- minimum pair confidence;
- metadata weighting.

### 23.5 `confidence.yaml`

Define review thresholds and pair disagreement margin.

No critical football scoring constants should be buried across Python files.

---

## 24. CLI Requirements

Provide clear commands.

Suggested:

```bash
# Inspect schemas and a fixture
python -m position_inference.cli inspect \
  --mot JetSweep_1_cvat_mot.zip \
  --actions KeyActions_Sheet.csv \
  --playertracks PlayerTrack_ID_Sheet.csv

# Infer one view
python -m position_inference.cli infer-video ...

# Infer one paired play
python -m position_inference.cli infer-pair \
  --sideline-mot ... \
  --sideline-actions ... \
  --endzone-mot ... \
  --endzone-actions ...

# Build inferred pairs from dataset order + DatasetSummary.csv
python -m position_inference.cli build-pairs ...

# Train/calibrate role models
python -m position_inference.cli train-role-models ...

# Evaluate all labeled clips/pairs
python -m position_inference.cli evaluate ...

# Batch inference
python -m position_inference.cli batch-infer ...
```

CLI errors must identify the offending file/video and schema issue.

---

## 25. Golden JetSweep_1 Test

Use the supplied JetSweep_1 files as the first end-to-end integration fixture.

### 25.1 Requirements

Antigravity must:

1. inspect the actual `PlayerTrack_ID_Sheet(2).csv` schema;
2. inspect the actual `KeyActions_Sheet(2).csv` schema;
3. inspect `JetSweep_1_cvat_mot.zip` and its label/category representation;
4. build normalized adapters without hardcoding JetSweep-specific column indices;
5. derive the authoritative expected role mapping from the supplied PlayerTrack annotation;
6. infer roles using MOT + action semantics;
7. compare inferred assignments with the ground truth;
8. emit a detailed review report showing each mismatch and evidence score.

### 25.2 Important Jet Sweep semantic anchors

V1 should recognize at minimum:

- Ball Snap → Center;
- Snap Receive → QB;
- Jet Motion → strong WR evidence;
- Toss → strong QB evidence;
- Ball Carry on Jet Sweep → strong ballcarrier/WR evidence;
- Zone/lead/second-level blocking → offensive blocker/OL/TE evidence as applicable.

Do not hardcode specific track IDs from JetSweep_1 into inference logic. Ground-truth IDs belong only in tests/fixtures.

### 25.3 Golden-test success

The first milestone is not “train a model.” It is:

> The deterministic/hybrid pipeline can load the real JetSweep_1 artifacts, establish the semantic anchors, extract stable geometry, solve a complete position mapping for each available view, and compare that result against the supplied PlayerTrack ground truth with transparent evidence.

---

## 26. Evaluation Protocol

### 26.1 Unit of split

Evaluation splits must be grouped by underlying paired play to prevent leakage.

### 26.2 Metrics

At minimum report separately for sideline, endzone, and combined paired inference:

1. overall visible-player role accuracy;
2. offense role accuracy;
3. defense role accuracy;
4. OL exact-player accuracy;
5. QB accuracy;
6. Center accuracy;
7. skill-position accuracy;
8. complete offense formation accuracy;
9. complete defense formation accuracy;
10. complete visible-view mapping accuracy;
11. `not_visible` slot precision/recall for endzone;
12. top-2 role accuracy;
13. high-confidence precision;
14. coverage at confidence >= 0.90;
15. expected calibration error or reliability summary;
16. paired-view disagreement rate;
17. false-positive rejection precision/recall where ground truth permits;
18. suspected-ID-switch report count and manually verifiable examples.

### 26.3 Operational primary metric

Prioritize:

```text
Precision(assignments with confidence >= AUTO_ACCEPT threshold)
```

The goal is safe annotation automation, not merely maximizing average accuracy.

### 26.4 Baselines

Compare at least:

- geometry-only;
- action-semantic + geometry;
- learned probabilities only;
- structured solver without pair fusion;
- full structured solver with pair fusion.

This ablation is important to prove paired-view and football-knowledge value.

---

## 27. V1 Implementation Phases

Antigravity should implement in this order.

### Phase 0 — Repository/bootstrap

- project structure;
- typed schemas;
- config loading;
- logging;
- pytest;
- input schema inspector.

### Phase 1 — Real input adapters + JetSweep fixture

- MOT ZIP loader;
- Key Actions loader;
- PlayerTrack loader;
- DatasetSummary loader;
- JetSweep_1 inspection test.

### Phase 2 — Single-view deterministic inference

- track summaries;
- ball separation;
- snap/pre-snap window;
- action anchors;
- bottom-center geometry;
- normalization;
- direction estimate;
- offense/defense seed partition;
- OL/QB/basic skill-position solver;
- defensive hierarchy baseline.

### Phase 3 — Pairing + endzone missing roles

- view inference;
- pair builder;
- paired personnel priors;
- `not_visible` slot support;
- disagreement policy.

### Phase 4 — Learned role probability models

- feature matrix from labeled clips;
- pair-grouped validation;
- separate sideline/endzone models;
- probability calibration;
- integration into solver.

### Phase 5 — Quality/review system

- false-positive filtering;
- suspected ID-switch reporting;
- confidence calibration;
- Markdown review report;
- auto/review states.

### Phase 6 — Batch inference

- batch discovery;
- resumable processing;
- deterministic outputs;
- summary CSV/JSON;
- failure isolation per pair/video.

---

## 28. Non-Goals for V1

Do not expand scope unless required to satisfy the above behavior.

V1 does **not** need to:

- train a new object detector;
- train a new MOT tracker;
- automatically repair/rewrite ID switches;
- synchronize sideline and endzone frames;
- perform cross-view appearance ReID;
- infer jersey numbers;
- use jersey OCR;
- require field homography/registration;
- infer X/Z/slot WR labels;
- infer detailed linebacker/defensive-line subtypes beyond the requested taxonomy;
- merge the implementation into TapeVision's neural model;
- modify existing source MOT ZIP files.

Architecture should allow these later.

---

## 29. Engineering Quality Requirements

### 29.1 Determinism

Given fixed inputs/config/model files, inference must be reproducible.

Set and record random seeds where applicable.

### 29.2 Explainability

Every assignment must retain decomposed evidence.

Never output only a black-box class/confidence when action/geometry/solver evidence exists.

### 29.3 No silent schema coercion

If a source CSV or MOT ZIP schema differs from the loader expectation:

- fail with a targeted validation error;
- show detected headers/files;
- do not silently shift column meanings.

### 29.4 No source mutation

Do not modify:

- original MOT ZIPs;
- original Key Actions CSVs;
- original PlayerTrack ground-truth CSVs;
- DatasetSummary.csv.

Write outputs to dedicated directories.

### 29.5 Logging

Use structured logging with:

- pair_id;
- video_id;
- view;
- phase;
- status;
- warnings.

### 29.6 Failure isolation

Batch inference must continue when one pair/video fails, recording a failed result with reason.

---

## 30. Testing Requirements

### 30.1 Unit tests

Cover:

- bbox bottom-center;
- stable median aggregation;
- missing-box handling;
- ball/player separation;
- action alias normalization;
- hard action anchors;
- Center-relative normalization;
- slot uniqueness;
- `not_visible` serialization;
- confidence threshold states;
- pair confidence logic;
- solver cardinality constraints;
- duplicate role slot numbering.

### 30.2 Synthetic solver tests

Create controlled formations where expected assignment is unambiguous.

Include:

- standard 3-WR/1-TE set;
- 2-TE set;
- FB personnel;
- endzone crop missing two WRs and two CB/DB players;
- extra referee/false-positive track;
- missing OL boxes;
- ambiguous TE/WR;
- suspected ID switch after snap;
- low-confidence direction.

### 30.3 Integration tests

- JetSweep_1 real fixture;
- at least one labeled sideline/endzone pair;
- batch of multiple labeled pairs;
- compare generated PlayerTrack-compatible CSV against ground truth.

### 30.4 Regression tests

Once a labeled fixture is passing, freeze expected normalized inputs/results within reasonable numerical tolerance.

---

## 31. Acceptance Criteria

V1 is complete when all of the following are true:

1. The codebase loads the actual project MOT ZIP, Key Actions CSV, PlayerTrack CSV, and DatasetSummary formats.
2. It can infer or consume sideline/endzone view labels.
3. It can construct paired-view candidates using dataset adjacency and supporting evidence.
4. It does not require shared track IDs or frame synchronization.
5. It separates `ball` from player candidates.
6. It tolerates missing MOT boxes.
7. It rejects/flags obvious extra non-football-player tracks without mutating source MOT.
8. It detects/reports suspected ID switches without rewriting tracks.
9. It identifies action-semantic anchors including Center and QB when corresponding actions exist.
10. It infers offensive direction with confidence.
11. It separates offense and defense.
12. It jointly assigns the requested offensive and defensive position taxonomy.
13. It supports duplicate general roles through canonical slot IDs.
14. It outputs all expected endzone slots, using `not_visible` for players outside the view.
15. It fuses paired-view role/personnel evidence without direct cross-view ReID.
16. It automatically resolves pair disagreement only above configurable confidence margin; otherwise it flags review.
17. It trains/evaluates separate sideline/endzone learned tabular role scorers from the labeled PlayerTrack subset.
18. It emits assignment-level confidence plus decomposed evidence.
19. It writes a PlayerTrack-compatible mapping plus detailed JSON and Markdown review output.
20. It evaluates the labeled subset with the metrics in Section 26.
21. JetSweep_1 is implemented as a real integration/golden fixture and no JetSweep-specific track IDs are hardcoded into production inference logic.
22. Unit/integration tests pass in a clean environment.

---

## 32. Antigravity Implementation Instructions

Before writing production code, Antigravity must perform these steps:

1. Inspect the existing project/repository structure.
2. Inspect the exact headers and representative rows of:
   - `PlayerTrack_ID_Sheet(2).csv`;
   - `KeyActions_Sheet(2).csv`;
   - `DatasetSummary.csv` when available.
3. Inspect the internal files and category/label encoding of `JetSweep_1_cvat_mot.zip`.
4. Write a short implementation note documenting the observed schemas and any necessary adapter decisions.
5. Preserve existing naming conventions where feasible.
6. Implement the phases in Section 27 incrementally.
7. Run the JetSweep_1 golden test after every material inference change.
8. Do not hardcode the golden fixture's position mapping into production code.
9. Use configuration files for football action-role rules, confidence thresholds, and score weights.
10. Prefer transparent, testable heuristics/structured optimization before adding unnecessary deep-learning complexity.
11. Do not declare V1 successful solely because it returns 22 labels; validate against the manually annotated PlayerTrack ground truth and report accuracy/confidence.

If an input schema materially conflicts with this specification, change the **adapter**, not the core architecture, unless the schema proves one of the resolved assumptions impossible.

---

## 33. Future Extensions (Explicitly Deferred)

The architecture should leave clean extension points for:

- field segmentation + homography to football-yard coordinates;
- direct paired-view person ReID;
- synchronized multi-view fusion where timing becomes available;
- player appearance/jersey embeddings;
- automatic MOT ID-switch repair;
- uncertainty-guided active learning;
- finer role taxonomy;
- formation classification;
- direct integration as a TapeVision actor-role prior/token.

The current TapeVision output architecture already has a player `role` concept and field-registration-compatible track coordinates; this position-inference codebase should therefore expose clean data that can later be consumed by TapeVision without coupling V1 to the neural model.

---

## 34. Design Principle Summary

V1 must follow these principles:

- **Actions anchor identities.**
- **Pre-snap geometry defines formation structure.**
- **Football rules constrain legal assignments.**
- **Learned models provide probabilities, not final truth.**
- **The solver assigns the formation jointly.**
- **Sideline and endzone views provide complementary evidence.**
- **Cross-view fusion does not require cross-view track IDs.**
- **Missing endzone players become explicit `not_visible` slots.**
- **Noisy MOT is tolerated and audited, not silently rewritten.**
- **Confidence determines automation vs human review.**
- **The labeled subset is a benchmark, not merely an example set.**

