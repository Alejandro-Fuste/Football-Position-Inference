# Football Position Inference — V1 Implementation Correction Specification

## 1. Purpose

This document is the authoritative **correction specification** for the existing **Football Position Inference** codebase.

The repository already contains a useful V1 module structure, but the first implementation produced incorrect player-position results because several required behaviors from `player_position_inference_v1_spec.md` were simplified, bypassed, hard-coded, or implemented in the wrong order.

This correction specification does **not** authorize a rewrite of the repository from scratch.

The goal is to:

1. preserve the existing modular architecture where it is sound,
2. correct the implementation defects identified during the code audit,
3. replace placeholder/sequential heuristics with the structured inference system required by V1,
4. make JetSweep_1 a true golden acceptance test,
5. remove undeclared dependencies on external local repositories,
6. make the codebase safe to evaluate on the user's manually annotated data before scaling to the remaining dataset,
7. create a reproducible `tests/fixtures/` suite from small real Jet Sweep and Power examples.

This file should be used together with:

- `player_position_inference_v1_spec.md`

When there is a conflict between the current implementation and this correction specification, **this correction specification governs the repair**.

When there is a conflict between this correction specification and the original V1 specification, the correction specification governs only the explicitly corrected behavior below; all other original V1 requirements remain in force.

---

## 2. Current-State Assessment

The current repository structure is generally acceptable and should be retained:

```text
src/position_inference/
├── data/
├── evaluation/
├── geometry/
├── inference/
├── learning/
├── output/
├── pairing/
├── quality/
├── semantics/
├── pipeline.py
├── cli.py
└── config.py
```

The following architectural ideas should remain:

- separate MOT and action loaders
- player vs ball handling
- semantic action anchors
- pre-snap geometry
- offense/defense partitioning
- view-specific role scoring
- paired sideline/endzone processing
- canonical slot IDs
- explicit `not_visible` / `out_of_view` handling
- human-review states
- learned role-model support
- separate output writers
- evaluation metrics
- JetSweep golden testing

The problem is primarily in **implementation correctness**, not repository organization.

---

## 3. Non-Negotiable Correction Principles

### 3.1 Do not rebuild the project from scratch

Preserve the existing module boundaries unless a specific correction below requires refactoring.

### 3.2 Do not hard-code JetSweep_1 production logic

JetSweep_1 may be used as a test fixture, a golden acceptance case, and a debugging example. JetSweep_1 track IDs must never appear in production inference logic.

### 3.3 Do not replace global assignment with sequential heuristics

This is the most important correction.

The system MUST perform a **true joint constrained assignment optimization** for each side of the ball.

The current pattern:

```text
pick Center
then pick QB
then sort linemen
then pick TE
then pick RB
then assign remaining tracks to WR
```

is explicitly prohibited as the final V1 solver.

Likewise, the current defensive pattern of selecting inside tracks as DT, outside tracks as DE, then assigning remaining players as LB/CB/Safety is prohibited as the final V1 solver.

Heuristics may generate candidate probabilities, priors, penalties, and solver features. They may NOT directly become the final role assignments except when a role is a valid hard semantic anchor.

### 3.4 Do not treat fixed confidence constants as calibrated probabilities

Values such as `0.85`, `0.86`, and `0.88` must not be assigned solely because a heuristic rule fired.

### 3.5 Fail safely on missing or mismatched data

The system must never silently substitute unrelated action rows, unrelated tracking files, or unrelated video metadata.

### 3.6 Preserve original MOT track identities

V1 may detect suspected ID switches. V1 must not rewrite, merge, swap, or repair MOT track IDs automatically.

---

## 4. Defect 1 — Hidden Dependency on `Combine_Tracks_and_Actions`

### Current problem

The current JetSweep golden integration test uses an absolute local path into the separate `Combine_Tracks_and_Actions` project. This makes the repository non-portable, non-reproducible, and dependent on an undeclared external codebase.

### Required correction

Remove all absolute paths to external user directories from tests, runtime code, CLI defaults, and examples.

### Required data layout

Recommended project data layout:

```text
data/
├── dataset_summary/
│   └── DatasetSummary.csv
├── key_actions/
│   └── JetSweep.csv
├── player_tracks/
│   └── JetSweep.csv
└── tracking/
    └── JetSweep/
        ├── JetSweep_1_cvat_mot.zip
        └── JetSweep_2_cvat_mot.zip
```

Recommended test fixture layout:

```text
tests/
└── fixtures/
    └── jetsweep_1/
        ├── key_actions.csv
        ├── player_tracks.csv
        ├── sideline_mot.zip
        └── endzone_mot.zip
```

If large MOT ZIPs should not be committed, integration tests must use an environment variable or documented configurable path, e.g.:

```text
POSITION_INFERENCE_TEST_DATA=/path/to/test/data
```

The test must skip clearly if fixture data is unavailable.

### Acceptance criteria

- no `/Users/...` absolute paths remain in source/tests
- unit tests run without access to `Combine_Tracks_and_Actions`
- integration tests either run against fixtures or skip with a clear message

---

## 5. Defect 2 — Unsafe KeyActions Video Fallback

### Current problem

If no KeyActions row matches the requested `video_id`, the pipeline currently falls back to using all actions from the CSV.

### Required correction

Delete the fallback behavior.

### Required behavior

If the requested video is not found in a multi-video action file, raise a typed exception:

```text
ActionVideoNotFoundError
```

The error must include the requested `video_id`, action file path, and a short list of available video IDs.

An optional `allow_missing_actions=True` mode may continue with `actions=[]`, but the result must be at least `HUMAN_REQUIRED` and include `missing_action_annotations`.

### Acceptance tests

Add tests for exact match, prefixed numeric match, no match, ambiguous match, single-video file, and multi-video file.

---

## 6. Defect 3 — View Classification Runs Before Required Geometry Exists

### Current problem

The fallback view classifier expects pre-snap median footpoints, but the pipeline classifies the view before those features are populated.

### Required correction

Refactor the ordering so view classification has actual geometry available.

### Required pipeline order

```text
1. load MOT
2. summarize tracks
3. compute preliminary stable geometry
4. read DatasetSummary view metadata if available
5. classify view
6. identify snap
7. compute snap-relative pre-snap formation geometry
8. continue view-specific inference
```

Two-stage geometry is allowed: preliminary geometry for view classification and snap-relative geometry for position inference.

### Unknown state

If evidence is insufficient, allow:

```text
view = unknown
```

Do not automatically convert insufficient evidence to sideline.

### Acceptance criteria

- geometry-based view classification executes on populated features
- missing metadata does not automatically imply sideline
- obvious sideline and endzone fixtures classify correctly
- low-information input can remain `unknown`

---

## 7. Defect 4 — Offense/Defense Partition Assumes Every Action Actor Is Offensive

### Current problem

The current partitioner adds every action actor track to offense seeds. This is wrong for defensive actions such as Sack Quarterback, Tackle, Close On Quarterback, Wrap Quarterback, Attempt Block, and Block Kick.

### Required correction

Action semantic rules must include side evidence.

Example:

```yaml
Ball Snap:
  side:
    offense: 1.0
  roles:
    C: 1.0
  mode: hard_anchor

Sack Quarterback:
  side:
    defense: 1.0
  roles:
    DE: 0.30
    DT: 0.20
    LB: 0.30
    CB: 0.10
    SS: 0.10
  mode: soft
```

### Partition inputs

Offense/defense separation should combine hard/soft semantic side anchors, Center/QB relation, LOS side/depth, formation clustering, player-count constraints, track quality, and optionally paired-view evidence.

### No blind 11-track clipping

False positives must be allowed to remain `UNASSIGNED / NOISE` rather than being forced onto defense.

---

## 8. Defect 5 — Offensive Direction and Left/Right OL Assignment

### Current problem

The current solver uses the sign of image-space lateral offset to assign LT/LG/RG/RT without reliably transforming into the offensive perspective.

### Required correction

Create an explicit normalized offensive coordinate system and expose features such as:

```text
lateral_offense
depth_offense
dist_center
dist_qb
los_depth
```

All LT/LG/RG/RT reasoning must use the offensive perspective.

### Acceptance test

JetSweep_1 must correctly distinguish all four non-center offensive linemen.

---

## 9. Defect 6 — Candidate Scores Are Computed but Final Solvers Mostly Ignore Them

### Required correction

Every non-hard-anchor final assignment must be derived from the optimizer objective.

For track `t` and slot `s`, construct:

```text
score(t,s)
```

from configurable evidence including action semantics, geometry, learned model, formation structure, paired view, track quality, visibility, and personnel prior.

Example conceptual formula:

```text
S(t,s) =
    w_action * S_action
  + w_geometry * S_geometry
  + w_model * S_model
  + w_structure * S_structure
  + w_pair * S_pair
  + w_quality * S_quality
  - penalties
```

Hard anchors may constrain the solver directly when valid.

---

## 10. Defect 7 — No True Global CP-SAT Assignment

### Current problem

`assignment_solver.py` imports OR-Tools but does not construct and solve a real CP-SAT model.

### Required correction

Implement an actual global optimization model using OR-Tools CP-SAT.

### Decision variables

Recommended:

```text
x[t,s] ∈ {0,1}
```

for visible track-to-slot assignment.

Optional:

```text
n[t] ∈ {0,1}
```

for noise/unassigned tracks.

Optional:

```text
m[s] ∈ {0,1}
```

for active but not-visible slots.

### Required constraints

At minimum:

- each track occupies at most one slot
- each canonical slot has at most one visible track
- hard anchors are enforced when valid
- offensive candidates cannot fill defensive slots and vice versa unless side assignment itself is jointly modeled
- five OL positions obey legal offensive lateral ordering
- underlying personnel represents 11 offense and 11 defense
- endzone views may have fewer visible tracks
- extra detections may remain unassigned

### Objective

Maximize total assignment evidence plus formation/paired-view consistency and minus penalties.

### Explicit prohibition

It is NOT acceptable to make sequential decisions first and call CP-SAT only after the assignments are already effectively fixed.

### Solver diagnostics

Output solver status, objective score, best-bound score if available, solve time, and infeasibility warnings.

### Acceptance criterion

A test should fail if the solver does not construct and solve a real `cp_model.CpModel`.

---

## 11. Defect 8 — Offensive Personnel Is Hard-Coded

### Current problem

The current solver effectively assumes 1 QB, 1 RB, 1 TE, 3 WR, and 5 OL.

### Required correction

Keep unique QB and OL roles fixed, but make skill-position counts variable.

Canonical superset may include:

```text
offense.QB_1
offense.LT_1
offense.LG_1
offense.C_1
offense.RG_1
offense.RT_1
offense.RB_1
offense.RB_2
offense.FB_1
offense.TE_1
offense.TE_2
offense.TE_3
offense.WR_1
offense.WR_2
offense.WR_3
offense.WR_4
offense.WR_5
```

The solver must activate a legal 11-player combination rather than outputting every superset slot.

Expose personnel counts such as:

```json
{"RB":1,"FB":0,"TE":1,"WR":3}
```

---

## 12. Defect 9 — Defensive Personnel Is Hard-Coded

The V1 defense taxonomy remains:

```text
DE
DT
LB
CB
FS
SS
```

but counts must be flexible.

Use hierarchical evidence:

```text
front
second level
defensive backfield
```

then infer general role.

The optimizer should support common structures such as 4-3, 3-4, nickel, and other reasonable packages within configurable bounds.

If FS vs SS cannot be distinguished confidently, lower confidence and require review instead of inventing certainty.

---

## 13. Defect 10 — Paired-View Fusion Is Structurally Present but Functionally Weak

### Required correction

Paired-view fusion must occur at the **hypothesis/evidence level** before final assignments are locked.

V1 still does NOT require direct cross-view ReID.

### Required paired object

```text
play_pair_id
sideline_result
endzone_result
shared_personnel_hypothesis
shared_role_priors
pair_confidence
pair_warnings
```

### Required flow

```text
independent preliminary inference
↓
extract personnel/role hypotheses
↓
fuse paired evidence
↓
rerun final per-view global optimization
↓
produce final outputs
```

### Example

If sideline strongly supports 3 WR / 1 TE / 1 RB and endzone sees only one WR, endzone should preserve two additional WR slots as `not_visible` rather than changing personnel.

If the views imply incompatible personnel and the confidence margin is too small, set `PAIR_REVIEW_REQUIRED`.

---

## 14. Defect 11 — Confidence Is Artificially High and Not Calibrated

### Required correction

Remove fixed final confidence constants.

Assignment confidence should consider final candidate score, margin to the second-best legal assignment, solver ambiguity, semantic-anchor strength, track quality, paired-view support, learned-model probability, and evidence consistency.

Overall confidence should consider critical-role confidence, average/median assignment confidence, ambiguity count, pair consistency, unresolved view/direction warnings, and missing hard anchors.

Before large-scale auto-accept is enabled, calibrate thresholds on manually annotated data. Until then, default conservatively.

---

## 15. Defect 12 — JetSweep Golden Test Does Not Validate the Full Mapping

### Required correction

JetSweep_1 must become a true full-mapping golden integration test.

### Locked JetSweep_1 offense mapping

Use only in fixture/test expectations:

```text
WR  -> 1
LG  -> 3
LT  -> 5
C   -> 7
RG  -> 9
TE  -> 12
RT  -> 13
QB  -> 17
WR  -> 19
RB  -> 20
WR  -> 21
```

Do not place these IDs in production inference code.

### Required assertions

- all 11 offense assignments correct
- all visible defense assignments correct from actual ground truth fixture
- correct `not_visible` slots
- Center correct
- QB correct
- all five OL correct
- skill personnel counts correct
- view correct
- offensive direction correct
- no duplicate visible track assignments
- no noise track forced into a football slot

If JetSweep_1 and JetSweep_2 are paired views, evaluate both independently and as a pair.

---

## 16. Defect 13 — Learned Role Model Integration

Do not make learned-model training a prerequisite for structural repairs.

After deterministic/probabilistic solver corrections, support separate:

```text
sideline_role_model
endzone_role_model
```

using manually annotated PlayerTrack ground truth.

Recommended features:

```text
normalized x/y
offense-relative lateral position
LOS-relative depth
distance to Center
distance to QB
normalized bbox height/width
bbox aspect ratio
track coverage
pre-snap movement
relative rank
nearest-neighbor spacing
view
action semantic features
play type when available
```

Use a small tabular model for V1.

---

## 17. Defect 14 — Missing-Player Handling Must Depend on Personnel

Distinguish:

```text
ACTIVE_VISIBLE
ACTIVE_NOT_VISIBLE
INACTIVE_SLOT
```

Example: in 11 personnel, inactive slots such as TE_2, TE_3, WR_4, WR_5, RB_2, FB_1 must not appear as `not_visible`.

If endzone misses two of three active WRs:

```text
WR_1 = visible
WR_2 = not_visible
WR_3 = not_visible
```

Normal PlayerTrack output should include every active expected slot and exclude inactive superset slots.

---

## 18. Defect 15 — False Positives / Referees Must Be Allowed to Remain Unassigned

The solver must include a noise/unassigned option.

Do not assume every track above a validity threshold must occupy a football position.

Review output should show rejected/unassigned tracks with track ID, reason, validity score, best football-role candidate, and optional flags.

---

## 19. Required Revised Pipeline

```text
DATA DISCOVERY
↓
LOAD DATA
    MOT
    KeyActions
    DatasetSummary
    optional PlayerTrack ground truth
↓
TRACK QUALITY SUMMARY
↓
PRELIMINARY GEOMETRY
↓
VIEW CLASSIFICATION
↓
PAIR BUILDING
↓
SNAP IDENTIFICATION
↓
STABLE PRE-SNAP FORMATION WINDOW
↓
SEMANTIC ACTION ANCHORS
↓
OFFENSIVE DIRECTION INFERENCE
↓
OFFENSE/DEFENSE SIDE EVIDENCE
↓
NORMALIZED FORMATION FEATURES
↓
VIEW-SPECIFIC ROLE PROBABILITIES
↓
PRELIMINARY PERSONNEL HYPOTHESES
↓
PAIRED-VIEW EVIDENCE FUSION
↓
FINAL GLOBAL CP-SAT ASSIGNMENT
    sideline
    endzone
↓
ACTIVE VISIBLE / ACTIVE NOT_VISIBLE / INACTIVE SLOT RESOLUTION
↓
CONFIDENCE + AMBIGUITY CALIBRATION
↓
OUTPUTS
    PlayerTrack CSV
    inference JSON
    review Markdown
```

---

## 20. Required Configuration Updates

Recommended config files:

```text
config/
├── action_role_rules.yaml
├── confidence.yaml
├── pairing.yaml
├── position_taxonomy.yaml
├── scoring_weights.yaml
└── personnel_constraints.yaml
```

Suggested V1 bounds:

```yaml
offense:
  QB: {min: 1, max: 1}
  LT: {min: 1, max: 1}
  LG: {min: 1, max: 1}
  C:  {min: 1, max: 1}
  RG: {min: 1, max: 1}
  RT: {min: 1, max: 1}
  RB: {min: 0, max: 2}
  FB: {min: 0, max: 1}
  TE: {min: 0, max: 3}
  WR: {min: 0, max: 5}

defense:
  DE: {min: 0, max: 3}
  DT: {min: 0, max: 4}
  LB: {min: 1, max: 5}
  CB: {min: 2, max: 5}
  FS: {min: 0, max: 1}
  SS: {min: 0, max: 1}
```

These are configurable solver bounds, not universal football laws.

---

## 21. Required Review Report Improvements

Add input summary, personnel hypothesis, per-assignment evidence, solver diagnostics, and explicit review triggers.

Per-assignment fields should include:

```text
slot_id
position
track_id
visibility
final confidence
action score
geometry score
learned-model score
paired-view score
track-quality score
second-best role
score margin
flags
```

Review triggers should include missing hard anchor, low view confidence, low direction confidence, pair disagreement, low assignment margin, suspected ID switch, personnel ambiguity, and too many unassigned player-like tracks.

---

## 22. Required Evaluation Metrics

Calculate at minimum:

```text
overall player-position accuracy
offense accuracy
defense accuracy
QB accuracy
Center accuracy
OL exact-position accuracy
skill-position accuracy
defensive-front accuracy
defensive-back accuracy
complete-offense-formation accuracy
complete-defense-formation accuracy
complete-video accuracy
not_visible precision
not_visible recall
high-confidence precision
coverage above confidence threshold
paired-view personnel agreement
```

Report sideline and endzone separately on the manually annotated dataset.

---

## 23. Required Reproducible `tests/fixtures/` Architecture

The corrected repository must create and use a small, deterministic fixture suite under `tests/fixtures/`.

The purpose of `tests/fixtures/` is **software correctness and regression testing**.

It is separate from the full real dataset under `data/`, which is used for:

- training,
- evaluation,
- confidence calibration,
- larger-scale benchmarking.

The test suite must not depend on the entire dataset in order to run its deterministic integration tests.

### 23.1 Required initial fixture pairs

Once the corresponding real files are available, create at least these two fixture groups:

```text
tests/
└── fixtures/
    ├── jetsweep_pair_001_002/
    │   ├── player_tracks.csv
    │   ├── key_actions.csv
    │   ├── dataset_summary.csv
    │   ├── JetSweep_1_cvat_mot.zip
    │   ├── JetSweep_2_cvat_mot.zip
    │   └── expected.json
    │
    └── power_pair_001_002/
        ├── player_tracks.csv
        ├── key_actions.csv
        ├── dataset_summary.csv
        ├── Power_1_cvat_mot.zip
        ├── Power_2_cvat_mot.zip
        └── expected.json
```

If committing the MOT ZIP fixtures would make the repository impractically large, a fixture may instead contain:

- a minimal extracted MOT subset sufficient for the test, or
- a documented environment-variable reference to a local fixture package.

However:

- no absolute user path may be hard-coded,
- the fixture must remain reproducible,
- the test must skip clearly if an optional external large fixture is not configured.

### 23.2 Fixture source-of-truth rule

Fixtures must be derived from the user's real source files.

Do not invent synthetic football positions for the golden pair when authoritative annotations exist.

For each fixture pair, Antigravity must extract only the rows/files needed for those videos from:

```text
PlayerTrack_ID_Sheet.csv
KeyActions_Sheet.csv
DatasetSummary.csv
CVAT MOT ZIP / gt.txt
```

The full source spreadsheets may remain under `data/`.

The fixture should contain only the minimal deterministic subset needed for the selected test videos.

### 23.3 Required `player_tracks.csv`

Each fixture `player_tracks.csv` must contain only the PlayerTrack ground-truth rows for the videos in that fixture.

For example:

```text
jetsweep_pair_001_002/player_tracks.csv
```

must contain only the authoritative position rows for:

```text
JetSweep_1
JetSweep_2
```

Do not copy the entire Jet Sweep annotation sheet into the fixture.

The same applies to Power.

### 23.4 Required `key_actions.csv`

Each fixture `key_actions.csv` must contain only the action annotation rows needed for the fixture videos.

This file must preserve the actual source schema.

The loader must be tested against this real reduced schema rather than an invented alternate CSV format.

### 23.5 Required `dataset_summary.csv`

Each fixture should contain the relevant DatasetSummary rows for its videos when those rows are available.

The fixture must preserve blank `view` values where the source data has blank view metadata.

Do not fill missing view metadata manually merely to make a test pass.

The purpose of the fixture is also to test the fallback view-inference logic.

### 23.6 Required MOT fixture data

Each pair must provide the real tracking geometry needed for inference.

Preferred:

```text
<PlayName>_<VideoID>_cvat_mot.zip
```

If a reduced fixture is created from `gt.txt`, preserve:

- frame numbers,
- track IDs,
- bounding boxes,
- player/ball class labels,
- visibility fields if available.

Do not renumber track IDs for test convenience.

### 23.7 Required normalized `expected.json`

Every integration fixture must contain an `expected.json` that represents the normalized expected inference result.

The raw PlayerTrack CSV remains the authoritative source annotation.

`expected.json` exists to make test assertions explicit and unambiguous.

Recommended structure:

```json
{
  "pair_id": "jetsweep_pair_001_002",
  "videos": {
    "JetSweep_1": {
      "expected_view": "sideline",
      "offense": [],
      "defense": []
    },
    "JetSweep_2": {
      "expected_view": "endzone",
      "offense": [],
      "defense": []
    }
  }
}
```

Each role entry should support:

```json
{
  "source_label": "WR1",
  "normalized_position": "WR",
  "track_state": "VISIBLE",
  "track_id": 4
}
```

or:

```json
{
  "source_label": "WR1",
  "normalized_position": "WR",
  "track_state": "NOT_VISIBLE",
  "track_id": null
}
```

or:

```json
{
  "source_label": "RB",
  "normalized_position": "RB",
  "track_state": "UNKNOWN_GROUND_TRUTH",
  "track_id": null
}
```

### 23.8 Three distinct ground-truth track states

The attached PlayerTrack annotation sheets demonstrate at least three distinct states.

The fixture loader and evaluator must preserve them separately.

#### A. Numeric track ID

Example:

```text
QB,17
```

Meaning:

```text
track_state = VISIBLE
track_id = 17
```

This is strict ground truth.

#### B. `NV`

The Power annotations include entries such as:

```text
WR1,NV
CB1,NV
C,NV
DT,NV
```

`NV` means:

```text
track_state = NOT_VISIBLE
track_id = null
```

This is authoritative ground truth that the football role is expected but has no visible track in that video.

`NV` must map to the inference concept:

```text
ACTIVE_NOT_VISIBLE
```

or equivalent.

It must not be treated as:

- unknown,
- missing annotation,
- inactive personnel,
- inference failure.

#### C. `?` / `[?]`

The Jet Sweep annotations include unresolved values such as:

```text
RB,?
DT,?
DT,[?]
```

These mean:

```text
track_state = UNKNOWN_GROUND_TRUTH
track_id = null
```

They are not equivalent to `NV`.

Unknown ground truth must normally be excluded from strict position/track accuracy assertions.

The system may still produce a prediction for that role, but the fixture must not declare the prediction wrong merely because the authoritative annotation is unresolved.

### 23.9 Required position-label normalization

The real PlayerTrack sheets contain more detailed and inconsistent source labels than the V1 inference taxonomy.

The fixture/evaluation layer must preserve:

```text
source_label
```

while also producing:

```text
normalized_position
```

for V1 evaluation.

The initial normalization table must support at least the real aliases observed in the Jet Sweep and Power PlayerTrack sheets.

#### Offense

```text
QB   -> QB
RB   -> RB
HB   -> RB or configurable backfield normalization
FB   -> FB
WR   -> WR
WR1  -> WR
WR2  -> WR
WR3  -> WR
TE   -> TE
LT   -> LT
LG   -> LG
C    -> C
RG   -> RG
RT   -> RT
```

`HB` handling must be explicitly documented.

If the project wants to distinguish HB from RB later, preserve the original source label even if V1 normalizes it to `RB`.

#### Defense

```text
DE    -> DE
DE1   -> DE
DE2   -> DE
LDE   -> DE
RDE   -> DE

DT    -> DT
DT1   -> DT
DT2   -> DT
DT-1  -> DT

LB    -> LB
LB1   -> LB
LB2   -> LB
OLB   -> LB
SLB   -> LB
MLB   -> LB
WLB   -> LB

CB    -> CB
CB1   -> CB
CB2   -> CB
CB3   -> CB
CB4   -> CB
RCB   -> CB

FS    -> FS
SS    -> SS
```

### 23.10 Ambiguous safety labels

The PlayerTrack sheets also contain:

```text
SAF
SAF1
SAF2
```

These labels do not necessarily distinguish `FS` from `SS`.

Do not silently normalize all `SAF` labels to either `FS` or `SS`.

Represent them as an evaluation-compatible ambiguous class, for example:

```text
normalized_position = SAF
allowed_predictions = [FS, SS]
```

A prediction of either `FS` or `SS` may be counted as general safety-role correct when the ground truth is only `SAF`.

Exact FS-vs-SS accuracy must be calculated only where the ground-truth source distinguishes those labels.

### 23.11 Repeated general positions must be evaluated as sets unless slot ordering is authoritative

The real annotation sheets frequently contain repeated positions:

```text
WR
CB
LB
DE
DT
TE
SAF
```

The source annotation does not always define a canonical relationship to internal slot IDs such as:

```text
offense.WR_1
offense.WR_2
offense.WR_3
```

Therefore, until an explicit slot-ordering convention is implemented, evaluation must compare repeated roles as sets.

Example:

```text
expected WR tracks = {1, 19, 21}
predicted WR tracks = {19, 21, 1}
```

This must count as correct.

Do not fail a test merely because:

```text
WR_1
WR_2
WR_3
```

were permuted.

If canonical ordering is later defined—for example offense-left-to-right—then slot-specific assertions may be added after that convention is documented and tested.

### 23.12 JetSweep fixture requirements

Use `JetSweep_1` as the strict initial full-view golden example.

The authoritative offense mapping includes:

```text
QB  -> 17
RB  -> 20
WR  -> 19
WR  -> 1
LT  -> 5
LG  -> 3
C   -> 7
RG  -> 9
RT  -> 13
TE  -> 12
WR  -> 21
```

The full defensive mapping must be derived from the fixture `player_tracks.csv`.

`JetSweep_2` includes unresolved ground-truth entries such as `?`.

Those unresolved entries must not be converted to `not_visible`.

Use `JetSweep_2` to test:

- paired-view inference,
- independent track IDs,
- partial/unresolved ground truth,
- evaluation masking for unknown labels.

### 23.13 Power fixture requirements

Use `Power_1` and `Power_2` as an explicit missing-player / endzone-style fixture pair when the corresponding real action/MOT/view files are supplied and confirm that they are paired views of the same play.

The attached PlayerTrack sheet contains explicit `NV` entries for Power Video 2, including wide receivers and cornerbacks.

This fixture must test that:

```text
NV -> ACTIVE_NOT_VISIBLE
```

and that the solver does not force unrelated visible tracks into those positions.

### 23.14 Fixture generation must be implemented as a reproducible utility

Do not make fixture creation a one-time undocumented manual process.

Add a developer utility or script, for example:

```text
scripts/build_test_fixtures.py
```

or:

```text
python -m position_inference.cli build-fixtures
```

The utility should:

1. read the full source annotation files,
2. select specified video IDs,
3. copy/extract the corresponding MOT data,
4. write reduced PlayerTrack rows,
5. write reduced KeyActions rows,
6. write reduced DatasetSummary rows,
7. normalize expected annotations into `expected.json`,
8. preserve `VISIBLE`, `NOT_VISIBLE`, and `UNKNOWN_GROUND_TRUTH`,
9. report any ambiguous/unsupported source labels.

The generated fixture should be deterministic.

Running the utility twice from unchanged source files must produce equivalent fixture content.

### 23.15 Fixture provenance

Each fixture should include provenance in `expected.json` or a small metadata file.

At minimum:

```text
source play type
source video IDs
source PlayerTrack file
source KeyActions file
source DatasetSummary file
MOT source filenames
fixture-generation version
```

Do not store user-specific absolute paths in fixture provenance.

### 23.16 Fixture tests must be independent of `data/`

After a fixture has been generated, golden integration tests must read from:

```text
tests/fixtures/
```

They must not silently fall back to:

```text
data/
```

or another repository.

This ensures that fixture tests remain:

- deterministic,
- portable,
- reproducible.

### 23.17 Full data remains separate

The full real files belong under:

```text
data/
```

Recommended:

```text
data/
├── dataset_summary/
│   └── DatasetSummary.csv
├── key_actions/
│   ├── JetSweep.csv
│   └── Power.csv
├── player_tracks/
│   ├── JetSweep.csv
│   └── Power.csv
└── tracking/
    ├── JetSweep/
    └── Power/
```

Do not use the full `data/` directory as a substitute for the deterministic fixture suite.

### 23.18 Fixture acceptance criteria

Fixture implementation is complete only when:

- `tests/fixtures/jetsweep_pair_001_002/` exists,
- `tests/fixtures/power_pair_001_002/` exists once all required Power pair inputs are supplied,
- each available fixture contains reduced source annotations,
- each available fixture contains or references reproducible MOT data,
- each fixture contains normalized expected ground truth,
- numeric IDs, `NV`, and `?` remain distinct,
- alias normalization is tested,
- ambiguous `SAF` labels are not falsely converted to FS or SS,
- repeated-role permutations do not incorrectly fail tests,
- golden tests read fixture data rather than the full dataset,
- no fixture/test contains a user-specific absolute filesystem dependency.

---

## 24. Required Test Suite Corrections

### Unit tests

Cover MOT parsing, player vs ball labels, missing boxes, KeyActions matching, action side semantics, view classification, snap detection, pre-snap window, offensive direction, team partitioning, false-positive unassigned handling, canonical slot activation, personnel constraints, CP-SAT exclusivity, OL ordering, active/inactive/not_visible, paired-view personnel fusion, pair disagreement, confidence margins, and ID-switch reporting.

### Integration tests

At minimum:

```text
JetSweep_1 sideline golden
JetSweep_2 endzone golden
JetSweep_1 + JetSweep_2 paired golden
```

Use real fixture-derived expectations.

### Regression test

Add a regression test ensuring the previous incorrect JetSweep offense mapping cannot pass as a golden result.

---

## 25. Implementation Order

### Phase A — Safety and data integrity
1. remove absolute paths
2. fix KeyActions video matching
3. add local/configurable fixtures
4. verify actual schemas

### Phase B — Pipeline ordering
5. preliminary geometry
6. view classification
7. snap/pre-snap ordering
8. offensive coordinate normalization

### Phase C — Team and semantic reasoning
9. action side semantics
10. offense/defense partitioning
11. noise/unassigned handling

### Phase D — Personnel model
12. flexible offense personnel
13. flexible defense personnel
14. active/inactive/not_visible distinction

### Phase E — True global solver
15. replace sequential offense assignment
16. replace sequential defense assignment
17. implement actual CP-SAT joint optimization
18. enforce OL ordering and hard anchors
19. use candidate evidence in objective

### Phase F — Paired-view correction
20. preliminary personnel hypotheses
21. evidence-level fusion
22. rerun final view-specific solvers
23. pair confidence/review logic

### Phase G — Confidence
24. remove fixed constants
25. score-margin/ambiguity confidence
26. conservative thresholds until calibration

### Phase H — Golden validation
27. full JetSweep_1 golden
28. full JetSweep_2 golden
29. paired golden
30. old incorrect mapping must fail

### Phase I — Learning integration
31. integrate manually annotated dataset
32. train sideline role scorer
33. train endzone role scorer
34. calibrate confidence thresholds
35. evaluate before batch inference

---

## 26. Antigravity Implementation Rules

Antigravity MUST inspect the current repository before editing, preserve working modules where possible, verify actual CSV/MOT schemas, implement corrections incrementally, run tests after each phase, report exact files modified, and surface unresolved ambiguity.

Antigravity MUST also:

- create reproducible fixture-generation tooling,
- derive fixture expected values from authoritative source annotations,
- preserve `NV` separately from `?`,
- preserve original source position labels alongside normalized V1 labels,
- avoid slot-order assertions for duplicate general roles unless an ordering convention is authoritative.

Antigravity MUST NOT:

- rewrite the entire repository
- hard-code JetSweep production IDs
- use external absolute paths
- silently use unrelated KeyActions rows
- default unknown view to sideline without evidence
- assume every action actor is offense
- assume fixed offensive or defensive personnel
- assign all tracks to positions
- force referees/false positives into defense
- use independent per-track argmax as final inference
- use sequential role picking as final inference
- claim CP-SAT exists unless a real model is built and solved
- use fixed confidence constants as probabilities
- perform direct cross-view ReID in V1
- rewrite suspected MOT ID switches

---

## 27. Definition of Done

The correction is complete only when:

- no hidden dependency on another local project remains
- no hard-coded absolute paths remain
- KeyActions mismatches fail safely
- view can remain unknown
- offense/defense semantics are correct
- offensive direction is normalized
- flexible personnel is supported
- false positives can remain unassigned
- endzone missing players can become `not_visible`
- inactive superset slots are not confused with `not_visible`
- a real CP-SAT model jointly optimizes assignments
- candidate evidence contributes to the objective
- OL ordering is constrained
- hard anchors are enforced when valid
- sequential heuristic assignment is not the final solver
- paired views fuse personnel/role evidence before final solve
- unresolved pair conflict triggers `PAIR_REVIEW_REQUIRED`
- assignment confidence is not based on arbitrary constants
- JetSweep_1 passes the full golden mapping
- paired endzone clip passes against real fixture ground truth
- unit/integration/regression tests pass
- no test relies on the user's absolute filesystem path

JetSweep_1 expected offense:

```text
WR  -> 1
LG  -> 3
LT  -> 5
C   -> 7
RG  -> 9
TE  -> 12
RT  -> 13
QB  -> 17
WR  -> 19
RB  -> 20
WR  -> 21
```

---

## 28. Required Final Implementation Report

After completing corrections, Antigravity must report:

```text
1. Files modified
2. Files added
3. Defects corrected
4. Solver architecture implemented
5. Personnel model implemented
6. Paired-view fusion behavior
7. Confidence behavior
8. JetSweep_1 golden results
9. JetSweep_2 golden results
10. Paired-view golden results
11. Unit-test results
12. Integration-test results
13. Remaining limitations
14. Any requirements not completed
```

Do not report V1 correction as complete if any critical requirement remains placeholder logic.

---

## 29. Final V1 Correction Objective

The corrected codebase must implement the intended behavior:

> Given paired, unsynchronized sideline/endzone football clips with independent MOT track IDs, independently annotated Key Actions, incomplete view metadata, noisy player/ball tracking, and limited position ground truth, infer separate complete position mappings for both videos using semantic action anchors, pre-snap geometry, flexible personnel hypotheses, view-specific role scoring, paired-view evidence fusion, and a true global constrained assignment solver—while allowing missing players, false positives, and suspected ID switches to remain explicit rather than being silently forced into incorrect assignments.

The corrected repository should not merely produce plausible-looking position files. It must produce **auditable, globally consistent, evidence-driven assignments that can be quantitatively validated against the manually annotated dataset**.
