# Football Position Inference V1

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

An automated American Football player-position inference engine designed for paired sideline and endzone play clips. The system reconstructs 22-player formation roles from noisy Multi-Object Tracking (MOT) trajectories, sparse action annotations, pre-snap geometry, and football constraints.

---

## 1. Purpose & Core Principles

In large-scale football film datasets (~4,000 video clips), manual player-position annotation is a significant bottleneck. This codebase automates position assignment for unannotated play clips, producing output compatible with downstream `PlayerTrack` annotation workflows.

Rather than treating position inference as an independent 22-class visual classifier for each track, **V1 uses a hybrid structured-inference architecture**:

1. **Actions Anchor Identities**: Definitions such as `Ball Snap` $\rightarrow$ Center and `Snap Receive` $\rightarrow$ QB act as hard semantic anchors.
2. **Pre-Snap Geometry Defines Formation Structure**: Uses robust median footpoints $(x + w/2, y + h)$ across stable pre-snap frames before motion.
3. **Football Constraints Govern Legal Assignments**: Enforces 5-OL line sequence (`LT`, `LG`, `C`, `RG`, `RT`), QB placement, skill-position alignment, and defense level partitions (Front $\rightarrow$ LB $\rightarrow$ DB).
4. **Cross-View Fusion at the Evidence Level**: Fuses sideline and endzone view evidence at the personnel/canonical slot level without requiring shared track IDs or frame synchronization.
5. **Explicit `not_visible` Representation**: Players outside cropped camera views (e.g. endzone receivers/DBs) are assigned to canonical slots with `track_id=None` (`not_visible`).
6. **Transparent Confidence & Review Reporting**: Automatically accepts high-confidence assignments ($\ge 0.90$) while generating Markdown review reports for ambiguous or conflicting cases.

---

## 2. Directory Structure

```text
Football-Position-Inference/
├── pyproject.toml               # Package dependencies & CLI entrypoints
├── setup.py                     # Setuptools configuration
├── README.md                    # Project documentation
│
├── config/                      # Configuration YAML files
│   ├── position_taxonomy.yaml   # Offense/Defense position taxonomy & aliases
│   ├── action_role_rules.yaml   # Action semantic aliases & role probability weights
│   ├── scoring_weights.yaml     # Weights for action, geometry, model, and paired evidence
│   ├── pairing.yaml             # View sequence order & pair confidence thresholds
│   └── confidence.yaml          # Auto-accept (0.90) and review thresholds
│
├── src/
│   └── position_inference/
│       ├── __init__.py
│       ├── cli.py               # Main CLI interface (inspect, infer-video, infer-pair, evaluate)
│       ├── config.py            # Configuration loader
│       ├── pipeline.py          # Core end-to-end position inference pipeline
│       │
│       ├── data/                # Data loaders & schemas
│       │   ├── schemas.py       # Pydantic & Dataclass domain objects
│       │   ├── mot_loader.py    # CVAT MOT ZIP & gt.txt parser (player vs ball)
│       │   ├── action_loader.py # KeyActions spreadsheet loader
│       │   ├── playertrack_loader.py # Ground truth PlayerTrack loader
│       │   ├── dataset_summary.py   # DatasetSummary.csv parser
│       │   └── discovery.py     # File & artifact discovery
│       │
│       ├── pairing/             # View classification & play pairing
│       │   ├── view_classifier.py   # Sideline vs Endzone prediction
│       │   ├── pair_builder.py      # Play clip pairing heuristic
│       │   └── pair_confidence.py   # Pair validity confidence
│       │
│       ├── quality/             # MOT auditing & track quality
│       │   ├── track_stats.py       # Trajectory statistics & coverage
│       │   ├── player_validity.py   # Non-player false positive filter
│       │   └── id_switch_detector.py # Suspected ID-switch detector
│       │
│       ├── geometry/            # Spatial features & direction
│       │   ├── footpoints.py        # Bottom-center footpoint calculation
│       │   ├── presnap_window.py    # Pre-snap stable window extraction
│       │   ├── spatial_features.py  # Center-relative normalized coordinates
│       │   ├── direction.py         # View-relative offensive direction inference
│       │   └── team_partition.py    # Offense vs Defense seed partition
│       │
│       ├── semantics/           # Action rules & personnel
│       │   ├── action_rules.py      # Action rule matcher
│       │   ├── action_anchors.py    # Hard & soft action semantic anchors
│       │   ├── personnel.py         # Canonical slot definitions
│       │   └── formation_rules.py   # Legal formation rules
│       │
│       ├── learning/            # Tabular role models
│       │   ├── feature_matrix.py    # Feature vector construction
│       │   ├── role_model.py        # View-specific tabular role classifiers
│       │   ├── calibration.py       # Probability calibration
│       │   └── model_io.py          # Model serialization
│       │
│       ├── inference/           # Solvers & confidence
│       │   ├── candidate_scores.py  # Evidence synthesis
│       │   ├── offense_solver.py    # Hierarchical Offense solver
│       │   ├── defense_solver.py    # Hierarchical Defense solver
│       │   ├── assignment_solver.py # Global CP-SAT constrained solver
│       │   ├── paired_fusion.py     # Cross-view evidence fusion
│       │   ├── missing_slots.py     # not_visible slot completion
│       │   └── confidence.py        # Confidence calibration & review states
│       │
│       ├── output/              # Output generators
│       │   ├── playertrack_writer.py # PlayerTrack CSV format writer
│       │   ├── json_writer.py        # Detailed sidecar JSON writer
│       │   └── review_writer.py      # Markdown human-review report writer
│       │
│       └── evaluation/          # Evaluation metrics
│           └── metrics.py       # Accuracy & calibration metrics
│
└── tests/                       # Test suite
    ├── integration/
    │   └── test_jetsweep_1_golden.py # End-to-end JetSweep_1 integration fixture
    └── unit/                    # Modular unit tests
```

---

## 3. Position Taxonomy

Every resolved role is assigned a **canonical slot ID** (e.g. `offense.WR_1`) while preserving the general position label:

### Offense Taxonomy
- `C` (Center) $\rightarrow$ `offense.C_1`
- `LT` (Left Tackle) $\rightarrow$ `offense.LT_1`
- `LG` (Left Guard) $\rightarrow$ `offense.LG_1`
- `RG` (Right Guard) $\rightarrow$ `offense.RG_1`
- `RT` (Right Tackle) $\rightarrow$ `offense.RT_1`
- `QB` (Quarterback) $\rightarrow$ `offense.QB_1`
- `RB` (Running Back) $\rightarrow$ `offense.RB_1`
- `FB` (Fullback) $\rightarrow$ `offense.FB_1`
- `TE` (Tight End) $\rightarrow$ `offense.TE_1`, `offense.TE_2`
- `WR` (Wide Receiver) $\rightarrow$ `offense.WR_1`, `offense.WR_2`, `offense.WR_3`

### Defense Taxonomy
- `DE` (Defensive End) $\rightarrow$ `defense.DE_1`, `defense.DE_2`
- `DT` (Defensive Tackle) $\rightarrow$ `defense.DT_1`, `defense.DT_2`
- `LB` (Linebacker) $\rightarrow$ `defense.LB_1`, `defense.LB_2`, `defense.LB_3`
- `CB` (Cornerback) $\rightarrow$ `defense.CB_1`, `defense.CB_2`
- `FS` (Free Safety) $\rightarrow$ `defense.FS_1`
- `SS` (Strong Safety) $\rightarrow$ `defense.SS_1`

---

## 4. Installation

1. Create a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies and the `position-inference` package in editable mode:
   ```bash
   pip install wheel setuptools
   pip install --no-build-isolation -e .
   ```

---

## 5. How to Run the Code

### 5.1 Inspect Input Schemas & Artifacts
Inspect a MOT ZIP, Key Actions CSV, or PlayerTrack ground-truth CSV:
```bash
python -m position_inference.cli inspect \
  --mot data/tracking/JetSweep/JetSweep_1_cvat_mot.zip \
  --actions data/key_actions/JetSweep.csv \
  --playertracks data/player_tracks/JetSweep.csv
```

### 5.2 Single-Video Inference
Infer positions for a single video clip:
```bash
python -m position_inference.cli infer-video \
  --mot data/tracking/JetSweep/JetSweep_1_cvat_mot.zip \
  --actions data/key_actions/JetSweep.csv \
  --video-id JetSweep_1 \
  --output-dir output/jetsweep_1_run
```

### 5.3 Paired Sideline / Endzone View Inference
Infer positions jointly for paired camera views:
```bash
python -m position_inference.cli infer-pair \
  --sideline-mot data/tracking/JetSweep/JetSweep_1_cvat_mot.zip \
  --endzone-mot data/tracking/JetSweep/JetSweep_2_cvat_mot.zip \
  --actions data/key_actions/JetSweep.csv \
  --sideline-id JetSweep_1 \
  --endzone-id JetSweep_2 \
  --output-dir output/jetsweep_pair_run
```

### 5.4 Evaluate Predictions Against Ground Truth
Compute accuracy metrics against ground truth:
```bash
python -m position_inference.cli evaluate \
  --mot data/tracking/JetSweep/JetSweep_1_cvat_mot.zip \
  --actions data/key_actions/JetSweep.csv \
  --playertracks data/player_tracks/JetSweep.csv \
  --video-id JetSweep_1
```

---

## 6. Generated Outputs

For every processed clip, the pipeline writes three complementary artifacts:

1. **PlayerTrack CSV** (`*_playertrack.csv`):
   Downstream-compatible CSV spreadsheet containing position and track ID pairings.
2. **Machine-Readable Sidecar JSON** (`*_inference.json`):
   Detailed JSON containing slot IDs, visibility status, confidence scores, decomposed evidence breakdown (`action_semantics`, `geometry`, `learned_model`, `paired_view`), alternative hypotheses, and warnings.
3. **Markdown Review Report** (`*_review.md`):
   Auditable human-review report showing assignment tables, out-of-view `not_visible` slots, rejected noise tracks, suspected ID switches, and confidence review status (`AUTO_ACCEPTED`, `REVIEW_RECOMMENDED`, `PAIR_REVIEW_REQUIRED`, `HUMAN_REQUIRED`).

---

## 7. Running Tests

Run the complete pytest test suite:
```bash
pytest -v
```

Run only the golden integration test:
```bash
pytest tests/integration/test_jetsweep_1_golden.py -v
```
