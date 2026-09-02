# Position Inference Review Report — JetSweep_2

- **Pair ID:** JetSweep_1_JetSweep_2
- **Video ID:** JetSweep_2
- **View:** `endzone` (Confidence: 98.00%)
- **View Source:** `metadata`
- **Metadata Source:** `tests/fixtures/jetsweep_pair_001_002/dataset_summary.csv`
- **Offensive Direction:** `down` (Confidence: 92.00%)
- **Solver Pass:** Pass 2
- **Overall Confidence:** 26.34%
- **Status:** `PAIR_REVIEW_REQUIRED`
- **Confidence Calibrated:** `no (conservative mode)`
- **Auto-Accept Enabled:** `no`

## ⚠️ Warnings & Review Triggers
- Ambiguous personnel disagreement across paired views (diff: 0.020 < threshold 0.12). Manual review required.

## Formation Personnel
- **Preliminary Personnel (Pass 1):** `{'C': 1, 'LT': 1, 'LG': 1, 'RG': 1, 'RT': 1, 'QB': 1, 'WR': 5, 'DE': 1, 'DT': 3, 'LB': 1, 'CB': 5, 'SS': 1}`
- **Shared Paired Prior:** `{'WR': 3, 'TE': 1, 'RB': 1, 'FB': 0, 'CB': 3, 'FS': 1, 'SS': 1, 'DE': 1, 'DT': 3, 'LB': 1}`
- **Final Active Personnel:** `{'C': 1, 'LT': 1, 'LG': 1, 'RG': 1, 'RT': 1, 'QB': 1, 'RB': 1, 'TE': 1, 'WR': 3, 'DE': 1, 'DT': 3, 'LB': 1, 'CB': 4, 'FS': 1, 'SS': 1}`
- **Pair Resolution Margin:** `0.0198`

## Offense Assignments
| Slot ID | Position | Track ID | State | Assigned | Alt Pos | Alt Score | Margin | Confidence | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| `offense.C_1` | `C` | `13` | `ACTIVE_VISIBLE` | 1.00 | `TE` | 0.07 | 0.93 | 99.00% | evidence_score: 1.00, score_margin: 0.93, cpsat_objective: 10679.00 |
| `offense.LT_1` | `LT` | `not_visible` | `ACTIVE_NOT_VISIBLE` | 0.00 | `-` | 0.00 | 0.00 | 82.00% | missing_canonical_slot: 1.00, prior_supported: 0.00, sideline_confirmed_slot: 0.97 |
| `offense.LG_1` | `LG` | `not_visible` | `ACTIVE_NOT_VISIBLE` | 0.00 | `-` | 0.00 | 0.00 | 82.00% | missing_canonical_slot: 1.00, prior_supported: 0.00, sideline_confirmed_slot: 0.97 |
| `offense.RG_1` | `RG` | `not_visible` | `ACTIVE_NOT_VISIBLE` | 0.00 | `-` | 0.00 | 0.00 | 82.00% | missing_canonical_slot: 1.00, prior_supported: 0.00, sideline_confirmed_slot: 0.97 |
| `offense.RT_1` | `RT` | `9` | `ACTIVE_VISIBLE` | 1.00 | `RG` | 0.40 | 0.60 | 96.00% | evidence_score: 1.00, score_margin: 0.60, cpsat_objective: 10679.00 |
| `offense.QB_1` | `QB` | `11` | `ACTIVE_VISIBLE` | 1.00 | `TE` | 0.13 | 0.87 | 99.00% | evidence_score: 1.00, score_margin: 0.87, cpsat_objective: 10679.00 |
| `offense.RB_1` | `RB` | `not_visible` | `ACTIVE_NOT_VISIBLE` | 0.00 | `-` | 0.00 | 0.00 | 82.00% | missing_canonical_slot: 1.00, prior_supported: 1.00, sideline_confirmed_slot: 0.96 |
| `offense.TE_1` | `TE` | `not_visible` | `ACTIVE_NOT_VISIBLE` | 0.00 | `-` | 0.00 | 0.00 | 82.00% | missing_canonical_slot: 1.00, prior_supported: 1.00, sideline_confirmed_slot: 0.65 |
| `offense.WR_1` | `WR` | `5` | `ACTIVE_VISIBLE` | 0.95 | `TE` | 0.55 | 0.39 | 96.00% | evidence_score: 0.95, score_margin: 0.39, cpsat_objective: 10679.00 |
| `offense.WR_2` | `WR` | `15` | `ACTIVE_VISIBLE` | 0.46 | `TE` | 0.95 | 0.00 | 43.65% | evidence_score: 0.46, score_margin: 0.00, cpsat_objective: 10679.00 |
| `offense.WR_3` | `WR` | `17` | `ACTIVE_VISIBLE` | 0.54 | `RB` | 0.95 | 0.00 | 51.73% | evidence_score: 0.54, score_margin: 0.00, cpsat_objective: 10679.00 |

## Defense Assignments
| Slot ID | Position | Track ID | State | Assigned | Alt Pos | Alt Score | Margin | Confidence | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| `defense.DE_1` | `DE` | `7` | `ACTIVE_VISIBLE` | 1.00 | `DT` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 10679.00 |
| `defense.DT_1` | `DT` | `12` | `ACTIVE_VISIBLE` | 0.15 | `CB` | 0.15 | 0.00 | 40.00% | evidence_score: 0.15, score_margin: 0.00, cpsat_objective: 10679.00 |
| `defense.DT_2` | `DT` | `14` | `ACTIVE_VISIBLE` | 0.15 | `CB` | 0.15 | 0.00 | 40.00% | evidence_score: 0.15, score_margin: 0.00, cpsat_objective: 10679.00 |
| `defense.DT_3` | `DT` | `8` | `ACTIVE_VISIBLE` | 0.15 | `CB` | 0.15 | 0.00 | 40.00% | evidence_score: 0.15, score_margin: 0.00, cpsat_objective: 10679.00 |
| `defense.LB_1` | `LB` | `4` | `ACTIVE_VISIBLE` | 1.00 | `CB` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 10679.00 |
| `defense.CB_1` | `CB` | `2` | `ACTIVE_VISIBLE` | 1.00 | `DT` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 10679.00 |
| `defense.CB_2` | `CB` | `3` | `ACTIVE_VISIBLE` | 1.00 | `DT` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 10679.00 |
| `defense.CB_3` | `CB` | `1` | `ACTIVE_VISIBLE` | 1.00 | `DT` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 10679.00 |
| `defense.CB_4` | `CB` | `6` | `ACTIVE_VISIBLE` | 1.00 | `DT` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 10679.00 |
| `defense.FS_1` | `FS` | `not_visible` | `ACTIVE_NOT_VISIBLE` | 0.00 | `-` | 0.00 | 0.00 | 82.00% | missing_canonical_slot: 1.00, prior_supported: 1.00, sideline_confirmed_slot: 0.96 |
| `defense.SS_1` | `SS` | `16` | `ACTIVE_VISIBLE` | 0.10 | `DT` | 0.15 | 0.00 | 40.00% | evidence_score: 0.10, score_margin: 0.00, cpsat_objective: 10679.00 |

## Inactive Package Slots (14)
- `offense.RB_2, offense.FB_1, offense.TE_2, offense.TE_3, offense.WR_4, offense.WR_5, defense.DE_2, defense.DE_3, defense.DT_4, defense.LB_2, defense.LB_3, defense.LB_4, defense.LB_5, defense.CB_5`

## Out of View / `not_visible` Slots (6)
- `offense.LT_1` (LT): {'missing_canonical_slot': 1.0, 'prior_supported': 0.0, 'sideline_confirmed_slot': 0.975}
- `offense.LG_1` (LG): {'missing_canonical_slot': 1.0, 'prior_supported': 0.0, 'sideline_confirmed_slot': 0.975}
- `offense.RG_1` (RG): {'missing_canonical_slot': 1.0, 'prior_supported': 0.0, 'sideline_confirmed_slot': 0.975}
- `offense.RB_1` (RB): {'missing_canonical_slot': 1.0, 'prior_supported': 1.0, 'sideline_confirmed_slot': 0.96}
- `offense.TE_1` (TE): {'missing_canonical_slot': 1.0, 'prior_supported': 1.0, 'sideline_confirmed_slot': 0.6522727272727272}
- `defense.FS_1` (FS): {'missing_canonical_slot': 1.0, 'prior_supported': 1.0, 'sideline_confirmed_slot': 0.96}

## Rejected / Noise Tracks (0)
- None

## Suspected ID Switches (0)
- None detected
