# Position Inference Review Report — JetSweep_1

- **Pair ID:** JetSweep_1_JetSweep_2
- **Video ID:** JetSweep_1
- **View:** `sideline` (Confidence: 98.00%)
- **View Source:** `metadata`
- **Metadata Source:** `tests/fixtures/jetsweep_pair_001_002/dataset_summary.csv`
- **Offensive Direction:** `left` (Confidence: 92.00%)
- **Solver Pass:** Pass 2
- **Overall Confidence:** 23.94%
- **Status:** `PAIR_REVIEW_REQUIRED`
- **Confidence Calibrated:** `no (conservative mode)`
- **Auto-Accept Enabled:** `no`

## ⚠️ Warnings & Review Triggers
- Ambiguous personnel disagreement across paired views (diff: 0.020 < threshold 0.12). Manual review required.

## Formation Personnel
- **Preliminary Personnel (Pass 1):** `{'C': 1, 'LT': 1, 'LG': 1, 'RG': 1, 'RT': 1, 'QB': 1, 'RB': 1, 'TE': 1, 'WR': 3, 'DE': 2, 'DT': 1, 'LB': 3, 'CB': 3, 'FS': 1, 'SS': 1}`
- **Shared Paired Prior:** `{'WR': 3, 'TE': 1, 'RB': 1, 'FB': 0, 'CB': 3, 'FS': 1, 'SS': 1, 'DE': 1, 'DT': 3, 'LB': 1}`
- **Final Active Personnel:** `{'C': 1, 'LT': 1, 'LG': 1, 'RG': 1, 'RT': 1, 'QB': 1, 'RB': 1, 'TE': 1, 'WR': 3, 'DE': 2, 'DT': 1, 'LB': 3, 'CB': 3, 'FS': 1, 'SS': 1}`
- **Pair Resolution Margin:** `0.0198`

## Offense Assignments
| Slot ID | Position | Track ID | State | Assigned | Alt Pos | Alt Score | Margin | Confidence | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| `offense.C_1` | `C` | `7` | `ACTIVE_VISIBLE` | 1.00 | `TE` | 0.07 | 0.93 | 99.00% | evidence_score: 1.00, score_margin: 0.93, cpsat_objective: 21026.00 |
| `offense.LT_1` | `LT` | `5` | `ACTIVE_VISIBLE` | 1.00 | `FB` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 21026.00 |
| `offense.LG_1` | `LG` | `3` | `ACTIVE_VISIBLE` | 1.00 | `LT` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 21026.00 |
| `offense.RG_1` | `RG` | `9` | `ACTIVE_VISIBLE` | 1.00 | `RT` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 21026.00 |
| `offense.RT_1` | `RT` | `13` | `ACTIVE_VISIBLE` | 0.45 | `TE` | 1.00 | 0.00 | 42.95% | evidence_score: 0.45, score_margin: 0.00, cpsat_objective: 21026.00 |
| `offense.QB_1` | `QB` | `17` | `ACTIVE_VISIBLE` | 1.00 | `FB` | 0.50 | 0.50 | 99.00% | evidence_score: 1.00, score_margin: 0.50, cpsat_objective: 21026.00 |
| `offense.RB_1` | `RB` | `20` | `ACTIVE_VISIBLE` | 1.00 | `QB` | 0.50 | 0.50 | 96.00% | evidence_score: 1.00, score_margin: 0.50, cpsat_objective: 21026.00 |
| `offense.TE_1` | `TE` | `12` | `ACTIVE_VISIBLE` | 0.68 | `RT` | 0.95 | 0.00 | 65.23% | evidence_score: 0.68, score_margin: 0.00, cpsat_objective: 21026.00 |
| `offense.WR_1` | `WR` | `21` | `ACTIVE_VISIBLE` | 1.00 | `TE` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 21026.00 |
| `offense.WR_2` | `WR` | `1` | `ACTIVE_VISIBLE` | 1.00 | `TE` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 21026.00 |
| `offense.WR_3` | `WR` | `19` | `ACTIVE_VISIBLE` | 0.95 | `RB` | 1.00 | 0.00 | 92.05% | evidence_score: 0.95, score_margin: 0.00, cpsat_objective: 21026.00 |

## Defense Assignments
| Slot ID | Position | Track ID | State | Assigned | Alt Pos | Alt Score | Margin | Confidence | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| `defense.DE_1` | `DE` | `8` | `ACTIVE_VISIBLE` | 1.00 | `DT` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 21026.00 |
| `defense.DE_2` | `DE` | `6` | `ACTIVE_VISIBLE` | 1.00 | `DT` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 21026.00 |
| `defense.DT_1` | `DT` | `4` | `ACTIVE_VISIBLE` | 1.00 | `CB` | 0.15 | 0.85 | 96.00% | evidence_score: 1.00, score_margin: 0.85, cpsat_objective: 21026.00 |
| `defense.LB_1` | `LB` | `18` | `ACTIVE_VISIBLE` | 1.00 | `DT` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 21026.00 |
| `defense.LB_2` | `LB` | `15` | `ACTIVE_VISIBLE` | 1.00 | `SS` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 21026.00 |
| `defense.LB_3` | `LB` | `10` | `ACTIVE_VISIBLE` | 1.00 | `DT` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 21026.00 |
| `defense.CB_1` | `CB` | `22` | `ACTIVE_VISIBLE` | 0.40 | `SS` | 1.00 | 0.00 | 40.00% | evidence_score: 0.40, score_margin: 0.00, cpsat_objective: 21026.00 |
| `defense.CB_2` | `CB` | `16` | `ACTIVE_VISIBLE` | 1.00 | `DT` | 0.40 | 0.60 | 96.00% | evidence_score: 1.00, score_margin: 0.60, cpsat_objective: 21026.00 |
| `defense.CB_3` | `CB` | `14` | `ACTIVE_VISIBLE` | 0.40 | `SS` | 1.00 | 0.00 | 40.00% | evidence_score: 0.40, score_margin: 0.00, cpsat_objective: 21026.00 |
| `defense.FS_1` | `FS` | `2` | `ACTIVE_VISIBLE` | 1.00 | `CB` | 0.40 | 0.60 | 96.00% | evidence_score: 1.00, score_margin: 0.60, cpsat_objective: 21026.00 |
| `defense.SS_1` | `SS` | `11` | `ACTIVE_VISIBLE` | 1.00 | `FS` | 1.00 | 0.00 | 97.50% | evidence_score: 1.00, score_margin: 0.00, cpsat_objective: 21026.00 |

## Inactive Package Slots (14)
- `offense.RB_2, offense.FB_1, offense.TE_2, offense.TE_3, offense.WR_4, offense.WR_5, defense.DE_3, defense.DT_2, defense.DT_3, defense.DT_4, defense.LB_4, defense.LB_5, defense.CB_4, defense.CB_5`

## Out of View / `not_visible` Slots (0)
- None (All formation players visible)

## Rejected / Noise Tracks (1)
- Track IDs: `[29]`

## Suspected ID Switches (0)
- None detected
