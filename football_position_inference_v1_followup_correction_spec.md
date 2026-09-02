# Football Position Inference — V1 Follow-Up Correction Specification

## 1. Purpose

This document defines the next focused correction phase for the existing **Football Position Inference** repository.

The repository has already undergone a major V1 repair. The following areas are now considered materially improved and should be preserved:

- repository-local data and fixtures
- safe KeyActions video matching
- preliminary geometry before view classification
- `unknown` camera-view support
- action offense/defense side semantics
- normalized offensive coordinate features
- a genuine OR-Tools CP-SAT assignment model
- unassigned/noise track support
- reproducible JetSweep/Power fixture generation
- a strict JetSweep_1 golden test
- regression coverage for the previous broken JetSweep_1 mapping

This follow-up specification is intentionally narrow.

Do not rewrite the codebase.

Do not revisit already-corrected areas unless necessary to implement one of the five remaining requirements below.

The authoritative files remain:

- `player_position_inference_v1_spec.md`
- `football_position_inference_v1_correction_spec_updated.md`

This specification governs the remaining follow-up corrections described here.

---

## 2. Follow-Up Correction Scope

This phase must implement exactly these five major improvements:

1. Wire `DatasetSummary.csv` metadata into normal CLI and inference/evaluation flows.
2. Make offensive and defensive personnel truly variable using configuration.
3. Implement true two-pass paired-view evidence fusion with a final re-solve.
4. Replace fixed/default confidence floors with score-margin and ambiguity-based confidence.
5. Strengthen endzone/pair golden tests and remove or regenerate stale outputs.

After these changes, the codebase should be ready for broader evaluation on the manually annotated dataset.

---

## 3. Non-Negotiable Rules

Antigravity MUST:

- preserve the current real CP-SAT architecture,
- preserve existing working data loaders,
- preserve the current fixture system,
- preserve safe KeyActions matching,
- preserve original MOT track identities,
- continue allowing noise/unassigned tracks,
- keep direct cross-view ReID out of V1,
- keep sideline and endzone track IDs independent,
- keep frame synchronization unnecessary,
- implement corrections incrementally,
- run tests after each phase.

Antigravity MUST NOT:

- hard-code JetSweep-specific production track IDs,
- hard-code a single offensive personnel package,
- hard-code a single defensive personnel package,
- perform paired fusion only after final assignments are locked,
- use arbitrary confidence floors such as `0.70`, `0.80`, `0.85`, or `0.88` as final calibrated confidence,
- weaken golden assertions to make tests pass,
- silently ignore available `DatasetSummary.csv` metadata,
- leave known-invalid demo outputs in `main`.

---

## 4. Correction 1 — Wire `DatasetSummary.csv` Into Normal Inference

### 4.1 Current problem

The repository contains `data/dataset_summary/DatasetSummary.csv`, and `load_dataset_summary()` correctly parses `view` into `VideoMetadata`.

`classify_view()` also correctly prioritizes explicit metadata when `video_metadata.view_raw` is present.

However, the normal CLI paths currently call `infer_video_positions(...)` without loading or passing DatasetSummary metadata.

Therefore, updated view annotations in `DatasetSummary.csv` do not affect ordinary:

- `infer-video`
- `infer-pair`
- `evaluate`

commands.

The current JetSweep golden test manually creates `VideoMetadata` for JetSweep_2, which hides this runtime integration gap.

### 4.2 Required correction

Add DatasetSummary support to every relevant CLI path.

Recommended CLI argument:

```text
--dataset-summary data/dataset_summary/DatasetSummary.csv
```

for:

- `infer-video`
- `infer-pair`
- `evaluate`

The argument may be optional, but when supplied it must be used.

### 4.3 Required metadata lookup flow

Implement a reusable helper such as:

```text
resolve_video_metadata(dataset_summary_path, video_id)
```

Required flow:

```text
CLI receives video_id
↓
load DatasetSummary once
↓
lookup authoritative VideoMetadata
↓
pass VideoMetadata into infer_video_positions()
↓
classify_view() uses view metadata when present
```

For paired inference:

```text
sideline_id -> sideline VideoMetadata
endzone_id -> endzone VideoMetadata
```

Do not assume the CLI argument name determines the view when authoritative metadata exists.

### 4.4 Video-ID matching

DatasetSummary lookup must normalize practical filename differences.

Support at least:

- `JetSweep_1`
- `JetSweep_1.mp4`
- output filename stem
- input filename stem

Do not use dangerous substring matching that could confuse `JetSweep_1` with `JetSweep_10` or `JetSweep_100`.

### 4.5 Missing metadata behavior

If DatasetSummary is supplied but the requested video is not found:

- do not silently use another row,
- either raise a typed metadata-not-found error or continue using geometry with a clear warning.

Recommended default:

```text
continue inference
warning = dataset_summary_video_not_found
```

If the view cannot otherwise be resolved, the result should be at least `REVIEW_RECOMMENDED`.

### 4.6 Blank view values

If the DatasetSummary row exists but `view` is blank:

```text
use geometric view inference
```

This remains valid behavior for play types whose view fields are incomplete.

### 4.7 JetSweep requirement

Because the user has populated `view` for all JetSweep rows, normal JetSweep CLI inference should use those values.

Expected metadata priority:

```text
DatasetSummary view
>
geometry fallback
```

### 4.8 Tests

Add tests proving:

- DatasetSummary loads,
- exact video metadata resolves,
- `.mp4` differences normalize safely,
- JetSweep_1 gets `sideline` from DatasetSummary,
- JetSweep_2 gets `endzone` from DatasetSummary,
- blank view falls back to geometry,
- missing video produces a clear warning or typed error,
- CLI passes metadata into inference.

---

## 5. Correction 2 — Make Offensive Personnel Truly Variable

### 5.1 Current problem

The repository now has `config/personnel_constraints.yaml` with flexible skill-position bounds.

However, the CP-SAT solver still forces a fixed offensive package equivalent to:

```text
QB
5 OL
1 RB
1 TE
3 WR
```

for every play.

This means the configuration exists but does not govern the final optimization.

### 5.2 Required correction

Remove fixed activation of:

- `offense.RB_1`
- `offense.TE_1`
- `offense.WR_1`
- `offense.WR_2`
- `offense.WR_3`

as universal requirements.

Keep only truly fixed offensive roles:

- `offense.QB_1`
- `offense.LT_1`
- `offense.LG_1`
- `offense.C_1`
- `offense.RG_1`
- `offense.RT_1`

These six positions remain active.

The remaining five offensive slots must be selected jointly from:

- RB
- FB
- TE
- WR

subject to configured bounds.

### 5.3 Load personnel constraints from configuration

The solver must read `config/personnel_constraints.yaml` rather than duplicating count limits in Python.

### 5.4 Required offensive count constraint

The solver must enforce:

```text
fixed offensive positions = 6
skill positions = 5
total offense = 11
```

Conceptually:

```text
RB_count + FB_count + TE_count + WR_count = 5
```

while respecting configured min/max bounds.

### 5.5 Hierarchical slot activation

Maintain slot activation hierarchy:

```text
WR_3 active -> WR_2 and WR_1 active
TE_2 active -> TE_1 active
RB_2 active -> RB_1 active
```

### 5.6 Personnel scoring

Do not treat every legal package equally.

Personnel hypothesis should be influenced by:

- visible track geometry,
- action-role evidence,
- learned model probabilities when available,
- paired sideline/endzone evidence,
- track validity,
- position-role score totals.

The solver must be able to represent packages such as:

```text
1 RB / 1 TE / 3 WR
2 RB / 1 TE / 2 WR
1 RB / 2 TE / 2 WR
0 RB / 1 TE / 4 WR
1 RB / 3 TE / 1 WR
```

when allowed by configuration.

### 5.7 Personnel output

Expose final active skill counts in the result:

```json
{
  "personnel_hypothesis": {
    "RB": 1,
    "FB": 0,
    "TE": 1,
    "WR": 3
  }
}
```

### 5.8 Active vs inactive slots

If the final package is:

```text
RB: 1
FB: 0
TE: 2
WR: 2
```

then inactive superset slots must be marked `INACTIVE_SLOT`, not `ACTIVE_NOT_VISIBLE`.

### 5.9 Tests

Add solver tests for at least:

- 11 personnel-like structure: 1 RB / 1 TE / 3 WR
- 12 personnel-like structure: 1 RB / 2 TE / 2 WR
- 21 personnel-like structure: 2 backs / 1 TE / 2 WR
- 10 personnel-like structure: 1 RB / 0 TE / 4 WR

The exact traditional personnel label does not need to be emitted.

---

## 6. Correction 3 — Make Defensive Personnel Truly Variable

### 6.1 Current problem

The solver currently forces a single defense equivalent to:

```text
2 DE
1 DT
3 LB
3 CB
1 FS
1 SS
```

This is not compatible with the flexible defensive constraints already defined in configuration.

### 6.2 Required correction

Remove the hard-coded universal defensive activation package.

The solver must choose a legal 11-player defensive package from:

- DE
- DT
- LB
- CB
- FS
- SS

within `config/personnel_constraints.yaml`.

### 6.3 Defensive structural priors

Use football structure:

```text
defensive front: DE + DT
second level: LB
defensive backfield: CB + FS + SS
```

The objective and/or constraints should favor plausible formation structure.

### 6.4 Config-driven constraints

Read min/max values from YAML.

Do not duplicate them in production Python.

### 6.5 Safety roles

Do not universally require both FS and SS if the configuration allows otherwise.

When exact FS/SS evidence is weak, preserve lower confidence rather than inventing certainty.

### 6.6 Example representable packages

The solver should be able to represent formations analogous to:

```text
4 DL / 3 LB / 4 DB
3 DL / 4 LB / 4 DB
4 DL / 2 LB / 5 DB
3 DL / 3 LB / 5 DB
```

using the coarse V1 labels.

### 6.7 Tests

Add tests for at least three distinct legal defensive count combinations.

Confirm:

```text
sum active defense slots == 11
```

in every case.

---

## 7. Correction 4 — Two-Pass Paired-View Evidence Fusion

### 7.1 Current problem

The repository now computes personnel hypotheses in `paired_fusion.py`.

However, shared personnel is derived after final per-view assignments already exist and is not fed back into a new CP-SAT solve.

The solver already accepts `personnel_priors`, but the main pipeline does not use paired evidence to rerun final optimization.

Therefore, paired-view fusion remains mostly post-processing.

### 7.2 Required correction

Implement true two-pass paired inference.

### 7.3 Required architecture

#### Pass 1 — independent preliminary inference

For sideline:

```text
MOT
+ KeyActions
+ DatasetSummary metadata
+ geometry
+ action semantics
↓
candidate scores
↓
preliminary CP-SAT solve
↓
preliminary personnel hypothesis
```

For endzone: same independent flow.

#### Fusion

Compare:

- sideline personnel hypothesis
- endzone personnel hypothesis
- sideline confidence
- endzone confidence
- view confidence
- visible skill-role evidence
- trench-role evidence

Create shared priors.

#### Pass 2 — final inference

Rerun:

```text
sideline CP-SAT
endzone CP-SAT
```

with paired priors.

Final outputs must come from Pass 2.

### 7.4 Required API design

Recommended options:

- create an intermediate evidence object returned by something like `infer_video_evidence(...)`, or
- expose enough state from the current pipeline to support pair re-solving.

Either is acceptable.

### 7.5 Sideline vs endzone weighting

Do not simply choose whichever `view_confidence` number is larger.

Use domain-specific evidence weighting.

Sideline is generally stronger for:

- full personnel counts
- WR visibility
- CB/DB visibility
- formation width

Endzone is generally stronger for:

- OL spacing
- DL alignment
- backfield depth
- interior box geometry

These weights should be configurable.

### 7.6 Personnel disagreement resolution

If the two views imply incompatible personnel:

- automatically resolve only if evidence/confidence margin exceeds configured threshold,
- otherwise set `PAIR_REVIEW_REQUIRED`.

### 7.7 Endzone `not_visible`

If the shared personnel hypothesis establishes three WRs and only one has a valid visible endzone track:

```text
WR_1 = ACTIVE_VISIBLE
WR_2 = ACTIVE_NOT_VISIBLE
WR_3 = ACTIVE_NOT_VISIBLE
```

Do not activate unrelated tracks merely to fill missing roles.

### 7.8 No cross-view ReID

Still prohibited:

```text
sideline track 7 == endzone track 15
```

via appearance-based ReID.

The pair shares role/personnel evidence, not track identity.

### 7.9 Required pair metadata

Expose at least:

```json
{
  "pair_id": "...",
  "preliminary_sideline_personnel": {},
  "preliminary_endzone_personnel": {},
  "shared_personnel_prior": {},
  "pair_resolution_margin": 0.0,
  "pair_warnings": []
}
```

### 7.10 Tests

Add explicit `test_jetsweep_pair_golden` proving:

- Pass 1 runs independently,
- paired personnel priors are created,
- Pass 2 re-solves both views,
- final output is from Pass 2,
- track IDs remain view-specific,
- no frame synchronization is required.

Add unit/synthetic cases for:

- sideline stronger,
- endzone stronger,
- ambiguous conflict -> `PAIR_REVIEW_REQUIRED`,
- missing endzone WRs.

---

## 8. Correction 5 — Replace Fixed Confidence Floors With Ambiguity-Based Confidence

### 8.1 Current problem

The solver still uses fallback/default confidence behavior such as:

- fallback candidate score `0.85`,
- visible confidence floor `0.70`,
- `not_visible` confidence `0.80`,
- hard-anchor confidence `0.99`.

Overall confidence remains largely based on the mean of visible assignment confidence.

These values are not calibrated probabilities.

### 8.2 Required correction

Separate:

```text
raw evidence score
assignment certainty
result confidence
```

Do not treat candidate score as calibrated confidence.

### 8.3 Required assignment diagnostics

For every assigned visible slot, compute at least:

```text
assigned_score
best_alternative_score
score_margin
```

The alternative must represent the best meaningful competing legal assignment.

Small margins must reduce confidence.

Large margins should increase confidence.

### 8.4 Solver-level ambiguity

Where feasible, derive alternative objective information by:

- temporarily forbidding the chosen assignment and re-solving, or
- using an efficient approximation based on candidate alternatives and constraint compatibility.

Do not make runtime prohibitively expensive.

### 8.5 Assignment confidence function

Create a configurable function combining:

- score margin
- anchor strength
- paired-view support
- track quality
- role-model probability
- view confidence
- direction confidence

Do not hard-code final confidence logic to JetSweep.

### 8.6 `not_visible` confidence

Do not use a universal constant for all `ACTIVE_NOT_VISIBLE` slots.

Confidence should depend on:

- shared personnel certainty,
- view type,
- visible-count evidence,
- paired-view support.

### 8.7 Overall video confidence

Do not use only the arithmetic mean of visible role confidences.

Include:

- minimum critical-role confidence,
- assignment ambiguity count,
- personnel certainty,
- pair consistency,
- view certainty,
- direction certainty,
- hard warnings,
- unresolved slots.

### 8.8 Conservative auto-accept policy

Until confidence is calibrated on the broader manually annotated dataset, preserve a configuration state such as:

```text
confidence.calibrated = false
```

When false, aggressive `AUTO_ACCEPTED` behavior should be disabled.

### 8.9 Tests

Add tests showing:

- large score margin -> higher confidence,
- small score margin -> lower confidence,
- pair support raises confidence,
- pair conflict lowers confidence,
- unknown view lowers result confidence,
- low direction confidence lowers result confidence.

---

## 9. Strengthen JetSweep_2 Endzone Golden Testing

Create a distinct:

```text
test_jetsweep_2_endzone_golden
```

using `tests/fixtures/jetsweep_pair_001_002/`.

Use `VISIBLE`, `NOT_VISIBLE`, and `UNKNOWN_GROUND_TRUTH` correctly.

For any JetSweep_2 `?` ground truth:

```text
exclude from strict track-ID accuracy
```

Do not convert it to `not_visible`.

Required assertions should include:

- `view == endzone`,
- DatasetSummary metadata used,
- known Center correct,
- known QB correct,
- known OL roles correct where authoritative,
- known visible roles correct,
- `UNKNOWN_GROUND_TRUTH` masked,
- no duplicate visible track assignment,
- active-not-visible behavior correct.

Only assert roles supported by authoritative fixture ground truth.

---

## 10. Strengthen Paired Golden Testing

Create a separate:

```text
test_jetsweep_pair_golden
```

Required assertions:

- JetSweep_1 view = sideline,
- JetSweep_2 view = endzone,
- views derived from DatasetSummary when available,
- preliminary personnel extracted for both,
- shared personnel prior created,
- Pass 2 occurs,
- final results come from Pass 2,
- sideline and endzone track IDs remain independent,
- paired fusion does not use frame synchronization,
- pair warnings surfaced,
- final shared personnel consistent.

If pair evidence is ambiguous, validate `PAIR_REVIEW_REQUIRED` rather than forcing a false resolution.

---

## 11. Power Fixture Follow-Up

The current fixture builder correctly reports missing `data/key_actions/Power.csv` rather than inventing it.

Preserve that behavior.

Once Power KeyActions become available, add:

```text
test_power_pair_not_visible
```

focused on:

```text
NV -> ACTIVE_NOT_VISIBLE
```

especially for wide receivers and cornerbacks absent from the endzone crop.

Do not block the JetSweep correction phase on missing Power KeyActions.

---

## 12. Remove or Regenerate Stale Demo Outputs

The repository still contains `output/jetsweep_pair_demo/` generated by the previous incorrect implementation.

Choose one:

### Preferred

Regenerate the directory using the corrected final pipeline after this follow-up phase.

### Acceptable

Delete the stale outputs from `main`.

If regenerated, include provenance such as:

- pipeline version,
- git commit SHA if available,
- fixture/source IDs,
- timestamp.

Do not treat demo output as authoritative ground truth.

---

## 13. CLI Changes

Recommended additions:

### `infer-video`

```text
--dataset-summary PATH
```

### `infer-pair`

```text
--dataset-summary PATH
```

The pair command must run the new two-pass paired inference.

Do not simply call single-view inference twice and mutate results afterward.

### `evaluate`

```text
--dataset-summary PATH
```

Evaluation must use the same metadata-aware inference path as production.

---

## 14. Result Schema Changes

Add fields as needed.

Recommended:

- `personnel_hypothesis`
- `preliminary_personnel_hypothesis`
- `paired_personnel_prior`
- `assignment_margin`
- `alternative_position`
- `alternative_score`
- `confidence_calibrated`
- `solver_pass`

Each assignment should ideally expose:

```json
{
  "slot_id": "offense.TE_1",
  "position": "TE",
  "track_id": 12,
  "slot_state": "ACTIVE_VISIBLE",
  "assigned_score": 0.0,
  "alternative_position": "WR",
  "alternative_score": 0.0,
  "score_margin": 0.0,
  "confidence": 0.0
}
```

---

## 15. Review Report Changes

Update review Markdown to show:

### Input metadata

- DatasetSummary path
- metadata video match
- metadata view
- view source = metadata | geometry

### Personnel

- Preliminary Sideline Personnel
- Preliminary Endzone Personnel
- Shared Paired Personnel Prior
- Final Personnel

### Assignment ambiguity

For each role:

- final role
- track
- assigned score
- alternative role
- alternative score
- margin
- confidence

### Pairing

- pair resolution margin
- pair status
- pair warnings

### Confidence

- confidence calibrated: yes/no
- auto-accept enabled: yes/no

---

## 16. Updated Implementation Order

### Phase 1 — DatasetSummary integration

1. add CLI argument
2. implement metadata resolver
3. wire metadata into single-view inference
4. wire metadata into paired inference
5. wire metadata into evaluation
6. add tests

### Phase 2 — Flexible personnel

7. remove fixed offensive skill package
8. load offense bounds from YAML
9. remove fixed defense package
10. load defense bounds from YAML
11. add active/inactive tests
12. add multiple personnel-package solver tests

### Phase 3 — Paired two-pass inference

13. expose preliminary evidence state
14. perform Pass 1 for both views
15. compute shared priors
16. perform Pass 2 for both views
17. return Pass 2 results
18. add pair diagnostics
19. add pair tests

### Phase 4 — Confidence

20. compute assignment alternatives
21. compute margins
22. implement ambiguity-based confidence
23. implement personnel/pair confidence
24. disable aggressive auto-accept while uncalibrated
25. add confidence tests

### Phase 5 — Golden validation

26. add JetSweep_2 endzone golden test
27. add explicit paired golden test
28. run JetSweep_1 golden regression
29. regenerate/remove stale demo outputs

---

## 17. Definition of Done

This follow-up correction is complete only when:

### DatasetSummary

- normal CLI inference uses `DatasetSummary.csv`,
- JetSweep view values affect real inference,
- `.mp4`/stem matching is safe,
- missing metadata produces clear behavior.

### Personnel

- offense skill counts are solver-selected,
- defense counts are solver-selected,
- YAML bounds govern production behavior,
- no fixed 1 RB / 1 TE / 3 WR requirement remains,
- no fixed defensive package remains,
- inactive slots are distinct from active-not-visible slots.

### Paired inference

- Pass 1 independent inference occurs,
- shared priors are computed,
- Pass 2 re-solves both views,
- final outputs come from Pass 2,
- no cross-view ReID exists,
- no frame synchronization is required,
- ambiguous pair conflicts trigger review.

### Confidence

- fixed assignment floors/defaults are removed as final confidence logic,
- assignment margin is calculated,
- alternative hypothesis is exposed,
- pair support affects confidence,
- uncalibrated mode remains conservative.

### Testing

- JetSweep_1 strict golden still passes,
- JetSweep_2 endzone golden exists,
- JetSweep pair golden exists,
- previous broken JetSweep regression still fails correctly,
- Power fixture behavior remains safe when KeyActions are absent,
- full unit/integration suite passes.

### Outputs

- known-invalid stale demo outputs are removed or regenerated,
- regenerated outputs identify current pipeline provenance.

---

## 18. Required Final Implementation Report

After implementing this follow-up correction, Antigravity must report:

1. Files modified
2. Files created
3. DatasetSummary integration behavior
4. JetSweep metadata lookup result
5. Offensive personnel constraints implemented
6. Defensive personnel constraints implemented
7. Example personnel packages solved
8. Pass 1 paired inference behavior
9. Shared paired prior behavior
10. Pass 2 final inference behavior
11. Confidence-margin implementation
12. Auto-accept calibration state
13. JetSweep_1 golden result
14. JetSweep_2 golden result
15. JetSweep paired golden result
16. Unit-test results
17. Integration-test results
18. Stale-output cleanup/regeneration
19. Remaining limitations
20. Any requirements not completed

Do not declare this follow-up complete if:

- personnel remains fixed,
- paired inference remains post-hoc only,
- DatasetSummary remains unused by normal CLI inference,
- confidence still relies on arbitrary floors,
- JetSweep_2 lacks meaningful endzone assertions.

---

## 19. Final Objective

After this correction, the V1 system should satisfy the following operational design:

> For a football play represented by independent sideline and endzone clips, load authoritative view metadata when available, infer flexible offensive and defensive personnel through a true global CP-SAT model, perform an independent preliminary solve for each view, fuse personnel/role evidence without requiring shared track IDs or synchronized frames, rerun the final optimization for both views using shared priors, and expose confidence based on assignment ambiguity rather than arbitrary constants.

This phase is complete when the system moves from a structurally repaired prototype to a **metadata-aware, personnel-flexible, genuinely paired, auditable V1 inference engine** suitable for broader validation on the manually annotated dataset.
