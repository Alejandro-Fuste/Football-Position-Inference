from pathlib import Path
from typing import List, Union

from position_inference.data.schemas import ViewInferenceResult


def write_review_report_markdown(
    result: ViewInferenceResult,
    output_path: Union[str, Path],
    pair_id: str = "pair_0001",
):
    """
    Generates an auditable Markdown human-review report with:
    - Input metadata and view provenance
    - Personnel hypotheses (preliminary, shared prior, final)
    - Assignment ambiguity, competing alternatives, and score margins
    - Pairing diagnostics & calibration status
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    lines.append(f"# Position Inference Review Report — {result.video_id}")
    lines.append("")
    lines.append(f"- **Pair ID:** {pair_id}")
    lines.append(f"- **Video ID:** {result.video_id}")
    lines.append(f"- **View:** `{result.view}` (Confidence: {result.view_confidence:.2%})")
    lines.append(f"- **View Source:** `{'metadata' if result.metadata_source else 'geometric_inference'}`")
    if result.metadata_source:
        lines.append(f"- **Metadata Source:** `{result.metadata_source}`")
    lines.append(f"- **Offensive Direction:** `{result.offense_direction}` (Confidence: {result.offense_direction_confidence:.2%})")
    lines.append(f"- **Solver Pass:** Pass {result.solver_pass}")
    lines.append(f"- **Overall Confidence:** {result.confidence:.2%}")
    lines.append(f"- **Status:** `{result.status}`")
    lines.append(f"- **Confidence Calibrated:** `{'yes' if result.confidence_calibrated else 'no (conservative mode)'}`")
    lines.append(f"- **Auto-Accept Enabled:** `{'yes' if result.confidence_calibrated else 'no'}`")
    lines.append("")

    # Warnings section
    if result.warnings:
        lines.append("## ⚠️ Warnings & Review Triggers")
        for w in result.warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Personnel section
    lines.append("## Formation Personnel")
    if result.preliminary_personnel_hypothesis:
        lines.append(f"- **Preliminary Personnel (Pass 1):** `{result.preliminary_personnel_hypothesis}`")
    if result.paired_personnel_prior:
        lines.append(f"- **Shared Paired Prior:** `{result.paired_personnel_prior}`")
    lines.append(f"- **Final Active Personnel:** `{result.personnel_hypothesis}`")
    if result.pair_resolution_margin > 0.0:
        lines.append(f"- **Pair Resolution Margin:** `{result.pair_resolution_margin:.4f}`")
    lines.append("")

    # Offense assignments
    lines.append("## Offense Assignments")
    lines.append("| Slot ID | Position | Track ID | State | Assigned | Alt Pos | Alt Score | Margin | Confidence | Evidence |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for a in result.assignments:
        if a.side == "offense" and a.slot_state != "INACTIVE_SLOT":
            ev_str = ", ".join([f"{k}: {v:.2f}" for k, v in a.evidence.items()])
            alt_pos_str = a.alternative_position or "-"
            lines.append(
                f"| `{a.slot_id}` | `{a.position}` | `{a.track_id_display}` | `{a.slot_state}` | "
                f"{a.assigned_score:.2f} | `{alt_pos_str}` | {a.best_alternative_score:.2f} | "
                f"{a.score_margin:.2f} | {a.confidence:.2%} | {ev_str} |"
            )
    lines.append("")

    # Defense assignments
    lines.append("## Defense Assignments")
    lines.append("| Slot ID | Position | Track ID | State | Assigned | Alt Pos | Alt Score | Margin | Confidence | Evidence |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for a in result.assignments:
        if a.side == "defense" and a.slot_state != "INACTIVE_SLOT":
            ev_str = ", ".join([f"{k}: {v:.2f}" for k, v in a.evidence.items()])
            alt_pos_str = a.alternative_position or "-"
            lines.append(
                f"| `{a.slot_id}` | `{a.position}` | `{a.track_id_display}` | `{a.slot_state}` | "
                f"{a.assigned_score:.2f} | `{alt_pos_str}` | {a.best_alternative_score:.2f} | "
                f"{a.score_margin:.2f} | {a.confidence:.2%} | {ev_str} |"
            )
    lines.append("")

    # Inactive slots
    inactive_slots = [a.slot_id for a in result.assignments if a.slot_state == "INACTIVE_SLOT"]
    if inactive_slots:
        lines.append(f"## Inactive Package Slots ({len(inactive_slots)})")
        lines.append(f"- `{', '.join(inactive_slots)}`")
        lines.append("")

    # Not Visible slots
    not_vis = [a for a in result.assignments if a.slot_state == "ACTIVE_NOT_VISIBLE"]
    lines.append(f"## Out of View / `not_visible` Slots ({len(not_vis)})")
    if not_vis:
        for a in not_vis:
            lines.append(f"- `{a.slot_id}` ({a.position}): {a.evidence}")
    else:
        lines.append("- None (All formation players visible)")
    lines.append("")

    # Rejected tracks
    lines.append(f"## Rejected / Noise Tracks ({len(result.rejected_track_ids)})")
    if result.rejected_track_ids:
        lines.append(f"- Track IDs: `{result.rejected_track_ids}`")
    else:
        lines.append("- None")
    lines.append("")

    # Suspected ID switches
    lines.append(f"## Suspected ID Switches ({len(result.suspected_id_switches)})")
    if result.suspected_id_switches:
        for sw in result.suspected_id_switches:
            lines.append(f"- Track `{sw.get('track_id')}` at Frame `{sw.get('frame')}`: {sw.get('reason')}")
    else:
        lines.append("- None detected")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
