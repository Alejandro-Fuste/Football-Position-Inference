from pathlib import Path
from typing import List, Union

from position_inference.data.schemas import ViewInferenceResult


def write_review_report_markdown(
    result: ViewInferenceResult,
    output_path: Union[str, Path],
    pair_id: str = "pair_0001",
):
    """
    Generates a Markdown human-review report with auditable evidence sections.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    lines.append(f"# Position Inference Review Report — {result.video_id}")
    lines.append("")
    lines.append(f"- **Pair ID:** {pair_id}")
    lines.append(f"- **Video ID:** {result.video_id}")
    lines.append(f"- **View:** {result.view} (Confidence: {result.view_confidence:.2%})")
    lines.append(f"- **Offensive Direction:** {result.offense_direction} (Confidence: {result.offense_direction_confidence:.2%})")
    lines.append(f"- **Overall Confidence:** {result.confidence:.2%}")
    lines.append(f"- **Status:** `{result.status}`")
    lines.append("")

    # Warnings section
    if result.warnings:
        lines.append("## ⚠️ Warnings & Review Triggers")
        for w in result.warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Offense assignments
    lines.append("## Offense Assignments")
    lines.append("| Slot ID | Position | Track ID | Visibility | Confidence | Evidence Breakdown |")
    lines.append("|---|---|---|---|---|---|")
    for a in result.assignments:
        if a.side == "offense":
            ev_str = ", ".join([f"{k}: {v:.2f}" for k, v in a.evidence.items()])
            lines.append(f"| `{a.slot_id}` | `{a.position}` | `{a.track_id_display}` | `{a.visibility}` | {a.confidence:.2%} | {ev_str} |")
    lines.append("")

    # Defense assignments
    lines.append("## Defense Assignments")
    lines.append("| Slot ID | Position | Track ID | Visibility | Confidence | Evidence Breakdown |")
    lines.append("|---|---|---|---|---|---|")
    for a in result.assignments:
        if a.side == "defense":
            ev_str = ", ".join([f"{k}: {v:.2f}" for k, v in a.evidence.items()])
            lines.append(f"| `{a.slot_id}` | `{a.position}` | `{a.track_id_display}` | `{a.visibility}` | {a.confidence:.2%} | {ev_str} |")
    lines.append("")

    # Not Visible slots
    not_vis = [a for a in result.assignments if a.visibility == "out_of_view"]
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
