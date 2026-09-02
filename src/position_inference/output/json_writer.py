import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

from position_inference.data.schemas import ViewInferenceResult


def write_inference_json(
    result: ViewInferenceResult,
    output_path: Union[str, Path],
    pair_id: str = "pair_0001",
    pair_confidence: float = 1.0,
):
    """
    Writes detailed machine-readable sidecar JSON output for a video inference result.
    Includes personnel hypothesis, assignment score margins, alternatives, and calibration state.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    assignments_json = []
    for a in result.assignments:
        assignments_json.append(
            {
                "slot_id": a.slot_id,
                "side": a.side,
                "position": a.position,
                "track_id": a.track_id,
                "track_id_display": a.track_id_display,
                "visibility": a.visibility,
                "slot_state": getattr(a, "slot_state", "ACTIVE_VISIBLE"),
                "confidence": float(round(a.confidence, 4)),
                "assigned_score": float(round(getattr(a, "assigned_score", 0.0), 4)),
                "best_alternative_score": float(round(getattr(a, "best_alternative_score", 0.0), 4)),
                "score_margin": float(round(getattr(a, "score_margin", 0.0), 4)),
                "alternative_position": getattr(a, "alternative_position", None),
                "evidence": {k: float(round(v, 4)) for k, v in a.evidence.items()},
                "alternatives": a.alternatives,
                "flags": a.flags,
            }
        )

    out_data = {
        "schema_version": "1.1",
        "video_id": result.video_id,
        "view": result.view,
        "view_confidence": float(round(result.view_confidence, 4)),
        "pair_id": pair_id,
        "pair_confidence": float(round(pair_confidence, 4)),
        "offense_direction": result.offense_direction,
        "offense_direction_confidence": float(round(result.offense_direction_confidence, 4)),
        "personnel_hypothesis": result.personnel_hypothesis,
        "preliminary_personnel_hypothesis": result.preliminary_personnel_hypothesis,
        "paired_personnel_prior": result.paired_personnel_prior,
        "confidence_calibrated": result.confidence_calibrated,
        "solver_pass": result.solver_pass,
        "confidence": float(round(result.confidence, 4)),
        "assignments": assignments_json,
        "rejected_tracks": result.rejected_track_ids,
        "suspected_id_switches": result.suspected_id_switches,
        "warnings": result.warnings,
        "status": result.status,
        "metadata_source": result.metadata_source,
        "pair_resolution_margin": float(round(result.pair_resolution_margin, 4)),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
