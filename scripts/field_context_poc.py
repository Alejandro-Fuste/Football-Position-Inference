#!/usr/bin/env python3
"""Field-context proof of concept for football position inference.

This script is intentionally isolated from the production inference pipeline. It tests one
question only: if we rectify an initial formation frame into metric football-field coordinates,
do the player tracks separate cleanly enough relative to the Center/LOS to support reliable
team partitioning?

The homography is estimated from manually supplied image<->field correspondences, matching the
metric-rectification workflow used in the user's prior football kinematics project. No ground
truth player positions are used to estimate the homography or classify sides.

Canonical field coordinates used by this PoC:
    field_x = longitudinal coordinate in yards (goal-line to goal-line direction)
    field_y = lateral coordinate in yards (sideline to sideline direction)

Example:
    PYTHONPATH=src python scripts/field_context_poc.py \
      --mot tests/fixtures/jetsweep_pair_001_002/JetSweep_2_cvat_mot.zip \
      --actions tests/fixtures/jetsweep_pair_001_002/key_actions.csv \
      --video-id JetSweep_2 \
      --correspondences field_correspondences_jetsweep2.json \
      --output-csv output/JetSweep_2_field_context.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from position_inference.data import load_action_annotations, load_mot_detections
from position_inference.data.action_loader import filter_actions_for_video
from position_inference.geometry.footpoints import compute_footpoint
from position_inference.geometry.presnap_window import identify_snap_frame
from position_inference.quality.track_stats import summarize_tracks
from position_inference.semantics import extract_semantic_anchors


Point = Tuple[float, float]


def _normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley-normalize 2D points for numerically stable DLT."""
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_dist = float(np.mean(np.linalg.norm(shifted, axis=1)))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalized = (T @ homogeneous.T).T
    return normalized[:, :2], T


def estimate_homography(image_points: np.ndarray, field_points: np.ndarray) -> np.ndarray:
    """Estimate image->field homography with normalized Direct Linear Transform."""
    if image_points.shape != field_points.shape or image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("image_points and field_points must both have shape (N, 2)")
    if len(image_points) < 4:
        raise ValueError("At least four non-collinear correspondences are required")

    img_n, T_img = _normalize_points(image_points.astype(np.float64))
    fld_n, T_fld = _normalize_points(field_points.astype(np.float64))

    rows: List[List[float]] = []
    for (u, v), (x, y) in zip(img_n, fld_n):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])

    A = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(A)
    H_n = vh[-1].reshape(3, 3)
    H = np.linalg.inv(T_fld) @ H_n @ T_img

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def transform_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points.astype(np.float64), np.ones(len(points))])
    warped = (H @ homogeneous.T).T
    denom = warped[:, 2:3]
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError("Homography projected one or more points to infinity")
    return warped[:, :2] / denom


def reprojection_rmse(H: np.ndarray, image_points: np.ndarray, field_points: np.ndarray) -> float:
    projected = transform_points(H, image_points)
    squared = np.sum((projected - field_points) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared)))


def load_correspondences(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    payload = json.loads(path.read_text())
    points = payload.get("points", [])
    if len(points) < 4:
        raise ValueError("Correspondence JSON must contain at least four points")

    image_points = np.asarray([p["image"] for p in points], dtype=np.float64)
    field_points = np.asarray([p["field"] for p in points], dtype=np.float64)
    return image_points, field_points, payload


def _semantic_play_type(video_id: str) -> Optional[str]:
    stem = Path(video_id).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return None


def _primary_anchor_frame(track_summaries) -> int:
    frames = [d.frame for s in track_summaries.values() if s.label == "player" for d in s.detections]
    if not frames:
        raise ValueError("No player detections found")
    return int(min(frames))


def _footpoint_at_frame(summary, frame: int) -> Optional[Point]:
    dets = [d for d in summary.detections if d.frame == frame]
    if not dets:
        return None
    fps = [compute_footpoint(d.bbox_xywh) for d in dets]
    return (float(np.median([p[0] for p in fps])), float(np.median([p[1] for p in fps])))


def _earliest_presnap_fallback(summary, snap_frame: Optional[int]) -> Tuple[Optional[Point], Optional[int]]:
    candidates = [d for d in summary.detections if snap_frame is None or d.frame <= snap_frame]
    if not candidates:
        return None, None
    first_frame = min(d.frame for d in candidates)
    # Use a tiny 3-frame window to reduce one-box noise without averaging away formation location.
    window = [d for d in candidates if first_frame <= d.frame <= first_frame + 2]
    fps = [compute_footpoint(d.bbox_xywh) for d in window]
    return (
        float(np.median([p[0] for p in fps])),
        float(np.median([p[1] for p in fps])),
    ), int(first_frame)


def classify_los_side(signed_downfield_yards: float, deadband_yards: float) -> str:
    """Classify field side relative to the Center-defined LOS.

    Positive signed distance means downfield in the offensive attack direction, i.e. the
    defensive side of the LOS. Negative means the offensive/backfield side. The deadband
    intentionally leaves tightly stacked trench players unresolved for later football-specific
    reasoning rather than forcing a wrong side.
    """
    if signed_downfield_yards > deadband_yards:
        return "DEFENSE_SIDE"
    if signed_downfield_yards < -deadband_yards:
        return "OFFENSE_SIDE"
    return "LOS_AMBIGUOUS"


def build_rows(
    track_summaries,
    H: np.ndarray,
    primary_frame: int,
    snap_frame: Optional[int],
    center_tid: int,
    qb_tid: int,
    deadband_yards: float,
) -> Tuple[List[Dict[str, object]], float, float, int]:
    anchors: Dict[int, Tuple[Point, int, str]] = {}
    for tid, summary in track_summaries.items():
        if summary.label != "player":
            continue
        fp = _footpoint_at_frame(summary, primary_frame)
        if fp is not None:
            anchors[tid] = (fp, primary_frame, "PRIMARY")
            continue
        fp, fallback_frame = _earliest_presnap_fallback(summary, snap_frame)
        if fp is not None and fallback_frame is not None:
            anchors[tid] = (fp, fallback_frame, "FALLBACK")

    if center_tid not in anchors:
        raise ValueError(f"Center track {center_tid} has no usable formation anchor")
    if qb_tid not in anchors:
        raise ValueError(f"QB track {qb_tid} has no usable formation anchor")

    tids = sorted(anchors)
    img_pts = np.asarray([anchors[t][0] for t in tids], dtype=np.float64)
    field_pts = transform_points(H, img_pts)
    field_by_tid = {tid: tuple(map(float, xy)) for tid, xy in zip(tids, field_pts)}

    center_x, center_y = field_by_tid[center_tid]
    qb_x, qb_y = field_by_tid[qb_tid]
    attack_delta = center_x - qb_x
    if abs(attack_delta) < 0.05:
        raise ValueError(
            "Center and QB have nearly identical longitudinal field_x coordinates; "
            "check the correspondence world-axis convention"
        )
    offense_sign = 1 if attack_delta > 0 else -1

    rows: List[Dict[str, object]] = []
    for tid in tids:
        field_x, field_y = field_by_tid[tid]
        anchor_fp, anchor_frame, anchor_source = anchors[tid]
        longitudinal_from_center = field_x - center_x
        lateral_from_center = field_y - center_y
        signed_downfield = offense_sign * longitudinal_from_center
        rows.append(
            {
                "track_id": tid,
                "anchor_source": anchor_source,
                "anchor_frame": anchor_frame,
                "image_foot_x": round(anchor_fp[0], 3),
                "image_foot_y": round(anchor_fp[1], 3),
                "field_x_yd": round(field_x, 4),
                "field_y_yd": round(field_y, 4),
                "longitudinal_from_center_yd": round(longitudinal_from_center, 4),
                "lateral_from_center_yd": round(lateral_from_center, 4),
                "signed_downfield_from_los_yd": round(signed_downfield, 4),
                "los_side": classify_los_side(signed_downfield, deadband_yards),
                "is_center": tid == center_tid,
                "is_qb": tid == qb_tid,
            }
        )

    return rows, center_x, center_y, offense_sign


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: Sequence[Dict[str, object]], inspect_tracks: Optional[Iterable[int]] = None) -> None:
    selected = list(rows)
    if inspect_tracks is not None:
        wanted = set(inspect_tracks)
        selected = [r for r in rows if int(r["track_id"]) in wanted]

    header = (
        f"{'tid':>4}  {'src':>8}  {'frm':>4}  {'field_x':>9}  {'field_y':>9}  "
        f"{'LOS yd':>8}  {'lat yd':>8}  side"
    )
    print(header)
    print("-" * len(header))
    for r in selected:
        print(
            f"{int(r['track_id']):>4}  {str(r['anchor_source']):>8}  {int(r['anchor_frame']):>4}  "
            f"{float(r['field_x_yd']):>9.3f}  {float(r['field_y_yd']):>9.3f}  "
            f"{float(r['signed_downfield_from_los_yd']):>8.3f}  "
            f"{float(r['lateral_from_center_yd']):>8.3f}  {r['los_side']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot", type=Path, required=True, help="CVAT MOT zip or supported MOT source")
    parser.add_argument("--actions", type=Path, required=True, help="KeyActions/action annotation source")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--correspondences", type=Path, required=True, help="Manual image<->field points JSON")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--deadband-yards", type=float, default=0.35)
    parser.add_argument(
        "--inspect-tracks",
        default=None,
        help="Comma-separated track IDs to print; all tracks are written to CSV",
    )
    args = parser.parse_args()

    image_points, field_points, corr_meta = load_correspondences(args.correspondences)
    H = estimate_homography(image_points, field_points)
    rmse = reprojection_rmse(H, image_points, field_points)

    detections = load_mot_detections(args.mot)
    summaries = summarize_tracks(detections)
    primary_frame = _primary_anchor_frame(summaries)

    all_actions = load_action_annotations(args.actions)
    actions = filter_actions_for_video(
        all_actions,
        video_id=args.video_id,
        action_source=args.actions,
        allow_missing_actions=False,
    )
    snap_frame = identify_snap_frame(actions)
    play_type = _semantic_play_type(args.video_id)
    center_tid, qb_tid, _, _ = extract_semantic_anchors(actions, play_type)

    if center_tid is None or qb_tid is None:
        raise ValueError(
            f"Could not resolve Center/QB semantic anchors for {args.video_id}: "
            f"center={center_tid}, qb={qb_tid}"
        )

    rows, center_x, center_y, offense_sign = build_rows(
        summaries,
        H,
        primary_frame,
        snap_frame,
        center_tid,
        qb_tid,
        args.deadband_yards,
    )

    print(f"video_id: {args.video_id}")
    print(f"primary formation frame: {primary_frame}")
    print(f"snap frame: {snap_frame}")
    print(f"center track: {center_tid}")
    print(f"qb track: {qb_tid}")
    print(f"center field coordinate: ({center_x:.3f}, {center_y:.3f}) yd")
    print(f"offense longitudinal sign: {'+field_x' if offense_sign > 0 else '-field_x'}")
    print(f"homography reprojection RMSE: {rmse:.4f} yd")
    if "field_type" in corr_meta:
        print(f"field type: {corr_meta['field_type']}")
    print()

    inspect = None
    if args.inspect_tracks:
        inspect = [int(v.strip()) for v in args.inspect_tracks.split(",") if v.strip()]
    _print_table(rows, inspect)

    if args.output_csv:
        write_csv(args.output_csv, rows)
        print(f"\nwrote: {args.output_csv}")


if __name__ == "__main__":
    main()
