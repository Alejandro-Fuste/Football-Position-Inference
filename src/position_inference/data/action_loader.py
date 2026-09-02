import csv
from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple, Set
import re

from position_inference.data.schemas import ActionAnnotation


class ActionVideoNotFoundError(Exception):
    """Raised when a requested video_id is not found in an action annotations CSV."""
    def __init__(self, video_id: str, action_file: Union[str, Path], available_video_ids: List[str]):
        self.video_id = video_id
        self.action_file = str(action_file)
        self.available_video_ids = available_video_ids
        sample = available_video_ids[:10]
        super().__init__(
            f"Video ID '{video_id}' not found in action file '{action_file}'. "
            f"Available video IDs (sample): {sample} (total {len(available_video_ids)})"
        )


def load_action_annotations(csv_source: Union[str, Path], prefix_override: str = None) -> List[ActionAnnotation]:
    """
    Parses KeyActions CSV sheets into normalized ActionAnnotation dataclasses.
    """
    csv_path = Path(csv_source)
    if not csv_path.exists():
        raise FileNotFoundError(f"Key Actions CSV not found: {csv_path}")

    annotations: List[ActionAnnotation] = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = list(csv.reader(f))

    if len(reader) < 3:
        return annotations

    # Row 0 may specify Video Name prefix
    video_prefix = ""
    if prefix_override:
        video_prefix = prefix_override
    elif len(reader[0]) >= 2 and reader[0][0].strip().startswith("Video Name"):
        video_prefix = reader[0][1].strip()

    # Determine header row with action names
    header_idx = None
    for idx, row in enumerate(reader):
        if len(row) > 0 and row[0].strip().lower() in ("video #", "video_id", "video"):
            header_idx = idx
            break

    if header_idx is None:
        header_idx = 2 if len(reader) > 2 else 0

    action_names = reader[header_idx]

    # Process video data rows
    for r_idx in range(header_idx + 1, len(reader)):
        row = reader[r_idx]
        if not row or not row[0].strip():
            continue

        raw_video_num = row[0].strip()
        if raw_video_num.isdigit() and video_prefix:
            video_id = f"{video_prefix}{raw_video_num}"
        else:
            video_id = raw_video_num

        for col_idx in range(1, len(row)):
            if col_idx >= len(action_names):
                break
            action_name = action_names[col_idx].strip()
            cell = row[col_idx].strip()
            if not action_name or not cell or cell == "-":
                continue

            frame, actor_track_id, extra_val = _parse_action_cell(cell)
            if frame is not None or actor_track_id is not None or extra_val:
                extra_dict = {}
                if extra_val:
                    extra_dict["raw_target"] = extra_val

                annotations.append(
                    ActionAnnotation(
                        video_id=video_id,
                        action=action_name,
                        actor_track_id=actor_track_id,
                        start_frame=frame,
                        end_frame=frame,
                        source_row=r_idx + 1,
                        extra=extra_dict,
                    )
                )

    return annotations


def filter_actions_for_video(
    all_actions: List[ActionAnnotation],
    video_id: str,
    action_source: Optional[Union[str, Path]] = None,
    allow_missing_actions: bool = False,
) -> List[ActionAnnotation]:
    """
    Safely filters actions for a specific video_id.
    Matches exact video_id, or numeric suffix if prefix format differs.
    Raises ActionVideoNotFoundError if not found and allow_missing_actions is False.
    """
    if not all_actions:
        if allow_missing_actions:
            return []
        raise ActionVideoNotFoundError(video_id, action_source or "unknown", [])

    available_ids: List[str] = sorted(list({a.video_id for a in all_actions}))

    # 1. Exact match
    matched = [a for a in all_actions if a.video_id == video_id]
    if matched:
        return matched

    # 2. Match with/without common prefix (e.g. 'JetSweep_1' vs '1' or 'JetSweep_1.mp4')
    clean_vid = video_id.replace(".mp4", "").strip()
    digits_only = "".join(c for c in clean_vid.split("_")[-1] if c.isdigit())

    for av_id in available_ids:
        clean_av = av_id.replace(".mp4", "").strip()
        av_digits = "".join(c for c in clean_av.split("_")[-1] if c.isdigit())
        if clean_av == clean_vid or (digits_only and av_digits == digits_only and clean_vid.split("_")[0].lower() in clean_av.lower()):
            matched = [a for a in all_actions if a.video_id == av_id]
            if matched:
                return matched

    if allow_missing_actions:
        return []

    raise ActionVideoNotFoundError(video_id, action_source or "unknown", available_ids)


def _parse_action_cell(cell: str):
    """
    Parses cell format such as '132,7', '133,OL', '0,ALL', '192', 'OOB', etc.
    Returns tuple: (frame: Optional[int], actor_track_id: Optional[int], extra: Optional[str])
    """
    cell = cell.strip()
    if not cell or cell == "-":
        return None, None, None

    if "," in cell:
        parts = [p.strip() for p in cell.split(",")]
        frame = None
        actor_track_id = None
        extra = None

        if parts[0].isdigit():
            frame = int(parts[0])

        if len(parts) > 1:
            second_part = parts[1]
            if second_part.isdigit():
                actor_track_id = int(second_part)
            else:
                extra = second_part

        return frame, actor_track_id, extra

    # Single value in cell (e.g. frame number or status like 'TD', 'OOB', 'Tackle')
    if cell.isdigit():
        return int(cell), None, None
    else:
        return None, None, cell
