import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from position_inference.data.schemas import VideoMetadata

logger = logging.getLogger(__name__)


def load_dataset_summary(csv_source: Union[str, Path]) -> Dict[str, VideoMetadata]:
    """
    Parses DatasetSummary.csv into a dictionary of VideoMetadata indexed by video_id.
    """
    csv_path = Path(csv_source)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset summary CSV not found: {csv_path}")

    metadata_dict: Dict[str, VideoMetadata] = {}

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader, [])]

        col_map = {h.lower(): idx for idx, h in enumerate(headers)}

        clip_idx = col_map.get("clip number", col_map.get("clip", 0))
        play_idx = col_map.get("play", None)
        view_idx = col_map.get("view", None)
        input_idx = col_map.get("input_file", None)
        output_idx = col_map.get("output_file", None)
        name_idx = col_map.get("name", None)

        for row_idx, row in enumerate(reader, start=1):
            if not row or not any(cell.strip() for cell in row):
                continue

            clip_str = row[clip_idx].strip() if clip_idx < len(row) else str(row_idx)
            try:
                order = int(clip_str)
            except ValueError:
                order = row_idx

            name_prefix = row[name_idx].strip() if name_idx is not None and name_idx < len(row) else ""
            out_file = row[output_idx].strip() if output_idx is not None and output_idx < len(row) else ""
            view_raw = row[view_idx].strip() if view_idx is not None and view_idx < len(row) else None
            play_type = row[play_idx].strip() if play_idx is not None and play_idx < len(row) else None
            in_file = row[input_idx].strip() if input_idx is not None and input_idx < len(row) else None

            if out_file:
                video_id = Path(out_file).stem
            elif name_prefix and clip_str:
                video_id = f"{name_prefix}{clip_str}"
            else:
                video_id = f"clip_{order}"

            # Normalize view_raw
            norm_view = None
            if view_raw:
                v_clean = view_raw.strip().lower()
                if v_clean in ("s", "sideline", "side") or "sideline" in v_clean:
                    norm_view = "sideline"
                elif v_clean in ("e", "endzone", "end zone", "ez") or "endzone" in v_clean:
                    norm_view = "endzone"
                else:
                    norm_view = view_raw.strip()

            extra_fields = {headers[i]: row[i].strip() for i in range(len(row)) if i < len(headers)}

            metadata_dict[video_id] = VideoMetadata(
                video_id=video_id,
                dataset_order=order,
                view_raw=norm_view,
                play_type=play_type if play_type else None,
                input_file=in_file,
                output_file=out_file,
                extra=extra_fields,
            )

    return metadata_dict


def resolve_video_metadata(
    dataset_summary_source: Optional[Union[str, Path, Dict[str, VideoMetadata]]],
    video_id: str,
) -> Optional[VideoMetadata]:
    """
    Resolves authoritative VideoMetadata from DatasetSummary for a given video_id.
    Normalizes video_id:
      - 'JetSweep_1', 'JetSweep_1.mp4', stem names
      - clip numbers e.g. '1' or 'clip_1'
      - case-insensitive exact matching
    Strictly avoids ambiguous substring matching (e.g. JetSweep_1 never matches JetSweep_10).
    If metadata is missing, logs a clear warning and returns None.
    """
    if not dataset_summary_source or not video_id:
        return None

    if isinstance(dataset_summary_source, dict):
        metadata_dict = dataset_summary_source
    else:
        path = Path(dataset_summary_source)
        if not path.exists():
            logger.warning(f"DatasetSummary source path does not exist: {path}")
            return None
        metadata_dict = load_dataset_summary(path)

    # 1. Exact key match
    if video_id in metadata_dict:
        return metadata_dict[video_id]

    target_clean = Path(video_id).stem.strip()
    target_lower = target_clean.lower()

    # 2. Case-insensitive exact match on key or output_file stem
    for key, meta in metadata_dict.items():
        if key.lower() == target_lower:
            return meta
        if meta.output_file and Path(meta.output_file).stem.lower() == target_lower:
            return meta

    # 3. Numeric clip matching if target is a number (e.g. '1' matching order=1)
    if target_clean.isdigit():
        target_num = int(target_clean)
        for meta in metadata_dict.values():
            if meta.dataset_order == target_num:
                return meta

    # 4. Strip common prefix if video_id is e.g. 'JetSweep_1' and dict has 'clip_1'
    if "_" in target_clean:
        suffix = target_clean.split("_")[-1]
        if suffix.isdigit():
            target_num = int(suffix)
            for meta in metadata_dict.values():
                # Only match if name prefix also matches or output_file contains prefix
                if meta.dataset_order == target_num:
                    if meta.output_file and Path(meta.output_file).stem.lower() == target_lower:
                        return meta
                    if meta.extra and meta.extra.get("name"):
                        prefix = meta.extra["name"].strip().rstrip("_")
                        if prefix.lower() in target_lower:
                            return meta

    logger.warning(f"Video '{video_id}' not found in DatasetSummary metadata. Falling back to geometric inference.")
    return None
