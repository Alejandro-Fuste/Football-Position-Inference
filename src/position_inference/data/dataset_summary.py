import csv
from pathlib import Path
from typing import Dict, List, Union, Optional

from position_inference.data.schemas import VideoMetadata


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

            extra_fields = {headers[i]: row[i].strip() for i in range(len(row)) if i < len(headers)}

            metadata_dict[video_id] = VideoMetadata(
                video_id=video_id,
                dataset_order=order,
                view_raw=view_raw if view_raw else None,
                play_type=play_type if play_type else None,
                input_file=in_file,
                output_file=out_file,
                extra=extra_fields,
            )

    return metadata_dict
