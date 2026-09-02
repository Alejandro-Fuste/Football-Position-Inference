import csv
from pathlib import Path
from typing import List, Union, Dict, Optional

from position_inference.config import get_position_taxonomy
from position_inference.data.schemas import GroundTruthRole


def load_ground_truth_roles(csv_source: Union[str, Path], prefix_override: str = None) -> List[GroundTruthRole]:
    """
    Parses PlayerTrack ground-truth CSV sheets into normalized GroundTruthRole objects.
    """
    csv_path = Path(csv_source)
    if not csv_path.exists():
        raise FileNotFoundError(f"PlayerTrack CSV not found: {csv_path}")

    taxonomy = get_position_taxonomy()
    aliases = taxonomy.get("position_aliases", {})

    roles: List[GroundTruthRole] = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = list(csv.reader(f))

    if len(reader) < 3:
        return roles

    video_prefix = ""
    if prefix_override:
        video_prefix = prefix_override
    elif len(reader[0]) >= 2 and reader[0][0].strip().startswith("Video Name"):
        video_prefix = reader[0][1].strip()

    header_idx = None
    for idx, row in enumerate(reader):
        if len(row) > 0 and row[0].strip().lower() in ("video #", "video_id", "video"):
            header_idx = idx
            break

    if header_idx is None:
        header_idx = 1 if len(reader) > 1 else 0

    curr_video_num = None
    row_count_for_video = 0

    for r_idx in range(header_idx + 1, len(reader)):
        row = reader[r_idx]
        if not row:
            continue

        first_cell = row[0].strip()
        if first_cell:
            curr_video_num = first_cell
            row_count_for_video = 0
        else:
            row_count_for_video += 1

        if not curr_video_num:
            continue

        if curr_video_num.isdigit() and video_prefix:
            video_id = f"{video_prefix}{curr_video_num}"
        else:
            video_id = curr_video_num

        for col_idx in range(1, len(row)):
            cell = row[col_idx].strip()
            if not cell:
                continue

            pos, track_id = _parse_position_cell(cell, aliases)
            if pos:

                side = _determine_side(pos, taxonomy, row_count_for_video)

                roles.append(
                    GroundTruthRole(
                        video_id=video_id,
                        side=side,
                        position=pos,
                        track_id=track_id,
                        source_row=r_idx + 1,
                    )
                )

        if first_cell:
            row_count_for_video = 1

    return roles


def _parse_position_cell(cell: str, aliases: Dict[str, str]):
    cell = cell.strip()
    if not cell:
        return None, None

    if "," in cell:
        parts = [p.strip() for p in cell.split(",", 1)]
        raw_pos = parts[0]
        raw_track = parts[1]
    else:

        raw_pos = cell
        raw_track = ""

    norm_pos = aliases.get(raw_pos, raw_pos)

    track_id: Optional[int] = None
    clean_track = raw_track.replace("[", "").replace("]", "").strip()
    if clean_track.isdigit():
        track_id = int(clean_track)

    return norm_pos, track_id


def _determine_side(position: str, taxonomy: Dict, row_index: int) -> str:
    offense_positions = set(taxonomy.get("offense", {}).get("positions", []))
    defense_positions = set(taxonomy.get("defense", {}).get("positions", []))

    if position in offense_positions:
        return "offense"
    if position in defense_positions:
        return "defense"

    return "offense" if row_index == 0 else "defense"
