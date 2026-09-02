import csv
from pathlib import Path
from typing import List, Union

from position_inference.data.schemas import PositionAssignment, ViewInferenceResult


def write_playertrack_csv(
    result: ViewInferenceResult,
    output_path: Union[str, Path],
    video_number: str = "1",
    video_name_prefix: str = "JetSweep_",
):
    """
    Writes a CSV compatible with downstream PlayerTrack annotation sheets.
    Format:
    Row 0: ['Video Name:', video_name_prefix, ...]
    Row 1: ['Video #', 'Positions & Player Track ID', ...]
    Row 2 (Offense): [video_number, 'QB,17', 'RB,20', ...]
    Row 3 (Defense): [video_number, 'FS,2', 'CB,16', ...]
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    offense_assignments = [a for a in result.assignments if a.side == "offense" and a.slot_state != "INACTIVE_SLOT"]
    defense_assignments = [a for a in result.assignments if a.side == "defense" and a.slot_state != "INACTIVE_SLOT"]

    off_cells = [f"{a.position},{a.track_id_display}" for a in offense_assignments]
    def_cells = [f"{a.position},{a.track_id_display}" for a in defense_assignments]

    row0 = ["Video Name:", video_name_prefix] + [""] * 10
    row1 = ["Video #", "Positions & Player Track ID"] + [""] * 10
    row_off = [video_number] + off_cells
    row_def = [video_number] + def_cells

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row0)
        writer.writerow(row1)
        writer.writerow(row_off)
        writer.writerow(row_def)
