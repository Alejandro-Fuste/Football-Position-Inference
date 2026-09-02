from pathlib import Path
from typing import Dict, List, Optional, Tuple

from position_inference.data.schemas import VideoMetadata


def discover_video_artifacts(data_dir: Path, video_id: str) -> Dict[str, Optional[Path]]:
    """
    Discovers available MOT zip, actions CSV, and player_track ground truth for a single video_id.
    """
    data_dir = Path(data_dir)
    results = {
        "mot": None,
        "actions": None,
        "playertracks": None,
    }

    # Search for MOT zip
    mot_candidates = list(data_dir.rglob(f"{video_id}_cvat_mot.zip")) + list(data_dir.rglob(f"{video_id}.zip"))
    if mot_candidates:
        results["mot"] = mot_candidates[0]

    # Search for actions CSV
    play_prefix = video_id.split("_")[0] if "_" in video_id else video_id
    actions_candidates = list(data_dir.rglob(f"key_actions/{play_prefix}.csv")) + list(data_dir.rglob(f"*{play_prefix}*action*.csv"))
    if actions_candidates:
        results["actions"] = actions_candidates[0]

    # Search for player tracks ground truth CSV
    track_candidates = list(data_dir.rglob(f"player_tracks/{play_prefix}.csv")) + list(data_dir.rglob(f"*{play_prefix}*track*.csv"))
    if track_candidates:
        results["playertracks"] = track_candidates[0]

    return results
