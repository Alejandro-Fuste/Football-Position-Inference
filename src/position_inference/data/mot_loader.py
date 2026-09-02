from pathlib import Path
from typing import List, Union
import zipfile

from position_inference.data.schemas import MotDetection


def load_mot_detections(mot_source: Union[str, Path]) -> List[MotDetection]:
    """
    Loads MOT tracking detections from a CVAT MOT zip archive or a directory/gt.txt file.
    Supports player vs ball label separation via gt/labels.txt.
    """
    mot_path = Path(mot_source)
    if not mot_path.exists():
        raise FileNotFoundError(f"MOT file not found: {mot_path}")

    if mot_path.is_file() and mot_path.suffix.lower() == ".zip":
        return _load_from_zip(mot_path)
    elif mot_path.is_file() and mot_path.name == "gt.txt":
        return _load_from_gt_file(mot_path, mot_path.parent / "labels.txt")
    elif mot_path.is_dir():
        gt_file = mot_path / "gt" / "gt.txt"
        if not gt_file.exists():
            gt_file = mot_path / "gt.txt"
        labels_file = mot_path / "gt" / "labels.txt"
        if not labels_file.exists():
            labels_file = mot_path / "labels.txt"
        return _load_from_gt_file(gt_file, labels_file)
    else:
        raise ValueError(f"Unsupported MOT path: {mot_path}")


def _load_from_zip(zip_path: Path) -> List[MotDetection]:
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        gt_name = next((n for n in names if n.endswith("gt.txt")), None)
        if not gt_name:
            raise ValueError(f"No gt.txt found in MOT zip: {zip_path}")

        labels_name = next((n for n in names if n.endswith("labels.txt")), None)
        label_map = {1: "player", 2: "ball"}
        if labels_name:
            labels_content = z.read(labels_name).decode("utf-8").splitlines()
            labels_list = [l.strip().lower() for l in labels_content if l.strip()]
            for idx, lbl in enumerate(labels_list, start=1):
                if "ball" in lbl:
                    label_map[idx] = "ball"
                else:
                    label_map[idx] = "player"

        gt_content = z.read(gt_name).decode("utf-8")
        return _parse_gt_lines(gt_content.splitlines(), label_map)


def _load_from_gt_file(gt_path: Path, labels_path: Path) -> List[MotDetection]:
    label_map = {1: "player", 2: "ball"}
    if labels_path.exists():
        labels_list = [l.strip().lower() for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for idx, lbl in enumerate(labels_list, start=1):
            if "ball" in lbl:
                label_map[idx] = "ball"
            else:
                label_map[idx] = "player"

    gt_lines = gt_path.read_text(encoding="utf-8").splitlines()
    return _parse_gt_lines(gt_lines, label_map)


def _parse_gt_lines(lines: List[str], label_map: dict) -> List[MotDetection]:
    detections: List[MotDetection] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            conf = float(parts[6]) if len(parts) > 6 and parts[6] != "" else 1.0
            class_id = int(float(parts[7])) if len(parts) > 7 and parts[7] != "" else 1
            vis = float(parts[8]) if len(parts) > 8 and parts[8] != "" else 1.0

            label = label_map.get(class_id, "player")

            detections.append(
                MotDetection(
                    frame=frame,
                    track_id=track_id,
                    label=label,
                    bbox_xywh=(x, y, w, h),
                    confidence=conf,
                    visibility=vis,
                )
            )
        except ValueError:
            continue

    return detections
