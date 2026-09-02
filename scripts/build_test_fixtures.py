#!/usr/bin/env python3
"""
scripts/build_test_fixtures.py

Reproducible test-fixture generation utility.
Extracts minimal deterministic test fixtures from authoritative local source data under data/
into tests/fixtures/jetsweep_pair_001_002/ and tests/fixtures/power_pair_001_002/.

Does NOT contain hardcoded absolute user paths.
Preserves ground-truth track states: VISIBLE, NOT_VISIBLE (NV), UNKNOWN_GROUND_TRUTH (?).
"""

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path so we can import loaders
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from position_inference.data.playertrack_loader import load_ground_truth_roles
from position_inference.data.schemas import GroundTruthRole


def extract_csv_rows_by_video_ids(
    src_csv: Path,
    dest_csv: Path,
    target_video_nums: List[str],
    video_id_prefix: str,
):
    """
    Extracts header and only rows corresponding to target video numbers/IDs.
    Preserves original formatting.
    """
    if not src_csv.exists():
        return False

    with open(src_csv, "r", encoding="utf-8-sig") as f:
        reader = list(csv.reader(f))

    if not reader:
        return False

    # Find header row
    header_idx = 0
    for idx, row in enumerate(reader):
        if len(row) > 0 and row[0].strip().lower() in ("video #", "video_id", "video", "clip number"):
            header_idx = idx
            break

    out_rows = reader[: header_idx + 1]

    curr_vid = None
    keep_row = False
    for r_idx in range(header_idx + 1, len(reader)):
        row = reader[r_idx]
        if not row:
            continue
        first = row[0].strip()
        if first:
            curr_vid = first
            # Match number or full name
            clean_num = curr_vid.replace(video_id_prefix, "").strip()
            keep_row = (curr_vid in target_video_nums) or (clean_num in target_video_nums)

        if keep_row:
            out_rows.append(row)

    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    return True


def extract_dataset_summary_rows(
    src_csv: Path,
    dest_csv: Path,
    target_filenames: List[str],
):
    if not src_csv.exists():
        return False

    with open(src_csv, "r", encoding="utf-8-sig") as f:
        reader = list(csv.reader(f))

    if not reader:
        return False

    header = reader[0]
    out_rows = [header]

    # Find output_file or name column index
    out_file_idx = 7
    for idx, col in enumerate(header):
        if "output_file" in col.lower() or "output" in col.lower():
            out_file_idx = idx
            break

    for row in reader[1:]:
        if len(row) > out_file_idx:
            val = row[out_file_idx].strip()
            for target in target_filenames:
                if target.lower() in val.lower():
                    out_rows.append(row)
                    break

    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    return True


def build_expected_json(
    roles: List[GroundTruthRole],
    video_ids: List[str],
    views: Dict[str, str],
    pair_id: str,
    provenance: Dict[str, str],
) -> Dict:
    data = {
        "pair_id": pair_id,
        "provenance": provenance,
        "videos": {},
    }

    for vid in video_ids:
        vid_roles = [r for r in roles if r.video_id == vid]
        offense = []
        defense = []

        for r in vid_roles:
            entry = {
                "source_label": r.source_label,
                "normalized_position": r.position,
                "track_state": r.track_state,
                "track_id": r.track_id,
            }
            if r.allowed_predictions:
                entry["allowed_predictions"] = r.allowed_predictions

            if r.side == "offense":
                offense.append(entry)
            else:
                defense.append(entry)

        data["videos"][vid] = {
            "expected_view": views.get(vid, "unknown"),
            "offense": offense,
            "defense": defense,
        }

    return data


def build_jetsweep_fixture(data_root: Path, fixtures_root: Path):
    dest_dir = fixtures_root / "jetsweep_pair_001_002"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding JetSweep fixture -> {dest_dir}...")

    # 1. PlayerTrack
    pt_src = data_root / "player_tracks" / "JetSweep.csv"
    pt_dst = dest_dir / "player_tracks.csv"
    extract_csv_rows_by_video_ids(pt_src, pt_dst, ["1", "2"], "JetSweep_")

    # 2. KeyActions
    ka_src = data_root / "key_actions" / "JetSweep.csv"
    ka_dst = dest_dir / "key_actions.csv"
    extract_csv_rows_by_video_ids(ka_src, ka_dst, ["1", "2"], "JetSweep_")

    # 3. DatasetSummary
    ds_src = data_root / "dataset_summary" / "DatasetSummary.csv"
    ds_dst = dest_dir / "dataset_summary.csv"
    extract_dataset_summary_rows(ds_src, ds_dst, ["JetSweep_1", "JetSweep_2"])

    # 4. MOT files
    mot1_src = data_root / "tracking" / "JetSweep" / "JetSweep_1_cvat_mot.zip"
    mot2_src = data_root / "tracking" / "JetSweep" / "JetSweep_2_cvat_mot.zip"
    if mot1_src.exists():
        shutil.copy2(mot1_src, dest_dir / "JetSweep_1_cvat_mot.zip")
    if mot2_src.exists():
        shutil.copy2(mot2_src, dest_dir / "JetSweep_2_cvat_mot.zip")

    # 5. Expected JSON
    roles = load_ground_truth_roles(pt_dst)
    expected_data = build_expected_json(
        roles=roles,
        video_ids=["JetSweep_1", "JetSweep_2"],
        views={"JetSweep_1": "sideline", "JetSweep_2": "endzone"},
        pair_id="jetsweep_pair_001_002",
        provenance={
            "play_type": "JetSweep",
            "source_videos": ["JetSweep_1", "JetSweep_2"],
            "mot_files": ["JetSweep_1_cvat_mot.zip", "JetSweep_2_cvat_mot.zip"],
            "version": "1.0",
        },
    )

    with open(dest_dir / "expected.json", "w", encoding="utf-8") as f:
        json.dump(expected_data, f, indent=2)

    print(f"  ✓ JetSweep fixture created with {len(roles)} ground-truth entries.")


def build_power_fixture(data_root: Path, fixtures_root: Path):
    dest_dir = fixtures_root / "power_pair_001_002"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding Power fixture -> {dest_dir}...")

    # 1. PlayerTrack
    pt_src = data_root / "player_tracks" / "Power.csv"
    pt_dst = dest_dir / "player_tracks.csv"
    extract_csv_rows_by_video_ids(pt_src, pt_dst, ["1", "2"], "Power_")

    # 2. KeyActions
    ka_src = data_root / "key_actions" / "Power.csv"
    ka_dst = dest_dir / "key_actions.csv"
    if ka_src.exists():
        extract_csv_rows_by_video_ids(ka_src, ka_dst, ["1", "2"], "Power_")
        print("  ✓ Extracted Power key actions.")
    else:
        print("  [NOTE] data/key_actions/Power.csv is missing. Reporting missing real input.")

    # 3. DatasetSummary
    ds_src = data_root / "dataset_summary" / "DatasetSummary.csv"
    ds_dst = dest_dir / "dataset_summary.csv"
    extract_dataset_summary_rows(ds_src, ds_dst, ["Power_1", "Power_2"])

    # 4. MOT files
    mot1_src = data_root / "tracking" / "Power" / "Power_1_cvat_mot.zip"
    mot2_src = data_root / "tracking" / "Power" / "Power_2_cvat_mot.zip"
    if mot1_src.exists():
        shutil.copy2(mot1_src, dest_dir / "Power_1_cvat_mot.zip")
    if mot2_src.exists():
        shutil.copy2(mot2_src, dest_dir / "Power_2_cvat_mot.zip")

    # 5. Expected JSON
    roles = load_ground_truth_roles(pt_dst)
    expected_data = build_expected_json(
        roles=roles,
        video_ids=["Power_1", "Power_2"],
        views={"Power_1": "sideline", "Power_2": "endzone"},
        pair_id="power_pair_001_002",
        provenance={
            "play_type": "Power",
            "source_videos": ["Power_1", "Power_2"],
            "mot_files": ["Power_1_cvat_mot.zip", "Power_2_cvat_mot.zip"],
            "missing_inputs": ["key_actions/Power.csv"] if not ka_src.exists() else [],
            "version": "1.0",
        },
    )

    with open(dest_dir / "expected.json", "w", encoding="utf-8") as f:
        json.dump(expected_data, f, indent=2)

    print(f"  ✓ Power fixture created with {len(roles)} ground-truth entries.")
    nv_count = sum(1 for r in roles if r.track_state == "NOT_VISIBLE")
    print(f"  ✓ Preserved {nv_count} NOT_VISIBLE (NV) roles.")


def main():
    data_root = REPO_ROOT / "data"
    fixtures_root = REPO_ROOT / "tests" / "fixtures"

    print("=" * 60)
    print("Building test fixtures from authoritative local source data...")
    print(f"Source data: {data_root}")
    print(f"Fixtures destination: {fixtures_root}")
    print("=" * 60)

    build_jetsweep_fixture(data_root, fixtures_root)
    build_power_fixture(data_root, fixtures_root)

    print("\nFixture build complete!")


if __name__ == "__main__":
    main()
