import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from position_inference.pipeline import infer_video_positions
from position_inference.inference import fuse_paired_views
from position_inference.data.schemas import VideoMetadata
from position_inference.output import (
    write_playertrack_csv,
    write_inference_json,
    write_review_report_markdown,
)

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "jetsweep_pair_003_004"
mot3 = FIXTURE_DIR / "JetSweep_3_cvat_mot.zip"
mot4 = FIXTURE_DIR / "JetSweep_4_cvat_mot.zip"
actions = FIXTURE_DIR / "key_actions.csv"
out_dir = REPO_ROOT / "output" / "jetsweep_pair_2"

print("--- Running Sideline JetSweep_3 ---", flush=True)
s_meta = VideoMetadata(video_id="JetSweep_3", dataset_order=3, view_raw="sideline")
s_res = infer_video_positions(mot3, actions, video_id="JetSweep_3", video_metadata=s_meta)
print(f"JetSweep_3 complete! Status: {s_res.status}, Assigned: {len(s_res.assignments)}", flush=True)

print("--- Running Endzone JetSweep_4 ---", flush=True)
e_meta = VideoMetadata(video_id="JetSweep_4", dataset_order=4, view_raw="endzone")
e_res = infer_video_positions(mot4, actions, video_id="JetSweep_4", video_metadata=e_meta)
print(f"JetSweep_4 complete! Status: {e_res.status}, Assigned: {len(e_res.assignments)}", flush=True)

print("--- Fusing paired views ---", flush=True)
s_fused, e_fused, warnings = fuse_paired_views(s_res, e_res)

out_dir.mkdir(parents=True, exist_ok=True)
pair_name = "jetsweep_pair_003_004"
write_playertrack_csv(s_fused, out_dir / "JetSweep_3_playertrack.csv", video_number="JetSweep_3")
write_inference_json(s_fused, out_dir / "JetSweep_3_inference.json")
write_review_report_markdown(s_fused, out_dir / "JetSweep_3_review.md", pair_id=pair_name)

write_playertrack_csv(e_fused, out_dir / "JetSweep_4_playertrack.csv", video_number="JetSweep_4")
write_inference_json(e_fused, out_dir / "JetSweep_4_inference.json")
write_review_report_markdown(e_fused, out_dir / "JetSweep_4_review.md", pair_id=pair_name)

print(f"Done! Outputs written to {out_dir}", flush=True)
