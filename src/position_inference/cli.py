import argparse
import json
from pathlib import Path
import sys
from typing import Optional

from position_inference.data import (
    load_action_annotations,
    load_ground_truth_roles,
    load_mot_detections,
    resolve_video_metadata,
)
from position_inference.evaluation import evaluate_predictions
from position_inference.inference import fuse_paired_views_two_pass
from position_inference.output import (
    write_inference_json,
    write_playertrack_csv,
    write_review_report_markdown,
)
from position_inference.pipeline import infer_video_positions


def main():
    parser = argparse.ArgumentParser(prog="position-inference", description="Football Position Inference System V1 CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect input files and schemas")
    inspect_parser.add_argument("--mot", type=str, help="Path to CVAT MOT zip")
    inspect_parser.add_argument("--actions", type=str, help="Path to Key Actions CSV")
    inspect_parser.add_argument("--playertracks", type=str, help="Path to PlayerTrack CSV")

    # Command: infer-video
    infer_parser = subparsers.add_parser("infer-video", help="Infer player positions for a single video clip")
    infer_parser.add_argument("--mot", type=str, required=True, help="Path to MOT zip")
    infer_parser.add_argument("--actions", type=str, help="Path to Key Actions CSV")
    infer_parser.add_argument("--dataset-summary", type=str, default=None, help="Path to DatasetSummary.csv")
    infer_parser.add_argument("--video-id", type=str, default="video_1", help="Video identifier")
    infer_parser.add_argument("--output-dir", type=str, default="output", help="Output directory")

    # Command: infer-pair
    pair_parser = subparsers.add_parser("infer-pair", help="Infer player positions jointly for paired sideline and endzone clips")
    pair_parser.add_argument("--sideline-mot", type=str, required=True, help="Path to sideline MOT zip")
    pair_parser.add_argument("--endzone-mot", type=str, required=True, help="Path to endzone MOT zip")
    pair_parser.add_argument("--actions", type=str, help="Path to Key Actions CSV")
    pair_parser.add_argument("--dataset-summary", type=str, default=None, help="Path to DatasetSummary.csv")
    pair_parser.add_argument("--sideline-id", type=str, default="JetSweep_1", help="Sideline video ID")
    pair_parser.add_argument("--endzone-id", type=str, default="JetSweep_2", help="Endzone video ID")
    pair_parser.add_argument("--pair-id", type=str, default=None, help="Pair identifier")
    pair_parser.add_argument("--output-dir", type=str, default="output", help="Output directory")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions against ground truth")
    eval_parser.add_argument("--mot", type=str, required=True, help="Path to MOT zip")
    eval_parser.add_argument("--actions", type=str, help="Path to Key Actions CSV")
    eval_parser.add_argument("--dataset-summary", type=str, default=None, help="Path to DatasetSummary.csv")
    eval_parser.add_argument("--playertracks", type=str, required=True, help="Path to PlayerTrack CSV")
    eval_parser.add_argument("--video-id", type=str, default="JetSweep_1", help="Video ID to evaluate")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "inspect":
        print("=== Football Position Inference V1 Inspector ===")
        if args.mot:
            dets = load_mot_detections(args.mot)
            print(f"MOT detections loaded: {len(dets)}")
        if args.actions:
            acts = load_action_annotations(args.actions)
            print(f"Key Action annotations loaded: {len(acts)}")
        if args.playertracks:
            gt = load_ground_truth_roles(args.playertracks)
            print(f"Ground Truth role annotations loaded: {len(gt)}")

    elif args.command == "infer-video":
        out_dir = Path(args.output_dir)
        result = infer_video_positions(
            args.mot,
            args.actions,
            video_id=args.video_id,
            dataset_summary=args.dataset_summary,
        )
        write_playertrack_csv(result, out_dir / f"{args.video_id}_playertrack.csv", video_number=args.video_id)
        write_inference_json(result, out_dir / f"{args.video_id}_inference.json")
        write_review_report_markdown(result, out_dir / f"{args.video_id}_review.md")
        print(f"Inference complete for {args.video_id}. Status: {result.status}. Outputs written to {out_dir}")

    elif args.command == "infer-pair":
        out_dir = Path(args.output_dir)
        pair_name = args.pair_id or f"{args.sideline_id}_{args.endzone_id}"

        s_fused, e_fused, pair_summary = fuse_paired_views_two_pass(
            sideline_mot=args.sideline_mot,
            endzone_mot=args.endzone_mot,
            action_source=args.actions,
            sideline_id=args.sideline_id,
            endzone_id=args.endzone_id,
            dataset_summary=args.dataset_summary,
            pair_id=pair_name,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        write_playertrack_csv(s_fused, out_dir / f"{args.sideline_id}_playertrack.csv", video_number=args.sideline_id)
        write_inference_json(s_fused, out_dir / f"{args.sideline_id}_inference.json", pair_id=pair_name)
        write_review_report_markdown(s_fused, out_dir / f"{args.sideline_id}_review.md", pair_id=pair_name)

        write_playertrack_csv(e_fused, out_dir / f"{args.endzone_id}_playertrack.csv", video_number=args.endzone_id)
        write_inference_json(e_fused, out_dir / f"{args.endzone_id}_inference.json", pair_id=pair_name)
        write_review_report_markdown(e_fused, out_dir / f"{args.endzone_id}_review.md", pair_id=pair_name)

        with open(out_dir / "pair_summary.json", "w", encoding="utf-8") as f:
            json.dump(pair_summary, f, indent=2)

        print(f"Two-pass paired inference complete for {args.sideline_id} and {args.endzone_id}. Outputs written to {out_dir}")

    elif args.command == "evaluate":
        result = infer_video_positions(
            args.mot,
            args.actions,
            video_id=args.video_id,
            dataset_summary=args.dataset_summary,
        )
        gt_all = load_ground_truth_roles(args.playertracks)
        gt_video = [g for g in gt_all if g.video_id == args.video_id or g.video_id.endswith(args.video_id)]
        metrics = evaluate_predictions(gt_video, result.assignments)
        print(f"=== Evaluation Results for {args.video_id} ===")
        for k, v in metrics.items():
            if "accuracy" in k or "precision" in k:
                print(f"  {k}: {v:.2%}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
