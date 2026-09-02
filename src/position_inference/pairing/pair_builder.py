from typing import Dict, List, Optional, Tuple

from position_inference.data.schemas import VideoMetadata


def infer_pairs_from_dataset(
    metadata_list: List[VideoMetadata],
) -> List[Tuple[VideoMetadata, Optional[VideoMetadata], float, str]]:
    """
    Pairs adjacent sideline and endzone clips using dataset order, view metadata, and play stems.
    Returns list of tuples: (sideline_metadata, endzone_metadata, pair_confidence, pair_status).
    """
    # Sort metadata by dataset_order
    sorted_meta = sorted(metadata_list, key=lambda m: m.dataset_order)

    pairs = []
    i = 0
    while i < len(sorted_meta):
        curr = sorted_meta[i]
        next_meta = sorted_meta[i + 1] if i + 1 < len(sorted_meta) else None

        if next_meta:
            # Check if curr is Sideline and next_meta is Endzone
            curr_view = (curr.view_raw or "").strip().lower()
            next_view = (next_meta.view_raw or "").strip().lower()

            # Check matching play stem if present (e.g. JetSweep_1 and JetSweep_2, or play name)
            curr_stem = curr.video_id.rsplit("_", 1)[0] if "_" in curr.video_id else curr.video_id
            next_stem = next_meta.video_id.rsplit("_", 1)[0] if "_" in next_meta.video_id else next_meta.video_id

            if curr_view == "sideline" and next_view == "endzone" and curr_stem == next_stem:
                pairs.append((curr, next_meta, 0.98, "PAIR_CONFIRMED_METADATA"))
                i += 2
                continue
            elif curr_stem == next_stem and i + 1 == curr.dataset_order:
                pairs.append((curr, next_meta, 0.85, "PAIR_INFERRED_HIGH_CONFIDENCE"))
                i += 2
                continue

        # Unpaired clip
        pairs.append((curr, None, 0.0, "UNPAIRED"))
        i += 1

    return pairs
