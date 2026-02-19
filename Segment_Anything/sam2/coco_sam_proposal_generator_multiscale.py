#!/usr/bin/env python3
"""
SAM-2.1 proposal generation for COCO / RefCOCO(+)-style datasets.

What this script does
---------------------
- Loads a JSON list of visual grounding samples (expects each sample to include an `image_id`).
- Finds unique COCO image IDs and loads the corresponding images from a COCO directory
  (default filename pattern: COCO_train2014_{image_id:012d}.jpg).
- Runs SAM-2.1 automatic mask generation to obtain mask proposals.
- Converts each mask into a tight bounding box, filters out invalid/too-small boxes,
  and returns a final list of proposals per image.
- Supports BOTH:
    (1) single-scale proposal generation, and
    (2) multi-scale proposal generation with merge + NMS (paper COCO-300 setting).

Paper alignment (COCO proposal settings)
----------------------------------------
This script is designed to match the COCO proposal-generation settings described in the paper:
- pred_iou_thresh          = 0.70
- stability_score_thresh   = 0.80
- min_mask_region_area     = 100 pixels
- proposal-level NMS IoU   = 0.65
- COCO-20:  --max_proposals 20   (typically without --multiscale)
- COCO-300: --max_proposals 300  --multiscale --scales 0.85 1.0 1.15

Important: This script ONLY generates proposals. Downstream classifier scoring / mAP evaluation
are handled elsewhere.

Output format
-------------
Saves a .pth file (torch.save) with:
  results["sam_proposals"][image_id] = {
      "image_path": str,
      "image_id": int,
      "image_size": [W, H],
      "proposals": [ {bbox_xywh, bbox_xyxy, sam_score, stability_score, mask_area, proposal_id, source_scale?}, ... ],
      "num_proposals": int
  }
and metadata in results["metadata"].

Dependencies
------------
- sam2 (build_sam2_robust, SAM2AutomaticMaskGenerator)
- torch, torchvision
- numpy, pillow
- opencv-python (cv2)  [only used when --multiscale to resize images]
"""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import masks_to_boxes
import torchvision.ops as ops

import cv2

from sam2.build_sam import build_sam2_robust
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# -------------------------
# Reproducibility
# -------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# -------------------------
# Geometry helpers
# -------------------------
def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12
    return float(inter / union)


def _nms_props(props, iou_thresh: float):
    """torchvision NMS over bbox_xyxy with score = sam_score."""
    if len(props) <= 1:
        return props
    boxes = torch.tensor([p["bbox_xyxy"] for p in props], dtype=torch.float32)
    scores = torch.tensor([float(p.get("sam_score", 0.0)) for p in props], dtype=torch.float32)
    keep = ops.nms(boxes, scores, iou_threshold=float(iou_thresh))
    return [props[i] for i in keep.tolist()]


def _resize_image_np(img: np.ndarray, scale: float):
    """Resize HxWxC uint8 image by scale. Returns (resized_img, new_h, new_w)."""
    if scale == 1.0:
        h, w = img.shape[:2]
        return img, h, w
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, new_h, new_w


# -------------------------
# Proposal generation
# -------------------------
def _mask_to_bbox_xyxy(mask_arr: np.ndarray):
    """
    Convert a binary mask to a tight bbox in xyxy.
    Uses torchvision.ops.masks_to_boxes for consistency.
    """
    if mask_arr.dtype != np.uint8:
        mask_arr = (mask_arr > 0).astype(np.uint8)
    elif mask_arr.max() > 1:
        mask_arr = (mask_arr > 0).astype(np.uint8)

    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)  # (1, H, W)
    bbox = masks_to_boxes(mask_tensor)[0].int().tolist()   # [x1, y1, x2, y2]
    return bbox


def _process_masks_to_proposals(
    masks,
    H: int,
    W: int,
    *,
    proposal_id_prefix: str = "",
    source_scale: float = 1.0,
    min_bbox_side: int = 10,
):
    """
    Turn SAM masks into proposal dicts (paper-faithful: no extra heuristics beyond sanity checks).

    Notes:
    - The SAM generator already enforces min_mask_region_area (100 px) per the paper.
      We do NOT apply any extra relative min-area filter here.
    - We keep a simple min_bbox_side check to drop degenerate boxes.
    """
    proposals = []
    for mi, m in enumerate(masks):
        try:
            mask_arr = m.get("segmentation", m.get("mask"))
            if mask_arr is None:
                continue

            # Mask area in pixels (in current coordinate system)
            mask_area = int(np.sum(mask_arr > 0))

            # Tight bbox
            x1, y1, x2, y2 = _mask_to_bbox_xyxy(mask_arr)

            # Clamp to image bounds
            x1 = max(0, min(int(x1), W))
            y1 = max(0, min(int(y1), H))
            x2 = max(0, min(int(x2), W))
            y2 = max(0, min(int(y2), H))

            # Validate bbox
            bw, bh = (x2 - x1), (y2 - y1)
            if bw <= 0 or bh <= 0:
                continue
            if bw < min_bbox_side or bh < min_bbox_side:
                continue

            prop = {
                "bbox_xywh": [x1, y1, bw, bh],
                "bbox_xyxy": [x1, y1, x2, y2],
                "sam_score": float(m.get("predicted_iou", 0.5)),
                "stability_score": float(m.get("stability_score", 0.5)),
                "mask_area": int(mask_area),
                "proposal_id": f"{proposal_id_prefix}{mi}",
            }
            if source_scale is not None:
                prop["source_scale"] = float(source_scale)

            proposals.append(prop)

        except Exception:
            # Keep robust: skip the bad mask and continue.
            continue

    return proposals


def generate_single_scale_sam_proposals(image_np: np.ndarray, sam_generator, max_proposals: int):
    """Single-scale: generate → convert → sort → NMS(0.65) → topK."""
    H, W = image_np.shape[:2]

    try:
        masks = sam_generator.generate(image_np)
        print(f"    SAM generated {len(masks)} initial masks")
    except Exception as e:
        print(f"    SAM failed: {e}")
        print("    Using fallback full-image proposal")
        return [{
            "bbox_xywh": [0, 0, W, H],
            "bbox_xyxy": [0, 0, W, H],
            "sam_score": 1.0,
            "mask_area": int(W * H),
            "stability_score": 1.0,
            "proposal_id": "fallback_0",
            "source_scale": 1.0,
        }]

    if not masks:
        print("    No masks generated, using fallback")
        return [{
            "bbox_xywh": [0, 0, W, H],
            "bbox_xyxy": [0, 0, W, H],
            "sam_score": 1.0,
            "mask_area": int(W * H),
            "stability_score": 1.0,
            "proposal_id": "fallback_0",
            "source_scale": 1.0,
        }]

    props = _process_masks_to_proposals(masks, H, W, proposal_id_prefix="", source_scale=1.0)

    if not props:
        print("    No valid proposals after filtering, using fallback")
        return [{
            "bbox_xywh": [0, 0, W, H],
            "bbox_xyxy": [0, 0, W, H],
            "sam_score": 1.0,
            "mask_area": int(W * H),
            "stability_score": 1.0,
            "proposal_id": "fallback_0",
            "source_scale": 1.0,
        }]

    # Sort by SAM score (desc)
    props.sort(key=lambda p: float(p.get("sam_score", 0.0)), reverse=True)

    # Proposal-level NMS (paper: 0.65)
    props = _nms_props(props, iou_thresh=0.65)
    print(f"    After NMS: {len(props)} proposals")

    return props[:max_proposals]


def generate_multiscale_sam_proposals(
    image_np: np.ndarray,
    sam_generator,
    max_proposals: int,
    scales,
):
    """
    Multi-scale (paper COCO-300): generate at each scale → map boxes back → merge → NMS(0.65) → topK.

    IMPORTANT: We do not apply any additional diversity heuristics beyond multiscale+NMS,
    to stay aligned with the paper description/settings.
    """
    H, W = image_np.shape[:2]
    all_props = []

    print(f"    Generating proposals at scales: {scales}")

    for s in scales:
        try:
            scaled_img, Hs, Ws = _resize_image_np(image_np, float(s))
            print(f"    Scale {s}: Processing {Ws}x{Hs} image")

            masks_s = sam_generator.generate(scaled_img)
            print(f"    Scale {s}: Generated {len(masks_s)} initial masks")

            props_s = _process_masks_to_proposals(
                masks_s, Hs, Ws,
                proposal_id_prefix=f"scale_{s}_",
                source_scale=float(s),
            )

            # Map each bbox from scaled coords → original coords
            if float(s) != 1.0:
                inv_sx = W / float(Ws)
                inv_sy = H / float(Hs)
                for p in props_s:
                    x1, y1, x2, y2 = p["bbox_xyxy"]
                    x1 = int(round(x1 * inv_sx))
                    x2 = int(round(x2 * inv_sx))
                    y1 = int(round(y1 * inv_sy))
                    y2 = int(round(y2 * inv_sy))

                    # Clamp
                    x1 = max(0, min(x1, W))
                    x2 = max(0, min(x2, W))
                    y1 = max(0, min(y1, H))
                    y2 = max(0, min(y2, H))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    p["bbox_xyxy"] = [x1, y1, x2, y2]
                    p["bbox_xywh"] = [x1, y1, x2 - x1, y2 - y1]

            all_props.extend(props_s)
            print(f"    Scale {s}: Created {len(props_s)} valid proposals")

        except Exception as e:
            print(f"    Error at scale {s}: {e}")
            continue

    if not all_props:
        print("    No valid proposals generated, using fallback")
        return [{
            "bbox_xywh": [0, 0, W, H],
            "bbox_xyxy": [0, 0, W, H],
            "sam_score": 1.0,
            "mask_area": int(W * H),
            "stability_score": 1.0,
            "proposal_id": "fallback_0",
            "source_scale": 1.0,
        }]

    print(f"    Total proposals before merging: {len(all_props)}")

    # Sort by SAM score (desc)
    all_props.sort(key=lambda p: float(p.get("sam_score", 0.0)), reverse=True)

    # Proposal-level NMS (paper: 0.65)
    all_props = _nms_props(all_props, iou_thresh=0.65)

    # Cap to max_proposals
    final_props = all_props[:max_proposals]
    print(f"    Final proposals after merge+NMS+cap: {len(final_props)}")
    return final_props


def generate_sam_proposals(image_np, sam_generator, max_proposals: int, use_multiscale: bool, scales):
    if use_multiscale:
        return generate_multiscale_sam_proposals(image_np, sam_generator, max_proposals, scales=scales)
    return generate_single_scale_sam_proposals(image_np, sam_generator, max_proposals)


# -------------------------
# Dataset helpers
# -------------------------
def load_visual_grounding_data(annotations_file: str):
    """Loads a JSON list of samples. Each sample should include an `image_id` key."""
    with open(annotations_file, "r") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} visual grounding samples")
    return samples


def get_coco_image_path(image_id: int, images_dir: str, split: str = "train2014"):
    """
    Default COCO filename format for train2014:
        COCO_train2014_000000123456.jpg
    """
    if split == "train2014":
        filename = f"COCO_train2014_{image_id:012d}.jpg"
    elif split == "val2014":
        filename = f"COCO_val2014_{image_id:012d}.jpg"
    else:
        # allow user to pass custom split; keep pattern consistent
        filename = f"COCO_{split}_{image_id:012d}.jpg"
    return Path(images_dir) / filename


def main():
    parser = argparse.ArgumentParser(description="Paper-faithful SAM-2.1 proposal generator (COCO / RefCOCO).")
    parser.add_argument("--annotations_file", required=True,
                        help="Path to JSON list of grounding samples (expects `image_id` in each entry).")
    parser.add_argument("--images_dir", required=True,
                        help="Directory containing COCO images (e.g., train2014 JPGs).")
    parser.add_argument("--coco_split", default="train2014",
                        help="COCO split string used in filename pattern (default: train2014).")

    parser.add_argument("--output_dir", default="./sam_proposals_refcoco",
                        help="Output directory for saved .pth proposals.")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Max unique images to process (0 = all). Useful for quick tests.")
    parser.add_argument("--max_proposals", type=int, default=20,
                        help="Max proposals per image (paper: 20 or 300).")

    parser.add_argument("--sam_config", required=True,
                        help="SAM-2.1 config YAML path.")
    parser.add_argument("--sam_checkpoint", required=True,
                        help="SAM-2.1 checkpoint path.")

    parser.add_argument("--multiscale", action="store_true",
                        help="Enable multiscale proposal generation (paper: used for COCO-300).")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.85, 1.0, 1.15],
                        help="Scales for multiscale inference (paper: 0.85 1.0 1.15).")

    parser.add_argument("--approach_name", default="paperfaithful",
                        help="Name tag used in output filename / metadata.")

    args = parser.parse_args()

    # Validate paths
    for pth, name in [
        (args.annotations_file, "annotations_file"),
        (args.images_dir, "images_dir"),
        (args.sam_config, "sam_config"),
        (args.sam_checkpoint, "sam_checkpoint"),
    ]:
        if not Path(pth).exists():
            raise FileNotFoundError(f"{name} not found: {pth}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize SAM generator (paper settings)
    print("Initializing SAM-2.1 automatic mask generator...")
    sam_model = build_sam2_robust(args.sam_config, args.sam_checkpoint)

    sam_generator = SAM2AutomaticMaskGenerator(
        model=sam_model,
        points_per_side=32,
        pred_iou_thresh=0.70,
        stability_score_thresh=0.80,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        box_nms_thresh=0.65,
    )

    print("SAM Configuration (paper):")
    print("  points_per_side: 32")
    print("  pred_iou_thresh: 0.70")
    print("  stability_score_thresh: 0.80")
    print("  min_mask_region_area: 100")
    print("  box_nms_thresh: 0.65")
    print(f"  max_proposals: {args.max_proposals}")
    print(f"  multiscale: {args.multiscale}")
    if args.multiscale:
        print(f"  scales: {args.scales}")

    # Load annotations / image IDs
    print("\nLoading visual grounding annotations...")
    samples = load_visual_grounding_data(args.annotations_file)

    image_ids = sorted(list({int(s["image_id"]) for s in samples if "image_id" in s}))
    print(f"Found {len(image_ids)} unique images from {len(samples)} samples")

    if args.max_images and args.max_images > 0:
        image_ids = image_ids[:args.max_images]
        print(f"Processing first {len(image_ids)} images (max_images={args.max_images})")
    else:
        print(f"Processing all {len(image_ids)} images")

    results = {
        "sam_proposals": {},
        "image_metadata": {},
        "processing_stats": {
            "total_images": len(image_ids),
            "successful_images": 0,
            "failed_images": 0,
            "total_proposals_generated": 0,
        },
        "metadata": {
            "annotations_file": args.annotations_file,
            "images_dir": args.images_dir,
            "coco_split": args.coco_split,
            "sam_config_path": args.sam_config,
            "sam_checkpoint_path": args.sam_checkpoint,
            "sam_settings": {
                "points_per_side": 32,
                "pred_iou_thresh": 0.70,
                "stability_score_thresh": 0.80,
                "min_mask_region_area": 100,
                "box_nms_thresh": 0.65,
                "proposal_nms_iou": 0.65,
                "max_proposals_per_image": args.max_proposals,
                "multiscale": bool(args.multiscale),
                "scales": list(map(float, args.scales)),
            },
            "approach_name": args.approach_name,
            "seed": SEED,
        }
    }

    failed_images = []

    for i, image_id in enumerate(image_ids):
        print(f"\nProcessing image {i+1}/{len(image_ids)}: {image_id}")
        try:
            image_path = get_coco_image_path(image_id, args.images_dir, split=args.coco_split)
            if not image_path.exists():
                print(f"  ERROR: image not found: {image_path}")
                failed_images.append(image_id)
                results["processing_stats"]["failed_images"] += 1
                continue

            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            H, W = image_np.shape[:2]
            print(f"  Image: {image_path.name} ({W}x{H})")

            proposals = generate_sam_proposals(
                image_np,
                sam_generator,
                max_proposals=int(args.max_proposals),
                use_multiscale=bool(args.multiscale),
                scales=list(map(float, args.scales)),
            )
            print(f"  Generated {len(proposals)} final proposals")

            results["sam_proposals"][image_id] = {
                "image_path": str(image_path),
                "image_id": int(image_id),
                "image_size": [int(W), int(H)],
                "proposals": proposals,
                "num_proposals": int(len(proposals)),
            }
            results["image_metadata"][image_id] = {
                "filename": image_path.name,
                "width": int(W),
                "height": int(H),
                "proposals_generated": int(len(proposals)),
            }

            results["processing_stats"]["successful_images"] += 1
            results["processing_stats"]["total_proposals_generated"] += int(len(proposals))

        except Exception as e:
            print(f"  ERROR processing image {image_id}: {e}")
            traceback.print_exc()
            failed_images.append(image_id)
            results["processing_stats"]["failed_images"] += 1
            continue

    results["metadata"]["failed_images"] = failed_images

    out_file = output_dir / f"sam_proposals_{args.approach_name}_{len(image_ids)}images.pth"
    torch.save(results, out_file)

    stats = results["processing_stats"]
    print("\n" + "=" * 60)
    print(f"PROCESSING SUMMARY - {args.approach_name.upper()}")
    print("=" * 60)
    print(f"Total images attempted: {stats['total_images']}")
    print(f"Successfully processed: {stats['successful_images']}")
    print(f"Failed: {stats['failed_images']}")
    print(f"Total SAM proposals generated: {stats['total_proposals_generated']}")
    avg = stats["total_proposals_generated"] / max(1, stats["successful_images"])
    print(f"Average proposals per image: {avg:.1f}")
    if failed_images:
        print(f"Failed image IDs: {failed_images}")
    print(f"\nSaved: {out_file}")
    print("Done.")


if __name__ == "__main__":
    main()
