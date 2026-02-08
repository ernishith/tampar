# Dynamic file to accomodate diferent images through annotations.

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

import cv2

# NumPy compatibility fix for TAMPAR (NumPy 1.24+)
import numpy as np
import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from src.maskrcnn.data import register_datasets  # noqa
from src.tampering.parcel import Parcel, ParcelView

if not hasattr(np, "int0"):
    print("Applying NumPy int0 compatibility fix")
    np.int0 = np.int32
if not hasattr(np, "object"):
    np.object = object

# DEFAULTS
DEFAULT_COCO = [ROOT / "data" / "tampar_sample" / "tampar_adversarial_validation.json"]
DEFAULT_IMAGE_ROOT = ROOT / "data" / "tampar_sample"
DEFAULT_UVMAPS_DIR = "uvmaps"
DEFAULT_WEIGHTS = ROOT / "src" / "maskrcnn" / "weights" / "ResNet-50-FPN.pth"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TAMPAR UV Map Generator")
    parser.add_argument(
        "--json", "-j", nargs="+", help="COCO JSON annotation files", default=None
    )
    parser.add_argument(
        "--image-root",
        "-i",
        type=str,
        default=str(DEFAULT_IMAGE_ROOT),
        help="Image root directory",
    )
    parser.add_argument(
        "--uvmaps-dir",
        "-u",
        type=str,
        default=DEFAULT_UVMAPS_DIR,
        help="UV maps output folder name (default: uvmaps_adversarial)",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to model weights",
    )
    parser.add_argument(
        "--gt-only",
        action="store_true",
        help="Only generate GT UV maps (skip predictions)",
    )
    return parser.parse_args()


def get_coco_annotations(args):
    """Get COCO annotations from args or use default"""
    if args.json:
        coco_paths = [Path(p) for p in args.json]
        print(f"Using provided JSONs: {len(coco_paths)} files")
        for p in coco_paths:
            if not p.exists():
                print(f"JSON not found: {p}")
                sys.exit(1)
        return coco_paths
    else:
        print("Using default JSONs")
        return DEFAULT_COCO


def save_uvmap(image_path: Path, output_path: Path, keypoints=None, predictor=None):
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return None

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        print(f"Failed to load: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if keypoints is None and predictor is not None:
        outputs = predictor(img_rgb)
        if len(outputs["instances"].pred_keypoints) == 0:
            return None
        keypoints = outputs["instances"].pred_keypoints[0, :, :2].cpu().numpy()

    print(f"Processing: {image_path} with keypoints: {keypoints}")
    view = ParcelView(image_path, np.array(keypoints))
    if view.uv_map is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(
            output_path.as_posix(), cv2.cvtColor(view.uv_map, cv2.COLOR_RGB2BGR)
        )
    else:
        print(f"UV map generation failed for: {image_path}")
    return view


def create_pred_uvmaps(predictor: DefaultPredictor, image_paths: List[Path]):
    for img_path in tqdm.tqdm(image_paths):
        new_path = img_path.parent / f"{img_path.stem}_uvmap_pred.png"
        save_uvmap(img_path, new_path, predictor=predictor)


def create_gt_uvmaps(
    coco_annotations, image_root: Path, uvmaps_dir: Path, groundtruth=True
):
    base_views = {i: [] for i in range(30)}  # parcel_id: List[ParcelView]
    view_infos = []
    identification_string = "gt" if groundtruth else "pred"

    for image_info in tqdm.tqdm(coco_annotations["images"]):
        rel_image_path = Path(image_info["file_name"])
        image_path = image_root / rel_image_path
        image_id = image_info["id"]
        annotations = [
            a for a in coco_annotations["annotations"] if a["image_id"] == image_id
        ]

        if len(annotations) == 1:
            keypoints = np.array(annotations[0]["keypoints"]).reshape(-1, 3)[..., :2]
            new_path = (
                image_path.parent
                / f"{image_path.stem}_uvmap_{identification_string}.png"
            )
            view = save_uvmap(image_path, new_path, keypoints=keypoints)

            if view is not None and image_path.parent.name == "base":
                base_views[view.parcel_id].append(view)

            for name, sidesurface in view.side_surfaces.items():
                view_infos.append(
                    {
                        "image_path": view.image_path.relative_to(
                            image_root
                        ).as_posix(),
                        "image_id": view.image_id,
                        "name_uvmap": name,
                        "convexness": sidesurface.convexness,
                        "rectangleness": sidesurface.rectangleness,
                        "area": sidesurface.area,
                        "score": sidesurface.score,
                        "keypoints_side": sidesurface.keypoints.tolist(),
                        "name_keypoints": view.side_surface_mapping[name],
                        "keypoints_parcel": view.keypoints.tolist(),
                        "angles": sidesurface.angles,
                    }
                )

    if groundtruth:
        for parcel_id, views in base_views.items():
            if len(views):
                parcel = Parcel(views)
                uvmap = parcel.uvmap
                if uvmap is not None:
                    new_path = uvmaps_dir / f"id_{str(parcel_id).zfill(2)}_uvmap.png"
                    cv2.imwrite(
                        new_path.as_posix(), cv2.cvtColor(uvmap, cv2.COLOR_RGB2BGR)
                    )
    return view_infos


def merge_coco_annotations(json_paths):
    merged = {"images": [], "annotations": [], "categories": None}
    img_id_offset = ann_id_offset = 0

    for json_path in json_paths:
        print(f"Loading {json_path}...")
        with open(json_path) as f:
            data = json.load(f)

        if merged["categories"] is None:
            merged["categories"] = data["categories"]

        for img in data["images"]:
            img["id"] += img_id_offset
            merged["images"].append(img)

        for ann in data["annotations"]:
            ann["id"] += ann_id_offset
            ann["image_id"] += img_id_offset
            merged["annotations"].append(ann)

        img_id_offset += len(data["images"])
        ann_id_offset += len(data["annotations"])

    print(
        f"Merged: {len(merged['images'])} images, {len(merged['annotations'])} annotations"
    )
    return merged


if __name__ == "__main__":
    args = parse_args()

    # Create paths from arguments
    image_root = Path(args.image_root)
    uvmaps_dir = image_root / args.uvmaps_dir
    uvmaps_dir.mkdir(exist_ok=True, parents=True)

    print(f"TAMPAR UV Generator")
    print(f"Image root: {image_root}")
    print(f"UV maps dir: {uvmaps_dir}")
    print(f"Weights: {args.weights}")

    # 1. Get COCO annotations
    coco_annotations = get_coco_annotations(args)

    # 2. GROUND TRUTH UV MAPS
    print("\nGenerating GT UV Maps...")
    coco_merged = merge_coco_annotations(coco_annotations)
    gt_infos = create_gt_uvmaps(coco_merged, image_root, uvmaps_dir, groundtruth=True)

    # 3. PREDICTION UV MAPS
    if not args.gt_only:
        print("\nGenerating PRED UV Maps...")
        cfg = get_cfg()
        cfg.merge_from_file(
            str(ROOT / "src" / "maskrcnn" / "configs" / "Base-RCNN-FPN.yaml")
        )
        cfg.MODEL.WEIGHTS = args.weights
        print("Configuration:")
        print(cfg)

        print(f"Loading model: {args.weights}")
        predictor = DefaultPredictor(cfg)

        image_paths = [
            image_root / Path(img["file_name"])
            for img in coco_merged["images"]
            if (image_root / Path(img["file_name"])).exists()
        ]
        print(f"Processing {len(image_paths)} images...")
        create_pred_uvmaps(predictor, image_paths)

    # Summary
    print(f"\nCOMPLETE!")
    print(f"GT side surfaces: {len(gt_infos)}")
    print(f"GT files: {len(list(image_root.rglob('*_uvmap_gt.png')))}")
    if not args.gt_only:
        print(f"PRED files: {len(list(image_root.rglob('*_uvmap_pred.png')))}")
    print(f"Parcel UVs: {uvmaps_dir}")
