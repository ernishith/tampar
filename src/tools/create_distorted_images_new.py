#!/usr/bin/env python3
"""
TAMPAR Distorted Images Generator - ROTATING DISTORTION MODE
Applies different distortion values to successive images (round-robin)
Fully self-contained with defaults - runs without any arguments!
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

import cv2
import numpy as np
import tqdm
from PIL import Image as PILImage
from skimage import measure
from wand.image import Image

# DEFAULT CONFIGURATION (ALWAYS AVAILABLE)
DEFAULT_IMAGE_ROOT = ROOT / "data" / "tampar_sample"
DEFAULT_OUTPUT_DIR = "Distortion"
DEFAULT_DISTORTION_VALUES = [-0.08, -0.04, -0.02, 0.04, 0.08, 0.16]
DEFAULT_JSON_FILES = ["tampar_sample_validation.json"]
KEYPOINT_COLORS = np.linspace(50, 255, 8).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments - ALL OPTIONAL with defaults"""
    parser = argparse.ArgumentParser(
        description="TAMPAR Distorted Images Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json",
        "-j",
        nargs="+",
        default=None,
        help="COCO JSON annotation files (default: tampar_sample_validation.json)",
    )
    parser.add_argument(
        "--image-root",
        "-i",
        type=str,
        default=None,
        help=f"Image root directory (default: {DEFAULT_IMAGE_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help=f"Output folder name inside image-root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--distortion-values",
        "-d",
        nargs="+",
        type=float,
        default=None,
        help="Distortion values (default: -0.08 -0.04 -0.02 0.04 0.08 0.16)",
    )
    return parser.parse_args()


def compute_new_keypoint_annotations(img: np.ndarray):
    """Extract keypoint positions from color-coded annotation image"""
    keypoints = []
    for color in KEYPOINT_COLORS:
        mask = img == color
        if np.sum(mask) <= 20:
            return None
        mask = mask.astype(np.uint8) * 255

        # Clean up noise
        kernel_size = 5
        eroded_image = cv2.erode(
            mask,
            np.ones((kernel_size, kernel_size), np.uint8),
            iterations=1,
        )
        _, cleaned_mask = cv2.threshold(eroded_image, 100, 255, cv2.THRESH_BINARY)

        # Extract keypoints
        true_indices = np.argwhere(cleaned_mask)
        mean = np.mean(true_indices, axis=0)  # y, x
        keypoints.append([*mean.tolist()[::-1], 2])
    return sum(keypoints, [])


def compute_new_segmentation_annotations(img: np.ndarray):
    """Extract bbox and segmentation from distorted mask"""
    # BBox
    true_indices = np.argwhere(img > 0)
    if len(true_indices) > 0:
        y_min, x_min = np.min(true_indices, axis=0)
        y_max, x_max = np.max(true_indices, axis=0)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        bbox = [float(i) for i in bbox]
    else:
        bbox = []

    # Segmentation
    img[img > 0.1] = 1
    img[img <= 0.1] = 0
    polygons = binary_mask_to_polygon(img)
    return polygons, bbox


def binary_mask_to_polygon(binary_mask: np.ndarray, tolerance: float = 0):
    """Converts a binary mask to COCO polygon representation
    From: https://github.com/waspinator/pycococreator
    License: Apache 2.0
    """

    def close_contour(contour: np.ndarray):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    polygons = []
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def create_keypoint_annotation_image(image_info: dict, keypoints: np.ndarray):
    """Encode keypoints as colored circles"""
    white_img = np.zeros([image_info["height"], image_info["width"]])
    for idx, point in enumerate(keypoints):
        x, y = int(point[0]), int(point[1])
        color = (int(KEYPOINT_COLORS[idx]), int(KEYPOINT_COLORS[idx]))
        cv2.circle(white_img, (x, y), 50, color, -1)
    white_img = white_img.astype(np.uint8)
    return white_img


def create_segm_annotation_image(image_info: dict, points: np.ndarray):
    """Create binary segmentation mask"""
    img = np.zeros([image_info["height"], image_info["width"]])
    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(img, [points], color=255)
    return img


def process_single_json(
    json_path: Path, image_root: Path, output_root: Path, distortion_values: List[float]
):
    """
    Process one JSON file - apply rotating distortions to images

    BEHAVIOR:
    - Image 0 → distortion_values[0]
    - Image 1 → distortion_values[1]
    - Image 6 → distortion_values[0] (cycle repeats)
    """
    print(f"\n{'='*70}")
    print(f" Processing: {json_path.name}")
    print(f" Distortion cycle: {distortion_values}")
    print(f"{'='*70}")

    # Load JSON
    try:
        with open(json_path) as f:
            info = json.load(f)
    except Exception as e:
        print(f" Failed to load {json_path}: {e}")
        return

    new_annotations = []
    new_image_infos = []
    num_distortions = len(distortion_values)
    skipped_count = 0

    # Process each image with rotating distortion
    for img_idx, image_info in enumerate(
        tqdm.tqdm(info["images"], desc="Processing images")
    ):
        # SELECT DISTORTION (round-robin)
        distortion_val = distortion_values[img_idx % num_distortions]

        # Load original data
        image_path = image_root / image_info["file_name"]
        if not image_path.exists():
            print(f"  Image not found: {image_path}")
            skipped_count += 1
            continue

        annotations = [
            a for a in info["annotations"] if a["image_id"] == image_info["id"]
        ]
        if len(annotations) != 1:
            skipped_count += 1
            continue
        annotation = annotations[0]

        # Extract keypoints and segmentation
        keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)[:, :2]
        segm = np.array(annotation["segmentation"][0]).reshape(-1, 2)

        # Create annotation images
        img_anno_kp = create_keypoint_annotation_image(image_info, keypoints)
        img_anno_seg = create_segm_annotation_image(image_info, segm)

        images_before = {
            "input": cv2.imread(str(image_path)),
            "anno_kp": img_anno_kp,
            "anno_seg": img_anno_seg,
        }

        # APPLY DISTORTION using ImageMagick Wand
        images_after = {}

        # Maintain folder structure: Distortion/base/id_00_00.jpg
        relative_path = Path(image_info["file_name"])
        new_image_path = output_root / relative_path
        new_image_path.parent.mkdir(exist_ok=True, parents=True)

        crop_area = None
        try:
            for im_type, img in images_before.items():
                with Image.from_array(img) as img_wand:
                    img_wand.virtual_pixel = "transparent"
                    img_wand.distort("barrel", (distortion_val, 0.0, 0.0, 1.0))
                    img_distorted = np.array(img_wand)

                    if im_type != "input":
                        img_distorted = img_distorted[:, :, 0]
                    else:
                        crop_area = PILImage.fromarray(img_distorted).getbbox()

                    if crop_area is None:
                        raise ValueError("Crop area computation failed")

                    images_after[im_type] = img_distorted[
                        crop_area[1] : crop_area[3], crop_area[0] : crop_area[2]
                    ]
        except Exception as e:
            print(f"  Distortion failed for {image_path}: {e}")
            skipped_count += 1
            continue

        # Save distorted image
        try:
            cv2.imwrite(str(new_image_path), images_after["input"])
        except Exception as e:
            print(f"  Failed to save {new_image_path}: {e}")
            skipped_count += 1
            continue

        # Recover new annotations
        new_keypoints = compute_new_keypoint_annotations(images_after["anno_kp"])
        new_segmentation, new_bbox = compute_new_segmentation_annotations(
            images_after["anno_seg"]
        )

        if new_keypoints is None or len(new_keypoints) != 8 * 3:
            print(f"  Keypoints lost for image {img_idx}")
            skipped_count += 1
            continue

        # Update annotation
        annotation["segmentation"] = new_segmentation
        annotation["keypoints"] = new_keypoints
        annotation["bbox"] = new_bbox
        annotation["area"] = new_bbox[2] * new_bbox[3] if len(new_bbox) == 4 else 0
        new_annotations.append(annotation)

        # Update image info
        image_info["file_name"] = str(relative_path)
        image_info["height"] = images_after["anno_kp"].shape[0]
        image_info["width"] = images_after["anno_kp"].shape[1]
        new_image_infos.append(image_info)

    # Save new JSON (one per input JSON)
    info["annotations"] = new_annotations
    info["images"] = new_image_infos

    output_json = output_root / json_path.name.replace(".json", "_distorted.json")
    try:
        with open(output_json, "w") as f:
            json.dump(info, f, indent=2)
        print(f" Saved: {output_json}")
        print(
            f" Processed: {len(new_image_infos)}/{len(info['images'])} images (skipped: {skipped_count})"
        )
    except Exception as e:
        print(f" Failed to save JSON: {e}")


def main():
    args = parse_args()

    # USE DEFAULTS IF NOT PROVIDED
    image_root = Path(args.image_root) if args.image_root else DEFAULT_IMAGE_ROOT
    output_dir_name = args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR
    output_root = image_root / output_dir_name
    output_root.mkdir(exist_ok=True, parents=True)

    distortion_values = (
        args.distortion_values if args.distortion_values else DEFAULT_DISTORTION_VALUES
    )

    # JSON files - use defaults if not provided
    json_files = args.json if args.json else DEFAULT_JSON_FILES

    print("=" * 70)
    print(" TAMPAR Distorted Images Generator - ROTATING MODE")
    print(f" Image root: {image_root}")
    print(f" Output root: {output_root}")
    print(f" Distortion values: {distortion_values}")
    print(f" JSON files: {json_files}")
    print("=" * 70)

    # Validate JSON paths
    json_paths = []
    for json_file in json_files:
        json_path = Path(json_file)

        # Try multiple locations
        if not json_path.exists():
            # Try relative to image_root
            json_path = image_root / json_file

        if not json_path.exists():
            # Try absolute path
            json_path = Path(json_file).resolve()

        if not json_path.exists():
            print(f" JSON not found: {json_file}")
            print(f"   Tried: {json_file}")
            print(f"   Tried: {image_root / json_file}")
            continue

        json_paths.append(json_path)

    if not json_paths:
        print(" No valid JSON files found!")
        print(f" Default location: {image_root / DEFAULT_JSON_FILES[0]}")
        sys.exit(1)

    # Process each JSON file
    for json_path in json_paths:
        process_single_json(json_path, image_root, output_root, distortion_values)

    print("\n" + "=" * 70)
    print(" ALL JSON FILES PROCESSED!")
    print(f" Output directory: {output_root}")
    print(f" Output JSONs: {len(json_paths)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
