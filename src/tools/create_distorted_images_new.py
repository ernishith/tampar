#!/usr/bin/env python3
"""
TAMPAR Distorted Images Generator - FIXED VERSION
 Robust contour handling
 Adaptive keypoint recovery
 Skip problematic images gracefully
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

# Remove wand dependency - use OpenCV distortion
try:
    from wand.image import Image

    WAND_AVAILABLE = True
except ImportError:
    print("  Wand not available - using OpenCV distortion")
    WAND_AVAILABLE = False

# DEFAULT CONFIGURATION
DEFAULT_IMAGE_ROOT = ROOT / "data" / "tampar_sample"
DEFAULT_OUTPUT_DIR = "Distortion"
DEFAULT_DISTORTION_VALUES = [-0.08, -0.04, -0.02, 0.04, 0.08, 0.16]  # Safer range
KEYPOINT_COLORS = np.linspace(50, 255, 8).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAMPAR Distorted Images Generator")
    parser.add_argument("--json", "-j", nargs="+", default=None)
    parser.add_argument("--image-root", "-i", type=str, default=None)
    parser.add_argument("--output-dir", "-o", type=str, default=None)
    parser.add_argument(
        "--distortion-values", "-d", nargs="+", type=float, default=None
    )
    return parser.parse_args()


def opencv_barrel_distortion(img: np.ndarray, k1: float):
    """Pure OpenCV barrel distortion - NO Wand dependency"""
    h, w = img.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
    center_x, center_y = w * 0.5, h * 0.5

    # Radial distance from center
    dx = map_x - center_x
    dy = map_y - center_y
    r = np.sqrt(dx * dx + dy * dy)
    r_max = np.sqrt(center_x**2 + center_y**2)

    # Barrel distortion factor (k1 < 0 = barrel)
    factor = 1 + k1 * (r / r_max) ** 2
    factor = np.maximum(factor, 0.1)  # Prevent extreme distortion

    # Apply distortion
    map_x_dist = center_x + dx / factor
    map_y_dist = center_y + dy / factor

    # Clamp to image bounds
    map_x_dist = np.clip(map_x_dist, 0, w - 1)
    map_y_dist = np.clip(map_y_dist, 0, h - 1)

    return cv2.remap(
        img,
        map_x_dist.astype(np.float32),
        map_y_dist.astype(np.float32),
        cv2.INTER_LINEAR,
    )


def compute_new_keypoint_annotations(img: np.ndarray, tolerance: int = 10):
    """ROBUST keypoint extraction with adaptive tolerance"""
    keypoints = []
    total_pixels = np.sum(img > 20)  # Total annotation pixels

    for color in KEYPOINT_COLORS:
        mask = img == color
        pixel_count = np.sum(mask)

        # Adaptive threshold based on total pixels
        if pixel_count < max(5, total_pixels * 0.01):
            print(f"  Keypoint {color} too small: {pixel_count} pixels")
            return None

        # Aggressive noise cleanup
        mask_clean = cv2.morphologyEx(
            mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
        )
        mask_clean = cv2.morphologyEx(
            mask_clean, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
        )

        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours or len(contours[0]) < 3:
            print(f"  No valid contour for color {color}")
            return None

        # Use largest contour's centroid
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        keypoints.append([cx, cy, 2])

    return sum(keypoints, [])


def compute_new_segmentation_annotations(img: np.ndarray):
    """ROBUST segmentation extraction"""
    # Binarize robustly
    img_binary = (img > 10).astype(np.uint8) * 255

    # Clean up small noise
    kernel = np.ones((3, 3), np.uint8)
    img_clean = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_OPEN, kernel)

    # Compute bbox
    contours, _ = cv2.findContours(
        img_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return [], []

    # Largest contour for bbox
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 3:
        return [], []

    x, y, w, h = cv2.boundingRect(largest_contour)
    bbox = [float(x), float(y), float(w), float(h)]

    # Convert to polygon (OpenCV contours → COCO format)
    polygon = largest_contour.flatten().tolist()
    polygons = [polygon]

    return polygons, bbox


def binary_mask_to_polygon_safe(binary_mask: np.ndarray, tolerance: float = 0):
    """SAFE polygon conversion - handles edge cases"""
    try:
        # Ensure binary
        binary_mask = (binary_mask > 0.5).astype(np.uint8)

        # Pad and find contours
        padded_mask = np.pad(binary_mask, pad_width=1, mode="constant")
        contours = measure.find_contours(padded_mask, 0.5)

        if len(contours) == 0:
            return []

        polygons = []
        for contour in contours:
            if len(contour) < 3:
                continue

            # Subtract padding
            contour = contour - 1

            # Close contour
            if not np.allclose(contour[0], contour[-1]):
                contour = np.vstack([contour, contour[0]])

            # Approximate polygon
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue

            # Convert to COCO format [x,y,x,y,...]
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            segmentation = [max(0, coord) for coord in segmentation]

            polygons.append(segmentation)

        return polygons if polygons else []

    except Exception as e:
        print(f"  Polygon conversion failed: {e}")
        return []


def create_keypoint_annotation_image(image_info: dict, keypoints: np.ndarray):
    """Larger circles for better distortion tolerance"""
    h, w = image_info["height"], image_info["width"]
    white_img = np.zeros((h, w), dtype=np.uint16)  # Higher precision

    for idx, point in enumerate(keypoints):
        x, y = int(point[0]), int(point[1])
        radius = max(60, min(100, int(np.sqrt(w * h) / 100)))  # Adaptive radius
        color = int(KEYPOINT_COLORS[idx])
        cv2.circle(white_img, (x, y), radius, color, -1)

    return white_img.astype(np.uint8)


def create_segm_annotation_image(image_info: dict, points: np.ndarray):
    """Robust polygon fill"""
    h, w = image_info["height"], image_info["width"]
    img = np.zeros((h, w), dtype=np.uint8)

    try:
        points = points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(img, [points], 255)
    except:
        # Fallback: draw contours
        contours = points.reshape(-1, 2)
        contours = contours.reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(img, [contours], -1, 255, -1)

    return img


def process_single_json(
    json_path: Path, image_root: Path, output_root: Path, distortion_values: List[float]
):
    """Main processing function with robust error handling"""
    print(f"\n{'='*70}")
    print(f" Processing: {json_path.name}")
    print(f" Distortion cycle: {distortion_values}")
    print(f"{'='*70}")

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
    success_count = 0

    for img_idx, image_info in enumerate(
        tqdm.tqdm(info["images"], desc="Processing images")
    ):
        distortion_val = distortion_values[img_idx % num_distortions]

        image_path = image_root / image_info["file_name"]
        if not image_path.exists():
            skipped_count += 1
            continue

        annotations = [
            a for a in info["annotations"] if a["image_id"] == image_info["id"]
        ]
        if len(annotations) != 1:
            skipped_count += 1
            continue
        annotation = annotations[0]

        try:
            # Extract original annotations
            keypoints_orig = np.array(annotation["keypoints"]).reshape(-1, 3)[:, :2]
            segm_orig = np.array(annotation["segmentation"][0]).reshape(-1, 2)

            # Create annotation images (LARGER circles)
            img_anno_kp = create_keypoint_annotation_image(image_info, keypoints_orig)
            img_anno_seg = create_segm_annotation_image(image_info, segm_orig)

            # Load input image
            img_input = cv2.imread(str(image_path))
            if img_input is None:
                skipped_count += 1
                continue

            images_before = {
                "input": img_input,
                "anno_kp": img_anno_kp,
                "anno_seg": img_anno_seg,
            }

            # Apply distortion
            images_after = {}
            relative_path = Path(image_info["file_name"])
            new_image_path = output_root / relative_path
            new_image_path.parent.mkdir(exist_ok=True, parents=True)

            # Distort all images
            for im_type, img in images_before.items():
                if WAND_AVAILABLE and im_type == "input":
                    # Use Wand for RGB (more accurate)
                    try:
                        with Image.from_array(img) as img_wand:
                            img_wand.virtual_pixel = "transparent"
                            img_wand.distort("barrel", (distortion_val, 0.0, 0.0, 1.0))
                            img_distorted = np.array(img_wand)

                        # Crop transparent borders
                        crop_area = PILImage.fromarray(img_distorted).getbbox()
                        if crop_area:
                            img_distorted = img_distorted[
                                crop_area[1] : crop_area[3], crop_area[0] : crop_area[2]
                            ]
                        images_after[im_type] = img_distorted
                    except:
                        # Fallback to OpenCV
                        img_distorted = opencv_barrel_distortion(img, distortion_val)
                        images_after[im_type] = img_distorted
                else:
                    # OpenCV for annotations (grayscale)
                    img_distorted = opencv_barrel_distortion(img, distortion_val)
                    images_after[im_type] = (
                        img_distorted[:, :, 0]
                        if len(img_distorted.shape) == 3
                        else img_distorted
                    )

            # Save distorted image
            cv2.imwrite(str(new_image_path), images_after["input"])

            # Recover annotations (ROBUST)
            new_keypoints = compute_new_keypoint_annotations(images_after["anno_kp"])
            new_segmentation = binary_mask_to_polygon_safe(images_after["anno_seg"])
            new_bbox = annotation.get("bbox", [])

            if new_keypoints is None or len(new_keypoints) != 24:
                print(
                    f"  Keypoints recovery failed for image {img_idx} (dist: {distortion_val:.3f})"
                )
                skipped_count += 1
                continue

            # Update annotations
            annotation["segmentation"] = (
                new_segmentation if new_segmentation else annotation["segmentation"]
            )
            annotation["keypoints"] = new_keypoints
            annotation["bbox"] = new_bbox
            new_annotations.append(annotation)

            image_info["file_name"] = str(relative_path)
            image_info["height"] = images_after["anno_kp"].shape[0]
            image_info["width"] = images_after["anno_kp"].shape[1]
            new_image_infos.append(image_info)

            success_count += 1

        except Exception as e:
            print(f"  Processing failed for image {img_idx}: {e}")
            skipped_count += 1
            continue

    # Save results
    info["annotations"] = new_annotations
    info["images"] = new_image_infos

    output_json = output_root / json_path.name.replace(".json", "_distorted.json")
    try:
        with open(output_json, "w") as f:
            json.dump(info, f, indent=2)
        print(f" Saved: {output_json}")
        print(
            f" Success: {success_count}/{len(info['images'])} (skipped: {skipped_count})"
        )
    except Exception as e:
        print(f" JSON save failed: {e}")


def main():
    args = parse_args()

    image_root = Path(args.image_root) if args.image_root else DEFAULT_IMAGE_ROOT
    output_root = image_root / (args.output_dir or DEFAULT_OUTPUT_DIR)
    output_root.mkdir(exist_ok=True, parents=True)

    distortion_values = args.distortion_values or DEFAULT_DISTORTION_VALUES
    json_files = args.json or ["tampar_validation.json"]

    print("=" * 70)
    print(" TAMPAR Distorted Images Generator - ROBUST VERSION")
    print(f" Image root: {image_root}")
    print(f" Output: {output_root}")
    print(f" Distortions: {distortion_values}")
    print(f" JSONs: {json_files}")
    print(f"  Wand: {'Present' if WAND_AVAILABLE else 'NA (OpenCV)'}")
    print("=" * 70)

    # Process JSONs
    json_paths = []
    for json_file in json_files:
        for candidate_path in [Path(json_file), image_root / json_file]:
            if candidate_path.exists():
                json_paths.append(candidate_path)
                break
        else:
            print(f" JSON not found: {json_file}")

    for json_path in json_paths:
        process_single_json(json_path, image_root, output_root, distortion_values)

    print("\n PROCESSING COMPLETE!")


if __name__ == "__main__":
    main()
