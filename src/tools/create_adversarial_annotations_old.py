"""
create_adversarial_annotations.py

Creates a COMBINED COCO annotation file containing:
- All original images
- All adversarially attacked images (FGSM + PGD)

Result: Double the number of original images in the JSON file.
"""

import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

import tqdm

# ============================================================================
# Configuration
# ============================================================================
IMAGE_ROOT = ROOT / "data" / "tampar_sample"
ATTACK_DIR = IMAGE_ROOT / "adversarial_attacks"
ADVERSARIAL_RENAME_MAP = ATTACK_DIR / "adversarial_rename_map.json"
ORIGINAL_ANNOTATION_FILE = IMAGE_ROOT / "tampar_sample_validation.json"

# Output: Single combined file
COMBINED_ANNOTATION_FILE = IMAGE_ROOT / "tampar_combined.json"


# ============================================================================
# Helper Functions
# ============================================================================


def clean_adversarial_annotations(image_root: Path):
    """
    Delete all adversarial annotation JSON files if they exist.

    Excludes adversarial_rename_map.json (needed for mapping).

    Args:
        image_root: Root directory to search for adversarial JSONs
    """
    # Find all adversarial JSONs but exclude the rename map
    adversarial_jsons = []
    for json_file in image_root.rglob("*adversarial*.json"):
        # Skip the rename map file
        if json_file.name != "adversarial_rename_map.json":
            adversarial_jsons.append(json_file)

    if adversarial_jsons:
        print(
            f"\n⚠️  Found {len(adversarial_jsons)} existing adversarial annotation file(s):"
        )
        for json_file in adversarial_jsons:
            print(f"  - {json_file.name}")

        print(f"\nDeleting existing adversarial annotations...")
        for json_file in adversarial_jsons:
            json_file.unlink()
            print(f"  ✓ Deleted: {json_file.name}")
    else:
        print("\n✓ No existing adversarial annotations found")


def get_max_ids(coco_dict: dict) -> tuple:
    """
    Get maximum image ID and annotation ID from COCO annotations.

    Args:
        coco_dict: COCO annotation dictionary

    Returns:
        (max_image_id, max_annotation_id)
    """
    max_image_id = 0
    max_annotation_id = 0

    if len(coco_dict["images"]) > 0:
        max_image_id = max([img["id"] for img in coco_dict["images"]])

    if len(coco_dict["annotations"]) > 0:
        max_annotation_id = max([ann["id"] for ann in coco_dict["annotations"]])

    return max_image_id, max_annotation_id


def load_coco_annotations(json_path: Path) -> dict:
    """
    Load COCO format annotations from JSON file.

    Args:
        json_path: Path to COCO JSON file

    Returns:
        COCO annotation dictionary
    """
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    return coco_data


# ============================================================================
# Annotation Creation Functions
# ============================================================================
def create_adversarial_annotations_for_image(
    image_info: dict,
    annotations_list: list,
    attack_type: str,
    image_root: Path,
    attack_dir: Path,
    new_image_id: int,
    new_annotation_id: int,
) -> tuple:
    """
    Create annotations for a single adversarial image.

    Args:
        image_info: Original image info dict
        annotations_list: List of annotations for original image
        attack_type: "fgsm" or "pgd"
        image_root: Root directory of original images
        attack_dir: Directory containing attacked images
        new_image_id: New image ID to assign
        new_annotation_id: Starting annotation ID

    Returns:
        (new_image_info, new_annotations_list, next_annotation_id) or None if image doesn't exist
    """
    original_image_path = Path(image_info["file_name"])

    # Construct attacked image path
    attacked_image_name = f"{original_image_path.stem}_{attack_type}.jpg"
    attacked_image_path = attack_dir / original_image_path.parent / attacked_image_name

    # Check if attacked image exists
    if not attacked_image_path.exists():
        return None

    # Create new image info
    new_image_info = copy.deepcopy(image_info)
    new_image_info["id"] = new_image_id
    new_image_info["file_name"] = attacked_image_path.relative_to(IMAGE_ROOT).as_posix()

    # Copy annotations for this image
    new_annotations = []
    next_annotation_id = new_annotation_id

    for annotation in annotations_list:
        new_annotation = copy.deepcopy(annotation)
        new_annotation["id"] = next_annotation_id
        new_annotation["image_id"] = new_image_id
        new_annotations.append(new_annotation)
        next_annotation_id += 1

    return new_image_info, new_annotations, next_annotation_id


def create_combined_annotations(
    original_coco: dict,
    image_root: Path,
    attack_dir: Path,
    adversarial_rename_map: dict,
) -> dict:
    """
    Create combined COCO annotations with ALL images:
    - Original images
    - Adversarially attacked images (alternating FGSM/PGD)

    Result: 2x the number of original images.

    Args:
        original_coco: Original annotations
        image_root: Root directory of original images
        attack_dir: Directory containing attacked images

    Returns:
        Combined COCO dict
    """
    print("\n" + "=" * 80)
    print("Building Combined Annotations (Original + Adversarial)")
    print("=" * 80)

    # Get max IDs from original JSON
    max_image_id, max_annotation_id = get_max_ids(original_coco)

    print(f"\nOriginal dataset statistics:")
    print(f"  Images: {len(original_coco['images'])}")
    print(f"  Annotations: {len(original_coco['annotations'])}")
    print(f"  Max image ID: {max_image_id}")
    print(f"  Max annotation ID: {max_annotation_id}")

    # Create combined dataset starting with original data
    combined_coco = copy.deepcopy(original_coco)

    if "info" in combined_coco:
        combined_coco["info"][
            "description"
        ] = "Combined dataset: Original images + Adversarial attacks (FGSM + PGD)"

    # Start with original images and annotations
    all_images = copy.deepcopy(original_coco["images"])
    all_annotations = copy.deepcopy(original_coco["annotations"])

    # Create annotation lookup by image_id
    annotations_by_image = {}
    for ann in original_coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Track IDs
    next_image_id = max_image_id + 1
    next_annotation_id = max_annotation_id + 1

    fgsm_count = 0
    pgd_count = 0

    print(f"\nProcessing adversarial images...")
    print(f"  Starting image ID: {next_image_id}")
    print(f"  Starting annotation ID: {next_annotation_id}")

    # Process each original image and create adversarial version
    for idx, image_info in enumerate(
        tqdm.tqdm(original_coco["images"], desc="Creating adversarial annotations")
    ):
        print("Image Info:", image_info)
        # Get annotations for this image
        image_annotations = annotations_by_image.get(image_info["id"], [])

        original_image = image_info["file_name"].split("/")[-1]
        print("Original Image:", original_image)

        # Determine attack type based on index (matching adversarial_attack_generator.py)
        # Odd indices (0, 2, 4, ...) -> FGSM
        # Even indices (1, 3, 5, ...) -> PGD
        attack_type = (
            "fgsm"
            if "fgsm" in adversarial_rename_map.get(original_image, "")
            else "pgd"
        )

        # Create adversarial annotations
        result = create_adversarial_annotations_for_image(
            image_info,
            image_annotations,
            attack_type,
            image_root,
            attack_dir,
            next_image_id,
            next_annotation_id,
        )

        if result is not None:
            new_image_info, new_annotations, next_annotation_id = result
            all_images.append(new_image_info)
            all_annotations.extend(new_annotations)
            next_image_id += 1

            if attack_type == "fgsm":
                fgsm_count += 1
            else:
                pgd_count += 1

    # Update combined dict
    combined_coco["images"] = all_images
    combined_coco["annotations"] = all_annotations

    # Print summary
    print("\n" + "=" * 80)
    print("Combined Dataset Summary")
    print("=" * 80)
    print(f"  Original images:     {len(original_coco['images']):>5}")
    print(f"  FGSM images:         {fgsm_count:>5}")
    print(f"  PGD images:          {pgd_count:>5}")
    print(f"  {'-' * 30}")
    print(f"  Total images:        {len(all_images):>5}")
    print(f"  Total annotations:   {len(all_annotations):>5}")

    expected_total = len(original_coco["images"]) * 2
    if len(all_images) == expected_total:
        print(
            f"\n  ✓ Image count is exactly 2x original ({len(all_images)} = 2 × {len(original_coco['images'])})"
        )
    else:
        print(
            f"\n  ⚠️  WARNING: Expected {expected_total} images, got {len(all_images)}"
        )
        print(f"     Some adversarial images may be missing!")

    # Verify uniqueness
    image_ids = [img["id"] for img in all_images]
    annotation_ids = [ann["id"] for ann in all_annotations]

    unique_image_ids = len(set(image_ids))
    unique_annotation_ids = len(set(annotation_ids))

    print(f"\n  Unique image IDs:      {unique_image_ids:>5}")
    print(f"  Unique annotation IDs: {unique_annotation_ids:>5}")

    if len(image_ids) == unique_image_ids:
        print("  ✓ All image IDs are unique")
    else:
        print(f"  ⚠️  WARNING: {len(image_ids) - unique_image_ids} duplicate image IDs!")

    if len(annotation_ids) == unique_annotation_ids:
        print("  ✓ All annotation IDs are unique")
    else:
        print(
            f"  ⚠️  WARNING: {len(annotation_ids) - unique_annotation_ids} duplicate annotation IDs!"
        )

    return combined_coco


def save_coco_annotations(coco_dict: dict, output_path: Path):
    """
    Save COCO annotations to JSON file.

    Args:
        coco_dict: COCO annotation dictionary
        output_path: Output JSON file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(coco_dict, f, indent=2)

    print(f"\n✓ Saved to: {output_path}")


def validate_annotations(coco_dict: dict):
    """
    Validate annotations structure.

    Args:
        coco_dict: COCO annotation dict
    """
    print("\nValidating annotations...")

    # Check for missing adversarial images
    missing_images = []
    for image_info in coco_dict["images"]:
        image_path = IMAGE_ROOT / image_info["file_name"]
        if not image_path.exists():
            missing_images.append(image_path)

    if missing_images:
        print(f"  ⚠️  WARNING: {len(missing_images)} images not found")
        for img_path in missing_images[:3]:
            print(f"      - {img_path}")
        if len(missing_images) > 3:
            print(f"      ... and {len(missing_images) - 3} more")
    else:
        print(f"  ✓ All {len(coco_dict['images'])} images found")

    # Check for duplicate IDs
    image_ids = [img["id"] for img in coco_dict["images"]]
    annotation_ids = [ann["id"] for ann in coco_dict["annotations"]]

    if len(image_ids) == len(set(image_ids)):
        print(f"  ✓ All image IDs unique")
    else:
        print(f"  ⚠️  WARNING: Duplicate image IDs found!")

    if len(annotation_ids) == len(set(annotation_ids)):
        print(f"  ✓ All annotation IDs unique")
    else:
        print(f"  ⚠️  WARNING: Duplicate annotation IDs found!")


# ============================================================================
# Main Function
# ============================================================================
def main():
    """
    Main function to create combined COCO annotations.
    """
    print("=" * 80)
    print("TAMPAR Combined Annotations Generator")
    print("Original + Adversarial Images")
    print("=" * 80)

    # Check if original annotations exist
    if not ORIGINAL_ANNOTATION_FILE.exists():
        print(f"\n❌ ERROR: Original annotation file not found:")
        print(f"   {ORIGINAL_ANNOTATION_FILE}")
        return

    # Load original annotations
    print(f"\nLoading original annotations...")
    print(f"  File: {ORIGINAL_ANNOTATION_FILE.name}")
    original_coco = load_coco_annotations(ORIGINAL_ANNOTATION_FILE)

    # Load adversarial rename map
    print(f"\nLoading adversarial rename map...")
    with open(ADVERSARIAL_RENAME_MAP, "r") as f:
        adversarial_rename_map = json.load(f)

    # Create combined annotations
    combined_coco = create_combined_annotations(
        original_coco,
        image_root=IMAGE_ROOT,
        attack_dir=ATTACK_DIR,
        adversarial_rename_map=adversarial_rename_map,
    )

    # Save combined annotation file
    print("\n" + "=" * 80)
    print("Saving Combined Annotation File")
    print("=" * 80)

    save_coco_annotations(combined_coco, COMBINED_ANNOTATION_FILE)
    validate_annotations(combined_coco)

    # Final summary
    print("\n" + "=" * 80)
    print("✓ Annotation Generation Complete!")
    print("=" * 80)

    print(f"\nGenerated File:")
    print(f"  {COMBINED_ANNOTATION_FILE.name}")
    print(f"  - Contains ALL original images")
    print(f"  - Contains ALL adversarial images (FGSM + PGD)")
    print(f"  - Total: {len(combined_coco['images'])} images")
    print(
        f"  - Ratio: {len(combined_coco['images']) / len(original_coco['images']):.1f}x original"
    )

    print("\n" + "=" * 80)


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    main()
