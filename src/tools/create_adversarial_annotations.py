"""
create_adversarial_annotations.py (UPDATED - Separate JSON Files)

Creates SEPARATE adversarial COCO annotation files:
- One JSON file per original JSON file
- Placed in the same location as original
- Named: tampar_adversarial_<split>.json

NO combined file - keeps adversarial annotations separate.
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
def create_adversarial_annotations(
    original_coco: dict,
    image_root: Path,
    adversarial_rename_map: dict,
) -> dict:
    """
    Create adversarial COCO annotations (separate from original).

    Args:
        original_coco: Original annotations
        image_root: Root directory of images
        adversarial_rename_map: Mapping from original to adversarial filenames

    Returns:
        Adversarial COCO dict
    """
    print("\n" + "=" * 80)
    print("Building Adversarial Annotations")
    print("=" * 80)

    # Get max IDs from original JSON (for unique ID generation)
    max_image_id, max_annotation_id = get_max_ids(original_coco)

    print(f"\nOriginal dataset statistics:")
    print(f"  Images: {len(original_coco['images'])}")
    print(f"  Annotations: {len(original_coco['annotations'])}")
    print(f"  Max image ID: {max_image_id}")
    print(f"  Max annotation ID: {max_annotation_id}")

    # Create adversarial dataset
    adversarial_coco = copy.deepcopy(original_coco)

    if "info" in adversarial_coco:
        adversarial_coco["info"]["description"] = (
            f"Adversarial version of {adversarial_coco['info'].get('description', 'TAMPAR dataset')} "
            f"(FGSM + PGD attacks)"
        )

    # Create annotation lookup by image_id
    annotations_by_image = {}
    for ann in original_coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Track IDs (start from max_id + 1 for uniqueness)
    next_image_id = max_image_id + 1
    next_annotation_id = max_annotation_id + 1

    all_images = []
    all_annotations = []

    fgsm_count = 0
    pgd_count = 0
    missing_count = 0

    print(f"\nProcessing adversarial images...")
    print(f"  Starting image ID: {next_image_id}")
    print(f"  Starting annotation ID: {next_annotation_id}")

    # Process each original image
    for image_info in tqdm.tqdm(
        original_coco["images"], desc="Creating adversarial annotations"
    ):
        original_file_path = image_info["file_name"]

        # Check if this image has an adversarial version
        if original_file_path not in adversarial_rename_map:
            missing_count += 1
            continue

        adversarial_file_path = adversarial_rename_map[original_file_path]

        # Verify the adversarial image exists
        adversarial_full_path = image_root / adversarial_file_path
        if not adversarial_full_path.exists():
            missing_count += 1
            continue

        # Determine attack type from filename
        attack_type = "fgsm" if "_fgsm" in adversarial_file_path else "pgd"

        # Create new image info
        new_image_info = copy.deepcopy(image_info)
        new_image_info["id"] = next_image_id
        new_image_info["file_name"] = adversarial_file_path

        all_images.append(new_image_info)

        # Get annotations for this image
        image_annotations = annotations_by_image.get(image_info["id"], [])

        # Copy annotations
        for annotation in image_annotations:
            new_annotation = copy.deepcopy(annotation)
            new_annotation["id"] = next_annotation_id
            new_annotation["image_id"] = next_image_id
            all_annotations.append(new_annotation)
            next_annotation_id += 1

        next_image_id += 1

        if attack_type == "fgsm":
            fgsm_count += 1
        else:
            pgd_count += 1

    # Update adversarial dict
    adversarial_coco["images"] = all_images
    adversarial_coco["annotations"] = all_annotations

    # Print summary
    print("\n" + "=" * 80)
    print("Adversarial Dataset Summary")
    print("=" * 80)
    print(f"  FGSM images:           {fgsm_count:>5}")
    print(f"  PGD images:            {pgd_count:>5}")
    print(f"  {'-' * 30}")
    print(f"  Total images:          {len(all_images):>5}")
    print(f"  Total annotations:     {len(all_annotations):>5}")
    print(f"  Missing/Skipped:       {missing_count:>5}")

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

    return adversarial_coco


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


def validate_annotations(coco_dict: dict, image_root: Path):
    """
    Validate annotations structure.

    Args:
        coco_dict: COCO annotation dict
        image_root: Root directory for images
    """
    print("\nValidating annotations...")

    # Check for missing adversarial images
    missing_images = []
    for image_info in coco_dict["images"]:
        image_path = image_root / image_info["file_name"]
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
    Main function to create adversarial COCO annotations.
    """
    print("=" * 80)
    print("TAMPAR Adversarial Annotations Generator")
    print("Creates Separate Adversarial JSON Files")
    print("=" * 80)

    # Clean existing adversarial annotations
    clean_adversarial_annotations(IMAGE_ROOT)

    # Check if adversarial rename map exists
    if not ADVERSARIAL_RENAME_MAP.exists():
        print(f"\n❌ ERROR: Adversarial rename map not found:")
        print(f"   {ADVERSARIAL_RENAME_MAP}")
        print(f"\n   Please run adversarial_attack_generator.py first!")
        return

    # Load adversarial rename map
    print(f"\nLoading adversarial rename map...")
    with open(ADVERSARIAL_RENAME_MAP, "r") as f:
        adversarial_rename_map = json.load(f)

    print(f"  Loaded {len(adversarial_rename_map)} image mappings")

    # Find all original annotation files
    annotation_files = list(IMAGE_ROOT.rglob("tampar_sample_*.json"))

    if len(annotation_files) == 0:
        print(
            f"\n❌ ERROR: No annotation files found matching pattern: tampar_sample_*.json"
        )
        return

    print(f"\nFound {len(annotation_files)} annotation file(s):")
    for ann_file in annotation_files:
        print(f"  - {ann_file.name}")

    # Process each annotation file
    for annotation_file in annotation_files:
        print("\n" + "=" * 80)
        print(f"Processing: {annotation_file.name}")
        print("=" * 80)

        # Load original annotations
        print(f"\nLoading original annotations...")
        original_coco = load_coco_annotations(annotation_file)

        # Create adversarial annotations
        adversarial_coco = create_adversarial_annotations(
            original_coco, IMAGE_ROOT, adversarial_rename_map
        )

        # Determine output path (same directory as original)
        # tampar_sample_validation.json -> tampar_adversarial_validation.json
        output_filename = annotation_file.name.replace(
            "tampar_sample", "tampar_adversarial"
        )
        output_path = annotation_file.parent / output_filename

        # Save adversarial annotation file
        print("\n" + "-" * 80)
        print("Saving Adversarial Annotation File")
        print("-" * 80)

        save_coco_annotations(adversarial_coco, output_path)
        validate_annotations(adversarial_coco, IMAGE_ROOT)

    # Final summary
    print("\n" + "=" * 80)
    print("✓ Annotation Generation Complete!")
    print("=" * 80)

    print(f"\nGenerated Files:")
    for annotation_file in annotation_files:
        output_filename = annotation_file.name.replace(
            "tampar_sample", "tampar_adversarial"
        )
        print(f"  - {output_filename}")

    print("\n" + "=" * 80)


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    main()
