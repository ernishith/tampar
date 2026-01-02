"""
adversarial_attack_generator.py (COMPLETE VERSION)

Generates adversarial attacks on ALL parcel images with two modes:
1. Random Noise Mode (fast, no model needed)
2. Gradient-Based Targeted Mode (more realistic adversarial examples)

Attack pattern:
- FGSM attack on odd-indexed images (1st, 3rd, 5th, ...)
- PGD attack on even-indexed images (2nd, 4th, 6th, ...)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

# ============================================================================
# Configuration
# ============================================================================
IMAGE_ROOT = ROOT / "data" / "tampar_sample"
ATTACK_DIR = IMAGE_ROOT / "adversarial_attacks"
ATTACK_DIR.mkdir(exist_ok=True)

# Attack parameters
EPSILON_FGSM = 8.0 / 255.0  # Perturbation budget for FGSM
EPSILON_PGD = 8.0 / 255.0  # Perturbation budget for PGD
ALPHA_PGD = 2.0 / 255.0  # Step size for PGD
PGD_ITERATIONS = 10  # Number of PGD iterations

# Attack mode: Choose which attack strategy to use
USE_GRADIENT_BASED = True  # False = Random noise, True = Gradient-based targeted


# ============================================================================
# Helper Functions
# ============================================================================
def image_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert BGR image to normalized tensor [0, 1].

    Args:
        image_bgr: Input image in BGR format (HxWx3)

    Returns:
        Tensor (1x3xHxW) in [0, 1] range
    """
    # BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    image_float = image_rgb.astype(np.float32) / 255.0

    # Convert to tensor: HxWx3 -> 3xHxW -> 1x3xHxW
    tensor = torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0)

    return tensor


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor back to BGR image.

    Args:
        tensor: Input tensor (1x3xHxW) in [0, 1] range

    Returns:
        BGR image (HxWx3) uint8
    """
    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 1)

    # 1x3xHxW -> HxWx3
    image_rgb = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Scale to [0, 255]
    image_rgb = (image_rgb * 255).astype(np.uint8)

    # RGB to BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr


# ============================================================================
# Mode 1: Random Noise Attacks (No Model Required)
# ============================================================================
def fgsm_noise(image_tensor: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Add FGSM-style random noise to image.

    This version doesn't use gradients - just adds random signed noise.

    Args:
        image_tensor: Input image (1x3xHxW) in [0, 1]
        epsilon: Perturbation magnitude

    Returns:
        Adversarial image tensor
    """
    # Generate random noise with same shape
    noise = torch.randn_like(image_tensor)

    # Take sign of noise (FGSM-style)
    noise_sign = torch.sign(noise)

    # Add scaled noise
    adversarial = image_tensor + epsilon * noise_sign

    # Clamp to valid range
    adversarial = torch.clamp(adversarial, 0, 1)

    return adversarial


def pgd_noise(
    image_tensor: torch.Tensor, epsilon: float, alpha: float, iterations: int
) -> torch.Tensor:
    """
    Add PGD-style iterative random noise to image.

    This version doesn't use gradients - iteratively adds random noise.

    Args:
        image_tensor: Input image (1x3xHxW) in [0, 1]
        epsilon: Max perturbation budget
        alpha: Step size per iteration
        iterations: Number of steps

    Returns:
        Adversarial image tensor
    """
    original = image_tensor.clone()
    adversarial = image_tensor.clone()

    for _ in range(iterations):
        # Generate random noise
        noise = torch.randn_like(adversarial)
        noise_sign = torch.sign(noise)

        # Take step
        adversarial = adversarial + alpha * noise_sign

        # Project back to epsilon ball around original
        perturbation = adversarial - original
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        adversarial = original + perturbation

        # Clamp to valid pixel range
        adversarial = torch.clamp(adversarial, 0, 1)

    return adversarial


# ============================================================================
# Mode 2: Gradient-Based Targeted Attacks
# ============================================================================
def create_target_pattern(
    image_tensor: torch.Tensor, pattern_type: str = "inverted"
) -> torch.Tensor:
    """
    Create a target pattern for gradient-based attacks.

    Args:
        image_tensor: Input image tensor (1x3xHxW)
        pattern_type: Type of target pattern
            - "inverted": 1.0 - image (inverted colors)
            - "gray": Grayscale image
            - "random": Random target image
            - "shifted": Color-shifted image

    Returns:
        Target tensor
    """
    if pattern_type == "inverted":
        # Invert colors: target is complement of original
        target = 1.0 - image_tensor

    elif pattern_type == "gray":
        # Convert to grayscale
        gray = (
            0.299 * image_tensor[:, 0:1, :, :]
            + 0.587 * image_tensor[:, 1:2, :, :]
            + 0.114 * image_tensor[:, 2:3, :, :]
        )
        target = gray.repeat(1, 3, 1, 1)

    elif pattern_type == "random":
        # Random target image
        target = torch.rand_like(image_tensor)

    elif pattern_type == "shifted":
        # Shift color channels: RGB -> BRG
        target = torch.cat(
            [
                image_tensor[:, 2:3, :, :],  # B -> R
                image_tensor[:, 0:1, :, :],  # R -> G
                image_tensor[:, 1:2, :, :],  # G -> B
            ],
            dim=1,
        )
    else:
        # Default: inverted
        target = 1.0 - image_tensor

    return target.detach()


def fgsm_attack_targeted(
    image_tensor: torch.Tensor, epsilon: float, target_pattern: str = "inverted"
) -> torch.Tensor:
    """
    FGSM attack that tries to change pixel values towards a target pattern.

    Uses gradient-based optimization to create adversarial examples.

    Args:
        image_tensor: Input image (1x3xHxW) in [0, 1]
        epsilon: Perturbation budget
        target_pattern: Type of target pattern (see create_target_pattern)

    Returns:
        Adversarial image tensor
    """
    # Ensure gradient tracking
    image_tensor = image_tensor.clone().detach()
    image_tensor.requires_grad = True

    # Create target
    target = create_target_pattern(image_tensor, target_pattern)

    # Compute loss (MSE between current and target)
    # We want to minimize distance to target = maximize change
    loss = F.mse_loss(image_tensor, target)

    # Compute gradient
    loss.backward()

    # FGSM step: move in gradient direction
    gradient_sign = image_tensor.grad.sign()
    adversarial = image_tensor + epsilon * gradient_sign
    adversarial = torch.clamp(adversarial, 0, 1).detach()

    return adversarial


def pgd_attack_targeted(
    image_tensor: torch.Tensor,
    epsilon: float,
    alpha: float,
    iterations: int,
    target_pattern: str = "inverted",
) -> torch.Tensor:
    """
    PGD attack that tries to change pixel values towards a target pattern.

    Iterative gradient-based attack that is stronger than FGSM.

    Args:
        image_tensor: Input image (1x3xHxW) in [0, 1]
        epsilon: Perturbation budget
        alpha: Step size per iteration
        iterations: Number of iterations
        target_pattern: Type of target pattern (see create_target_pattern)

    Returns:
        Adversarial image tensor
    """
    original = image_tensor.clone().detach()
    adversarial = image_tensor.clone().detach()

    # Create target once
    target = create_target_pattern(original, target_pattern)

    for iteration in range(iterations):
        adversarial.requires_grad = True

        # Compute loss
        loss = F.mse_loss(adversarial, target)

        # Backward pass
        loss.backward()

        # Get gradient sign
        gradient_sign = adversarial.grad.sign()

        # Take step
        adversarial = adversarial.detach() + alpha * gradient_sign

        # Project to epsilon ball
        perturbation = torch.clamp(adversarial - original, -epsilon, epsilon)
        adversarial = torch.clamp(original + perturbation, 0, 1)

    return adversarial


# ============================================================================
# Main Processing Function
# ============================================================================
def generate_adversarial_images(
    image_root: Path,
    output_dir: Path,
    use_gradient_based: bool = False,
    target_pattern: str = "inverted",
):
    """
    Generate adversarial attacks on ALL images alternately:
    - Odd indices (0, 2, 4, ...) -> FGSM
    - Even indices (1, 3, 5, ...) -> PGD

    Args:
        image_root: Root directory containing images
        output_dir: Directory to save adversarial images
        use_gradient_based: If True, use gradient-based targeted attacks.
                           If False, use simple random noise attacks.
        target_pattern: Target pattern for gradient-based attacks
    """
    # Find all JPG images
    rename_dict = {}
    image_paths = sorted(image_root.rglob("*.jpg"))
    print(f"Found {len(image_paths)} JPG images")

    if len(image_paths) == 0:
        print("No images found! Check IMAGE_ROOT path.")
        return

    # Display attack mode
    attack_mode = "Gradient-based targeted" if use_gradient_based else "Random noise"
    print(f"\nAttack mode: {attack_mode}")
    if use_gradient_based:
        print(f"Target pattern: {target_pattern}")

    print(f"\nAttack strategy:")
    print(f"  - Odd-indexed images (1st, 3rd, 5th, ...) -> FGSM")
    print(f"  - Even-indexed images (2nd, 4th, 6th, ...) -> PGD")
    print(f"  - Total images to attack: {len(image_paths)}")

    # Select attack functions based on mode
    if use_gradient_based:
        fgsm_fn = lambda img: fgsm_attack_targeted(img, EPSILON_FGSM, target_pattern)
        pgd_fn = lambda img: pgd_attack_targeted(
            img, EPSILON_PGD, ALPHA_PGD, PGD_ITERATIONS, target_pattern
        )
    else:
        fgsm_fn = lambda img: fgsm_noise(img, EPSILON_FGSM)
        pgd_fn = lambda img: pgd_noise(img, EPSILON_PGD, ALPHA_PGD, PGD_ITERATIONS)

    # Process images
    fgsm_count = 0
    pgd_count = 0

    for idx, image_path in enumerate(
        tqdm.tqdm(image_paths, desc="Generating adversarial attacks")
    ):
        # Read image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Warning: Could not read {image_path}")
            continue

        # Convert to tensor
        image_tensor = image_to_tensor(image_bgr)

        # Determine attack type based on index
        # Odd indices (0, 2, 4, ...) -> FGSM
        # Even indices (1, 3, 5, ...) -> PGD
        if idx % 2 == 0:  # Odd-indexed (0-based: 0, 2, 4 = 1st, 3rd, 5th)
            attack_type = "fgsm"
            adversarial_tensor = fgsm_fn(image_tensor)
            fgsm_count += 1
        else:  # Even-indexed (0-based: 1, 3, 5 = 2nd, 4th, 6th)
            attack_type = "pgd"
            adversarial_tensor = pgd_fn(image_tensor)
            pgd_count += 1

        # Convert back to image
        adversarial_bgr = tensor_to_image(adversarial_tensor)

        # Save with suffix maintaining directory structure
        relative_path = image_path.relative_to(image_root)
        output_path = (
            output_dir / relative_path.parent / f"{image_path.stem}_{attack_type}.jpg"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rename_dict[f"{image_path.stem}.jpg"] = f"{image_path.stem}_{attack_type}.jpg"

        cv2.imwrite(str(output_path), adversarial_bgr)

    print(f"\n{'='*80}")
    print(f"Attack Summary:")
    print(f"  Attack mode:               {attack_mode}")
    print(f"  Total images processed:    {len(image_paths)}")
    print(f"  FGSM attacked images:      {fgsm_count}")
    print(f"  PGD attacked images:       {pgd_count}")
    print(f"  Total adversarial images:  {fgsm_count + pgd_count}")
    print(f"  Adversarial images saved to: {output_dir}")
    print(f"{'='*80}")

    print("Writing rename Json file")

    output_path = output_dir / "adversarial_rename_map.json"

    with open(output_path, "w") as f:
        json.dump(rename_dict, f, indent=2)


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Adversarial Attack Generator - Complete Version")
    print("=" * 80)
    print(f"Image Root: {IMAGE_ROOT}")
    print(f"Output Directory: {ATTACK_DIR}")
    print(f"\nAttack Parameters:")
    print(f"  FGSM Epsilon: {EPSILON_FGSM:.4f} ({EPSILON_FGSM * 255:.1f}/255)")
    print(f"  PGD Epsilon:  {EPSILON_PGD:.4f} ({EPSILON_PGD * 255:.1f}/255)")
    print(f"  PGD Alpha:    {ALPHA_PGD:.4f} ({ALPHA_PGD * 255:.1f}/255)")
    print(f"  PGD Iterations: {PGD_ITERATIONS}")
    print("=" * 80)

    # Configure attack mode here
    # USE_GRADIENT_BASED = False  # Set at top of file
    TARGET_PATTERN = "inverted"  # Options: "inverted", "gray", "random", "shifted"

    print(
        f"\nAttack Mode: {'Gradient-based' if USE_GRADIENT_BASED else 'Random noise'}"
    )
    if USE_GRADIENT_BASED:
        print(f"Target Pattern: {TARGET_PATTERN}")
    print()

    generate_adversarial_images(
        IMAGE_ROOT,
        ATTACK_DIR,
        use_gradient_based=USE_GRADIENT_BASED,
        target_pattern=TARGET_PATTERN,
    )

    print(
        "\n✓ Done! Run create_adversarial_annotations.py to generate COCO annotations."
    )
