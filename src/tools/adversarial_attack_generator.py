"""
adversarial_attack_generator.py

Adds adversarial noise to parcel images without requiring keypoint detection.
- FGSM-style noise to every 2nd image
- PGD-style noise to every 4th image
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

import cv2
import numpy as np
import torch
import tqdm

# ============================================================================
# Configuration
# ============================================================================
IMAGE_ROOT = ROOT / "data" / "tampar_sample"
ATTACK_DIR = IMAGE_ROOT / "adversarial_attacks"
ATTACK_DIR.mkdir(exist_ok=True)

# Attack parameters
EPSILON_FGSM = (
    8.0 / 255.0
)  # Perturbation budget for FGSM (8 pixel values in [0,1] scale)
EPSILON_PGD = 8.0 / 255.0  # Perturbation budget for PGD
ALPHA_PGD = 2.0 / 255.0  # Step size for PGD
PGD_ITERATIONS = 10  # Number of PGD iterations


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
    image_rgb = tensor.squeeze(0).permute(1, 2, 0).numpy()

    # Scale to [0, 255]
    image_rgb = (image_rgb * 255).astype(np.uint8)

    # RGB to BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr


# ============================================================================
# Adversarial Attack Functions (No Model Required)
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
# Alternative: Gradient-based attacks using a simple target
# ============================================================================
def fgsm_attack_targeted(image_tensor: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    FGSM attack that tries to change pixel values towards a target pattern.

    Args:
        image_tensor: Input image (1x3xHxW)
        epsilon: Perturbation budget

    Returns:
        Adversarial image
    """
    image_tensor.requires_grad = True

    # Create a target (e.g., inverted image or random target)
    target = 1.0 - image_tensor.detach()  # Inverted image as target

    # Compute loss (MSE between current and target)
    loss = torch.nn.functional.mse_loss(image_tensor, target)

    # Compute gradient
    loss.backward()

    # FGSM step
    gradient_sign = image_tensor.grad.sign()
    adversarial = image_tensor + epsilon * gradient_sign
    adversarial = torch.clamp(adversarial, 0, 1).detach()

    return adversarial


def pgd_attack_targeted(
    image_tensor: torch.Tensor, epsilon: float, alpha: float, iterations: int
) -> torch.Tensor:
    """
    PGD attack that tries to change pixel values towards a target pattern.

    Args:
        image_tensor: Input image (1x3xHxW)
        epsilon: Perturbation budget
        alpha: Step size
        iterations: Number of iterations

    Returns:
        Adversarial image
    """
    original = image_tensor.clone()
    adversarial = image_tensor.clone()

    # Create target
    target = 1.0 - original  # Inverted image

    for _ in range(iterations):
        adversarial.requires_grad = True

        # Compute loss
        loss = torch.nn.functional.mse_loss(adversarial, target)

        # Backward
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
# Main Processing
# ============================================================================
def generate_adversarial_images(
    image_root: Path, output_dir: Path, use_gradient_based: bool = False
):
    """
    Generate adversarial noise on images.

    Args:
        image_root: Directory containing images
        output_dir: Output directory
        use_gradient_based: If True, use gradient-based attacks with targets.
                           If False, use simple random noise attacks.
    """
    # Find all JPG images
    image_paths = sorted(image_root.rglob("*.jpg"))
    print(f"Found {len(image_paths)} JPG images")

    if len(image_paths) == 0:
        print("No images found! Check IMAGE_ROOT path.")
        return

    # Choose attack functions
    if use_gradient_based:
        fgsm_fn = fgsm_attack_targeted
        pgd_fn = pgd_attack_targeted
        attack_mode = "Gradient-based (with targets)"
    else:
        fgsm_fn = fgsm_noise
        pgd_fn = pgd_noise
        attack_mode = "Random noise"

    print(f"Attack mode: {attack_mode}")

    # Process images
    fgsm_count = 0
    pgd_count = 0

    for idx, image_path in enumerate(
        tqdm.tqdm(image_paths, desc="Adding adversarial noise")
    ):
        # Read image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Warning: Could not read {image_path}")
            continue

        # Convert to tensor
        image_tensor = image_to_tensor(image_bgr)

        # Determine attack type
        attack_type = None

        if (idx + 1) % 4 == 0:  # Every 4th image -> PGD
            attack_type = "pgd"
            if use_gradient_based:
                adversarial_tensor = pgd_fn(
                    image_tensor, EPSILON_PGD, ALPHA_PGD, PGD_ITERATIONS
                )
            else:
                adversarial_tensor = pgd_fn(
                    image_tensor, EPSILON_PGD, ALPHA_PGD, PGD_ITERATIONS
                )
            pgd_count += 1

        elif (idx + 1) % 2 == 0:  # Every 2nd image (not 4th) -> FGSM
            attack_type = "fgsm"
            adversarial_tensor = fgsm_fn(image_tensor, EPSILON_FGSM)
            fgsm_count += 1
        else:
            continue  # Skip

        # Convert back to image
        adversarial_bgr = tensor_to_image(adversarial_tensor)

        # Save with suffix
        relative_path = image_path.relative_to(image_root)
        output_path = (
            output_dir / relative_path.parent / f"{image_path.stem}_{attack_type}.jpg"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), adversarial_bgr)

    print(f"\n{'='*80}")
    print(f"Total FGSM attacked images: {fgsm_count}")
    print(f"Total PGD attacked images: {pgd_count}")
    print(f"Adversarial images saved to: {output_dir}")
    print(f"{'='*80}")


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Simple Adversarial Noise Generator (No Keypoint Detection)")
    print("=" * 80)
    print(f"Image Root: {IMAGE_ROOT}")
    print(f"Output Directory: {ATTACK_DIR}")
    print(f"FGSM Epsilon: {EPSILON_FGSM:.4f}")
    print(
        f"PGD Epsilon: {EPSILON_PGD:.4f}, Alpha: {ALPHA_PGD:.4f}, Iterations: {PGD_ITERATIONS}"
    )
    print("=" * 80)

    # Choose attack mode:
    # False = Simple random noise (no gradients needed)
    # True = Gradient-based with target pattern
    USE_GRADIENT_BASED = True

    generate_adversarial_images(
        IMAGE_ROOT, ATTACK_DIR, use_gradient_based=USE_GRADIENT_BASED
    )

    print("\nDone!")
