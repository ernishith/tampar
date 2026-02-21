import sys
from pathlib import Path

from torch import name

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

#!/usr/bin/env python3
import argparse

import cv2
import numpy as np

from src.tampering.compare import METRICS, CompareType, apply_homogenization

# metric fns live in src.tampering.metrics but compare.py uses globals()[...]
from src.tampering.metrics import (
    compute_cwssim,
    compute_hog,
    compute_lpips,
    compute_mae,
    compute_mse,
    compute_msssim,
    compute_sift,
    compute_ssim,
)
from src.tampering.parcel import PATCH_ORDER
from src.tampering.utils import get_side_surface_patches


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path.as_posix(), bgr)


def clamp_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def compute_uvmap_similarity_cached(
    gt_patches: list[np.ndarray],
    gt_valid_mask: np.ndarray,
    cand_uvmap_rgb: np.ndarray,
    compare_type: str,
) -> dict:
    """
    Same logic as compute_uvmap_similarity(), but:
    - caches GT patches and GT 'mean<250' mask
    - still extracts candidate patches every call (candidate changes each iter)
    """
    cand_patches = get_side_surface_patches(cand_uvmap_rgb)
    results = {}

    for i, (p1, p2) in enumerate(zip(gt_patches, cand_patches)):
        if not gt_valid_mask[i]:
            continue
        if np.mean(p2) >= 250:
            continue

        patch1, patch2 = apply_homogenization(compare_type, p1, p2)

        metrics = {}
        for metric in METRICS:
            compute_metric = globals()[f"compute_{metric}"]
            if compare_type != CompareType.SIMSAC:
                metrics[metric] = compute_metric(
                    patch1.astype(np.float32), patch2.astype(np.float32)
                )
            else:
                metrics[metric] = 0.5 * (
                    compute_metric(patch1, np.zeros_like(patch2))
                    + compute_metric(np.zeros_like(patch1), patch2)
                )
        results[PATCH_ORDER[i]] = metrics

    return results


def agg_metric(res: dict, metric: str, agg: str) -> float:
    vals = []
    for _, md in res.items():
        if metric in md:
            vals.append(float(md[metric]))
    if not vals:
        return 0.0
    if agg == "mean":
        return float(np.mean(vals))
    if agg == "sum":
        return float(np.sum(vals))
    if agg == "max":
        return float(np.max(vals))
    raise ValueError(f"Unknown agg: {agg}")


def simsac_objective_cached(
    gt_patches, gt_valid_mask, cand_rgb, metric="mae", agg="mean"
) -> float:
    res = compute_uvmap_similarity_cached(
        gt_patches=gt_patches,
        gt_valid_mask=gt_valid_mask,
        cand_uvmap_rgb=cand_rgb,
        compare_type=CompareType.SIMSAC,
    )
    return agg_metric(res, metric=metric, agg=agg)


def square_like_attack_linf(
    gt_patches,
    gt_valid_mask,
    x0_rgb: np.ndarray,
    eps: int,
    iters: int,
    block_min: int,
    block_max: int,
    metric: str,
    agg: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)

    x0 = x0_rgb.astype(np.int16)
    delta = np.zeros_like(x0, dtype=np.int16)

    best = clamp_uint8(x0 + delta)
    best_score = simsac_objective_cached(
        gt_patches, gt_valid_mask, best, metric=metric, agg=agg
    )

    h, w, c = x0.shape

    for t in range(iters):
        bh = int(rng.integers(block_min, block_max + 1))
        bw = int(rng.integers(block_min, block_max + 1))
        y0 = int(rng.integers(0, max(1, h - bh + 1)))
        x00 = int(rng.integers(0, max(1, w - bw + 1)))

        sign = rng.choice([-1, 1], size=(1, 1, c)).astype(np.int16)
        step = int(rng.integers(1, max(2, eps // 2 + 1)))

        cand_delta = delta.copy()
        cand_delta[y0 : y0 + bh, x00 : x00 + bw, :] = (
            cand_delta[y0 : y0 + bh, x00 : x00 + bw, :] + sign * step
        )

        cand_delta = np.clip(cand_delta, -eps, eps)
        cand = clamp_uint8(x0 + cand_delta)

        cand_score = simsac_objective_cached(
            gt_patches, gt_valid_mask, cand, metric=metric, agg=agg
        )

        # We MINIMIZE objective (for metric=mae this hides tampering evidence)
        if cand_score < best_score:
            delta = cand_delta
            best = cand
            best_score = cand_score

    return best, {"best_score": float(best_score), "eps": eps, "iters": iters}


def find_gt_uvmap(uvmap_root: Path, parcel_id: int) -> Path | None:
    # matches compute_similarity_scores.py: f"{id:02d}uvmap.png" [file:175]
    p = uvmap_root / f"{str(parcel_id).zfill(2)}uvmap.png"
    return p if p.exists() else None


def parse_parcel_id_from_filename(name: str) -> int | None:
    # adjust if your naming differs
    try:
        return int(name[:2])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=str, required=True)
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--uvmap_root", type=str, required=True)

    ap.add_argument("--eps", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--block_min", type=int, default=8)
    ap.add_argument("--block_max", type=int, default=64)
    ap.add_argument(
        "--metric",
        type=str,
        default="mae",
        choices=["mae", "msssim", "ssim", "cwssim", "hog"],
    )
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "sum", "max"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    uv_root = Path(args.uvmap_root)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def should_skip(p: Path) -> bool:
        name = p.name.lower()
        return "uvmap" in name

    paths = [
        p
        for p in in_root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts and not should_skip(p)
    ]

    paths.sort()

    for idx, p in enumerate(paths):
        rel = p.relative_to(in_root)
        out_p = (out_root / rel).with_name(f"{rel.stem}_sqr{rel.suffix}")

        parcel_id = parse_parcel_id_from_filename(p.name)
        if parcel_id is None:
            continue

        gt_path = find_gt_uvmap(uv_root, parcel_id)
        if gt_path is None:
            continue

        x0 = load_rgb(p)
        gt = load_rgb(gt_path)

        # ---- CACHED invariants (per image/parcel) ----
        gt_patches = get_side_surface_patches(gt)
        gt_valid_mask = np.array([np.mean(pp) < 250 for pp in gt_patches], dtype=bool)

        adv, info = square_like_attack_linf(
            gt_patches=gt_patches,
            gt_valid_mask=gt_valid_mask,
            x0_rgb=x0,
            eps=args.eps,
            iters=args.iters,
            block_min=args.block_min,
            block_max=args.block_max,
            metric=args.metric,
            agg=args.agg,
            seed=args.seed + idx,
        )
        save_rgb(out_p, adv)
        print(
            f"[{idx+1}/{len(paths)}] {rel} parcel={parcel_id} best_score={info['best_score']:.6f}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
