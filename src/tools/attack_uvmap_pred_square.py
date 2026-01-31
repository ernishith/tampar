import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.tampering.compare import CompareType, compute_uvmap_similarity


def read_rgb(p: Path) -> np.ndarray:
    bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(p: Path, rgb: np.ndarray):
    p.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(p), bgr)


def objective_from_similarity(sim_dict: dict, metric: str) -> float:
    vals = []
    for _, metrics in sim_dict.items():
        if metric in metrics:
            vals.append(metrics[metric])
    if not vals:
        return -1e9
    return float(np.mean(vals))


def compute_objective(
    gt_rgb: np.ndarray,
    cand_rgb: np.ndarray,
    compare_type: str,
    metric: str,
    direction: str,
) -> float:
    sim = compute_uvmap_similarity(
        gt_rgb,
        cand_rgb,
        output_path=Path("."),  # not used when visualize=False
        compare_type=compare_type,
        visualize=False,
    )
    v = objective_from_similarity(sim, metric)

    # direction="max" means maximize the metric (useful for mae/hog)
    # direction="min" means minimize the metric (useful for ssim/msssim/cwssim)
    if direction == "max":
        return v
    if direction == "min":
        return -v
    raise ValueError("direction must be 'max' or 'min'")


def project_linf(x: np.ndarray, x0: np.ndarray, eps: int) -> np.ndarray:
    # x, x0 are uint8 RGB
    x_f = x.astype(np.int16)
    x0_f = x0.astype(np.int16)
    lo = x0_f - eps
    hi = x0_f + eps
    y = np.clip(x_f, lo, hi)
    y = np.clip(y, 0, 255).astype(np.uint8)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_uvmap", type=str, required=True)
    ap.add_argument("--start_uvmap_pred", type=str, required=True)
    ap.add_argument("--out_uvmap_pred", type=str, required=True)

    ap.add_argument(
        "--compare_type", type=str, default=CompareType.PLAIN
    )  # start with PLAIN
    ap.add_argument(
        "--metric",
        type=str,
        default="mae",
        choices=["mae", "hog", "ssim", "msssim", "cwssim"],
    )
    ap.add_argument(
        "--direction", type=str, default="max", choices=["max", "min"]
    )  # mae/hog: max, ssim-family: min

    ap.add_argument("--eps", type=int, default=16)  # L_inf in [0..255]
    ap.add_argument("--iters", type=int, default=2000)  # queries
    ap.add_argument("--square", type=int, default=32)  # starting square size
    ap.add_argument("--square_min", type=int, default=4)  # shrink schedule lower bound
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_csv", type=str, default="attack_log.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    gt = read_rgb(Path(args.gt_uvmap)).astype(np.uint8)
    x0 = read_rgb(Path(args.start_uvmap_pred)).astype(np.uint8)
    x = x0.copy()

    best = compute_objective(gt, x, args.compare_type, args.metric, args.direction)
    print("Initial objective:", best)

    H, W = x.shape[:2]
    log_path = Path(args.log_csv)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "best_objective", "square_size"])

        for t in range(args.iters):
            # simple schedule: shrink square over time
            frac = t / max(1, args.iters - 1)
            s = int(round(args.square * (1 - frac) + args.square_min * frac))
            s = max(args.square_min, min(s, H, W))

            x_try = x.copy()

            y = int(rng.integers(0, H - s + 1))
            z = int(rng.integers(0, W - s + 1))

            # Square update: push pixels toward boundary of L_inf ball
            # choose random sign update
            sign = rng.choice([-1, 1], size=(s, s, 3)).astype(np.int16)
            step = max(1, args.eps)  # strong step
            patch = x_try[y : y + s, z : z + s, :].astype(np.int16) + sign * step
            x_try[y : y + s, z : z + s, :] = np.clip(patch, 0, 255).astype(np.uint8)

            # project to L_inf ball around x0
            x_try = project_linf(x_try, x0, args.eps)

            val = compute_objective(
                gt, x_try, args.compare_type, args.metric, args.direction
            )

            if val > best:
                best = val
                x = x_try
                print(f"iter={t} improved objective={best} (square={s})")

            if t % 10 == 0:
                writer.writerow([t, best, s])

    write_rgb(Path(args.out_uvmap_pred), x)
    print("Saved attacked uvmap_pred:", args.out_uvmap_pred)
    print("Log saved:", log_path.resolve())
    print("Final objective:", best)


if __name__ == "__main__":
    main()
