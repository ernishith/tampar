import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def summarize_pair(clean_path: Path, adv_path: Path) -> dict:
    a = read_rgb(clean_path).astype(np.int16)
    b = read_rgb(adv_path).astype(np.int16)
    if a.shape != b.shape:
        return {
            "clean": clean_path.as_posix(),
            "adv": adv_path.as_posix(),
            "ok": False,
            "reason": f"shape_mismatch {a.shape} vs {b.shape}",
        }

    diff = np.abs(a - b).astype(np.float32)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    mae = mean_abs  # same as mean absolute difference in pixel space
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))

    return {
        "clean": clean_path.as_posix(),
        "adv": adv_path.as_posix(),
        "ok": True,
        "max_abs_diff_0_255": max_abs,
        "mean_abs_diff_0_255": mean_abs,
        "mse": mse,
        "psnr_db": psnr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_root", type=str, required=True)
    ap.add_argument("--adv_root", type=str, required=True)
    ap.add_argument("--adv_type", type=str, choices=["fgsm", "pgd"], required=True)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    adv_root = Path(args.adv_root)

    # We use clean uvmap_pred as anchor, and look for matching *_<adv>_uvmap_pred in adv_root
    clean_uvs = sorted(clean_root.rglob("*_uvmap_pred.png"))

    rows = []
    for p in clean_uvs[: args.limit]:
        rel = p.relative_to(clean_root)
        # Find corresponding adversarial uvmap_pred:
        # clean stem: "...something..._uvmap_pred"
        clean_stem = p.stem  # e.g., id_00_xxx_uvmap_pred
        prefix = clean_stem.replace("_uvmap_pred", "")
        adv_path = adv_root / f"{prefix}_{args.adv_type}_uvmap_pred.png"

        if not adv_path.exists():
            rows.append(
                {
                    "clean": p.as_posix(),
                    "adv": adv_path.as_posix(),
                    "ok": False,
                    "reason": "adv_missing",
                }
            )
            continue

        rows.append(summarize_pair(p, adv_path))

    df = pd.DataFrame(rows)
    out_csv = Path("uvmap_pred_diff_report.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv.resolve()}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
