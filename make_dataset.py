"""make_dataset.py - build a LABELLED image dataset from your good photos.

Idea: start from sharp, well-exposed photos and damage copies of them in a
controlled way. Because WE applied the damage, we know the label.
    Good    = untouched (or almost untouched) photo
    Average = moderate damage (one defect)
    Poor    = severe damage (one defect)
"""
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from features import load_image

ORIGINALS_DIR = Path("data/originals")
DATASET_DIR = Path("data/dataset")
LABELS_CSV = Path("data/labels.csv")

WORK_SIZE = 1024        # every original is shrunk to this long side first
MIN_ORIGINAL_SIDE = 800  # smaller photos are skipped (they can't be "Good")
SEED = 42

# For each defect: the (min, max) strength used for Average and for Poor.
DEFECTS = {
    "blur":         {"Average": (0.6, 1.0),   "Poor": (1.8, 4.5)},    # Gaussian sigma
    "noise":        {"Average": (10, 18),     "Poor": (25, 45)},      # noise sigma
    "dark":         {"Average": (0.55, 0.70), "Poor": (0.20, 0.40)},  # brightness x
    "bright":       {"Average": (1.35, 1.55), "Poor": (1.9, 2.6)},    # brightness x
    "low_contrast": {"Average": (0.50, 0.65), "Poor": (0.20, 0.40)},  # contrast x
    "low_res":      {"Average": (450, 650),   "Poor": (120, 300)},    # long side, px
}


def to_float(img):
    return img.astype(np.float32)


def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_defect(img, defect, strength, rng):
    """Return a damaged copy of img."""
    if defect == "blur":
        return cv2.GaussianBlur(img, (0, 0), strength)
    if defect == "noise":
        return to_uint8(to_float(img) + rng.normal(0, strength, img.shape))
    if defect in ("dark", "bright"):
        return to_uint8(to_float(img) * strength)
    if defect == "low_contrast":
        mean = img.mean()
        return to_uint8(mean + strength * (to_float(img) - mean))
    if defect == "low_res":
        scale = strength / max(img.shape[:2])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    raise ValueError(defect)


def shrink_to(img, long_side):
    scale = long_side / max(img.shape[:2])
    if scale >= 1:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def save_jpeg(img, path):
    ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    path.write_bytes(buffer.tobytes())


def main():
    rng_py = random.Random(SEED)
    rng_np = np.random.default_rng(SEED)

    # Start fresh so old files never mix with new ones.
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for label in ("Good", "Average", "Poor"):
        (DATASET_DIR / label).mkdir(parents=True)

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    originals = sorted({p for pat in patterns for p in ORIGINALS_DIR.glob(pat)})
    print(f"Found {len(originals)} original images")

    rows = []
    used = 0
    for path in originals:
        img = load_image(path)
        if img is None or max(img.shape[:2]) < MIN_ORIGINAL_SIDE:
            print(f"  skipped (unreadable or too small): {path.name}")
            continue
        img = shrink_to(img, WORK_SIZE)
        source = path.stem
        used += 1

        # 3 Good: the original + two almost-identical versions
        for k in range(3):
            if k == 0:
                variant, defect = img, "none"
            else:
                factor = rng_py.uniform(0.92, 1.08)
                variant = apply_defect(img, "dark", factor, rng_np)
                defect = "tiny_brightness_shift"
            name = f"{source}__Good_{k}.jpg"
            save_jpeg(variant, DATASET_DIR / "Good" / name)
            rows.append([f"data/dataset/Good/{name}", "Good", source, defect])

        # 3 Average and 3 Poor, each with a DIFFERENT randomly chosen defect
        for label in ("Average", "Poor"):
            for k, defect in enumerate(rng_py.sample(list(DEFECTS), 3)):
                low, high = DEFECTS[defect][label]
                strength = rng_py.uniform(low, high)
                variant = apply_defect(img, defect, strength, rng_np)
                name = f"{source}__{label}_{defect}_{k}.jpg"
                save_jpeg(variant, DATASET_DIR / label / name)
                rows.append([f"data/dataset/{label}/{name}", label, source, defect])

    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "source", "defect"])
        writer.writerows(rows)

    print(f"Used {used} originals -> {len(rows)} labelled images")
    print(f"Labels saved to {LABELS_CSV}")


if __name__ == "__main__":
    main()