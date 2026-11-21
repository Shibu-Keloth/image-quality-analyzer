"""try_it.py - see how the features react when we damage a photo.

Usage:  python try_it.py data\\originals\\YOUR_PHOTO.jpg
Prints a table of features and saves 4 test images in the test_images folder.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from features import extract_features, load_image

img = load_image(sys.argv[1])

# Shrink so the long side is 1000 px (about the size the model was trained on)
scale = 1000 / max(img.shape[:2])
if scale < 1:
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

rng = np.random.default_rng(0)
versions = {
    "good": img,
    "blur": cv2.GaussianBlur(img, (0, 0), 3),
    "dark": np.clip(img.astype(np.float32) * 0.3, 0, 255).astype(np.uint8),
    "noise": np.clip(img + rng.normal(0, 30, img.shape), 0, 255).astype(np.uint8),
}

out = Path("test_images")
out.mkdir(exist_ok=True)
rows = {}
for name, version in versions.items():
    cv2.imwrite(str(out / f"test_{name}.jpg"), version)
    rows[name] = extract_features(version)

print(pd.DataFrame(rows).T.round(2))
print("\nSaved 4 images in the test_images folder.")
