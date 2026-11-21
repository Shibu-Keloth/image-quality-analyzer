"""features.py - turn ONE image into a handful of numbers (feature extraction).

Everything else in the project (dataset building, training, the web app)
imports from this file, so the numbers are always computed the same way.
"""
import cv2
import numpy as np

# Sharpness and noise depend on how many pixels the image has, so we measure
# them on a copy whose long side is at most this many pixels.
ANALYSIS_SIZE = 1024

# The model is trained on exactly these columns, in exactly this order.
FEATURE_NAMES = ["sharpness", "brightness", "contrast", "noise", "megapixels"]


def decode_image(raw_bytes):
    """Bytes of a JPG/PNG file -> OpenCV image (BGR numpy array), or None."""
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_image(path):
    """Read an image from disk. Works with spaces/non-English letters in the path."""
    with open(path, "rb") as f:
        return decode_image(f.read())


def estimate_noise(gray):
    """Immerkaer's method: convolve with a mask that cancels smooth image
    structure, so what is left is mostly noise. Returns noise sigma (0-255 scale)."""
    mask = np.array([[1, -2, 1],
                     [-2, 4, -2],
                     [1, -2, 1]], dtype=np.float64)
    response = cv2.filter2D(gray.astype(np.float64), -1, mask)
    response = response[1:-1, 1:-1]               # drop the border pixels
    return float(np.mean(np.abs(response)) * np.sqrt(np.pi / 2) / 6)


def extract_features(img_bgr):
    """Return a dict with the five quality features for one image."""
    height, width = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Make a standard-size copy (only ever shrink, never enlarge).
    long_side = max(height, width)
    if long_side > ANALYSIS_SIZE:
        scale = ANALYSIS_SIZE / long_side
        small = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray

    return {
        # Variance of the Laplacian (edge response): high = crisp, low = blurry
        "sharpness": float(cv2.Laplacian(small, cv2.CV_64F).var()),
        # Average grey level, 0 (black) to 255 (white)
        "brightness": float(gray.mean()),
        # Spread of grey levels: low = flat/washed-out
        "contrast": float(gray.std()),
        # Estimated noise strength (measured on the same standard-size copy)
        "noise": estimate_noise(small),
        # Resolution in millions of pixels, from the ORIGINAL size
        "megapixels": width * height / 1_000_000,
    }

