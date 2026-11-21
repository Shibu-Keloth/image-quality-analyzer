"""rules.py - the RULE-BASED part (hand-picked thresholds, NO machine learning).

Two jobs:
  1. rule_based_label(): a baseline we compare the trained model against.
  2. suggestions(): plain-English tips shown in the app.
"""

# (bad, good): at or beyond "bad" the score is 0, at or beyond "good" it is 1.
# These numbers are hand-picked by us. A different person would pick different ones.
RAMPS = {
    "sharpness":  (20, 200),
    "contrast":   (15, 35),
    "noise":      (8, 2),         # lower is better, so bad > good
    "megapixels": (0.1, 0.4),
}
DARK_SIDE = (40, 90)              # brightness: (bad, good) for too dark
BRIGHT_SIDE = (230, 170)          # brightness: (bad, good) for too bright

WEIGHTS = {"sharpness": 0.30, "noise": 0.20, "brightness": 0.20,
           "contrast": 0.15, "megapixels": 0.15}


def ramp(value, bad, good):
    """Linear score between 0 (bad) and 1 (good), clipped to that range."""
    t = (value - bad) / (good - bad)
    return min(1.0, max(0.0, t))


def feature_scores(f):
    """Score each feature from 0 to 1. f is a dict of measured features."""
    scores = {name: ramp(f[name], *RAMPS[name]) for name in RAMPS}
    scores["brightness"] = min(ramp(f["brightness"], *DARK_SIDE),
                               ramp(f["brightness"], *BRIGHT_SIDE))
    return scores


def rule_based_score(f):
    scores = feature_scores(f)
    return sum(WEIGHTS[name] * scores[name] for name in WEIGHTS)


def rule_based_label(f):
    score = rule_based_score(f)
    if score >= 0.85:
        return "Good"
    if score >= 0.60:
        return "Average"
    return "Poor"


def suggestions(f):
    """A list of practical tips, based on any feature scoring below 0.5."""
    scores = feature_scores(f)
    tips = []
    if scores["sharpness"] < 0.5:
        tips.append("Looks soft or blurry: hold the camera steady, tap to focus, "
                    "and use a faster shutter speed.")
    if scores["noise"] < 0.5:
        tips.append("Looks noisy/grainy: shoot in better light, lower the ISO, "
                    "or apply noise reduction.")
    if scores["brightness"] < 0.5:
        if f["brightness"] < 128:
            tips.append("Looks too dark: add light or raise the exposure.")
        else:
            tips.append("Looks overexposed: lower the exposure or avoid direct glare.")
    if scores["contrast"] < 0.5:
        tips.append("Looks flat/washed-out: increase contrast or shoot in clearer light.")
    if scores["megapixels"] < 0.5:
        tips.append("Resolution is low: use a higher camera setting and avoid "
                    "heavy cropping or shrinking.")
    if not tips:
        tips.append("No obvious technical problems found by these five measurements.")
    return tips
