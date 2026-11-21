"""build_features.py - measure every labelled image and save one table.

Input : data/labels.csv   (file, label, source, defect)
Output: data/features.csv (the same columns + the 5 measured features)
"""
import pandas as pd

from features import FEATURE_NAMES, extract_features, load_image

labels = pd.read_csv("data/labels.csv")

rows = []
for i, row in labels.iterrows():
    img = load_image(row["file"])
    measurements = extract_features(img)
    rows.append({**row.to_dict(), **measurements})
    if (i + 1) % 50 == 0:
        print(f"  measured {i + 1}/{len(labels)}")

table = pd.DataFrame(rows)
table.to_csv("data/features.csv", index=False)

print(f"\nSaved data/features.csv with {len(table)} rows")
print("\nAverage of each feature per label:")
print(table.groupby("label")[FEATURE_NAMES].mean().round(2))