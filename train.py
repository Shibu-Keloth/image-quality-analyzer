"""train.py - train, compare and save the quality classifier."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES
from rules import rule_based_label

LABELS = ["Good", "Average", "Poor"]

# ---- 1. Load the table of measurements -------------------------------------
data = pd.read_csv("data/features.csv")
X = data[FEATURE_NAMES]          # inputs  (5 numbers per image)
y = data["label"]                # answer  (Good / Average / Poor)
groups = data["source"]          # which ORIGINAL photo each image came from
print(f"{len(data)} images from {groups.nunique()} original photos")

# ---- 2. Split into train / test, keeping each original photo together -------
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"train: {len(X_train)} images | test: {len(X_test)} images\n")

# ---- 3. The rule-based baseline (no training needed) -------------------------
rule_test = [rule_based_label(row) for row in X_test.to_dict("records")]
rule_all = [rule_based_label(row) for row in X.to_dict("records")]
print(f"Rule-based baseline accuracy : {accuracy_score(y_test, rule_test):.3f} "
      f"(test set) | {accuracy_score(y, rule_all):.3f} (all data)")

# ---- 4. Two real machine-learning models ------------------------------------
models = {
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
cv = GroupKFold(n_splits=5)
for name, model in models.items():
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cv_scores = cross_val_score(model, X, y, groups=groups, cv=cv)
    print(f"{name:20s} test accuracy: {test_acc:.3f} | "
          f"5-fold CV: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ---- 5. Look closely at the Random Forest -----------------------------------
forest = models["Random Forest"]
forest_pred = forest.predict(X_test)

print("\nClassification report (Random Forest, test set):")
print(classification_report(y_test, forest_pred, labels=LABELS, zero_division=0))

print("Confusion matrix (rows = true label, columns = predicted):")
matrix = confusion_matrix(y_test, forest_pred, labels=LABELS)
print(pd.DataFrame(matrix, index=LABELS, columns=LABELS))

print("\nFeature importance:")
importance = pd.Series(forest.feature_importances_, index=FEATURE_NAMES)
print(importance.sort_values(ascending=False).round(3))

# ---- 6. Retrain on ALL the data and save for the app --------------------------
final_model = RandomForestClassifier(n_estimators=200, random_state=42)
final_model.fit(X, y)
Path("models").mkdir(exist_ok=True)
joblib.dump({"model": final_model, "features": FEATURE_NAMES},
            "models/quality_model.joblib")
print("\nSaved models/quality_model.joblib")
