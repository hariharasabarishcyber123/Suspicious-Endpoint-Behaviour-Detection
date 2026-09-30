import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==================================================
# Load original prepared data
# ==================================================

df = pd.read_csv("data/endpoint_features.csv")

print("Dataset:", df.shape)

# ==================================================
# Features
# ==================================================

drop_features = [
    "target",
    "user_event_count",
    "pc_event_count"
]

X = df.drop(columns=drop_features)
y = df["target"]

# ==================================================
# IMPORTANT:
# Use user as grouping variable.
#
# We need the original user information, so load
# the original CERT CSV.
# ==================================================

raw = pd.read_csv(
    "data/cert_preprocessed_ml_comparison.csv"
)

groups = raw["user"].astype(str)

print("\nUnique users:", groups.nunique())

# ==================================================
# Group-based train/test split
# ==================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "\nTraining suspicious:",
    y_train.sum()
)

print(
    "Testing suspicious:",
    y_test.sum()
)

# ==================================================
# Random Forest
# ==================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")

# ==================================================
# Predictions
# ==================================================

y_pred = model.predict(X_test)

# ==================================================
# Metrics
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()

fpr = fp / (fp + tn)

fnr = fn / (fn + tp)

# ==================================================
# Results
# ==================================================

print("\n" + "=" * 55)
print("LEAKAGE-RESISTANT SECURITY EVALUATION")
print("=" * 55)

print(f"\nAccuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"False Positive : {fpr:.4f}")
print(f"False Negative : {fnr:.4f}")
print(f"Detection Rate : {recall:.4f}")

print("\nConfusion Matrix:")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Suspicious"
        ],
        zero_division=0
    )
)

# ==================================================
# Feature importance
# ==================================================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nTop Features:")

print(
    importance.head(10).to_string(
        index=False
    )
)

# ==================================================
# Save secure model
# ==================================================

joblib.dump(
    model,
    "models/endpoint_rf_secure.pkl"
)

print(
    "\nSecure model saved to:"
)

print(
    "models/endpoint_rf_secure.pkl"
)