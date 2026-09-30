import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# 1. LOAD DATA
# =========================================================

FILE = "data/cert_preprocessed_ml_comparison.csv"

df = pd.read_csv(FILE)

print("Original dataset:", df.shape)

# =========================================================
# 2. CREATE TARGET
# =========================================================

label_col = "case id;Cattivi"

df["target"] = (
    df[label_col]
    .fillna("")
    .astype(str)
    .str.split(";")
    .str[-1]
    .replace("", "0")
    .astype(int)
)

print("\nTarget distribution:")
print(df["target"].value_counts())

# =========================================================
# 3. TIMESTAMP FEATURES
# =========================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

df["is_after_hours"] = (
    (df["hour"] < 8) |
    (df["hour"] >= 18)
).astype(int)

# =========================================================
# 4. BEHAVIORAL FEATURES
# =========================================================

df["is_http"] = (
    df["file"].astype(str).str.lower() == "http"
).astype(int)

df["is_email"] = (
    df["file"].astype(str).str.lower() == "email"
).astype(int)

df["is_device"] = (
    df["file"].astype(str).str.lower() == "device"
).astype(int)

df["is_logon"] = (
    df["file"].astype(str).str.lower() == "logon"
).astype(int)

df["activity_length"] = (
    df["activity"]
    .fillna("")
    .astype(str)
    .str.len()
)

df["activity_has_number"] = (
    df["activity"]
    .fillna("")
    .astype(str)
    .str.contains(r"\d")
    .astype(int)
)

df["resource_present"] = (
    df["resource"].notna()
).astype(int)

df["resource_length"] = (
    df["resource"]
    .fillna("")
    .astype(str)
    .str.len()
)

# =========================================================
# 5. IMPORTANT
#
# Do NOT use:
# - user
# - pc
# - case id
#
# They can make the dataset artificially easy.
# =========================================================

features = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_after_hours",
    "is_http",
    "is_email",
    "is_device",
    "is_logon",
    "activity_length",
    "activity_has_number",
    "resource_present",
    "resource_length"
]

X = df[features].fillna(0)
y = df["target"]

print("\nFeatures:")
print(features)

# =========================================================
# 6. TRAIN ONLY ON NORMAL EVENTS
# =========================================================

X_normal = X[y == 0]

print("\nNormal training events:", len(X_normal))

print(
    "Suspicious events used for evaluation:",
    (y == 1).sum()
)

# =========================================================
# 7. ISOLATION FOREST
# =========================================================

model = IsolationForest(
    n_estimators=300,
    contamination=0.20,
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Isolation Forest...")

model.fit(X_normal)

print("Training complete.")

# =========================================================
# 8. PREDICTIONS
# =========================================================

raw_prediction = model.predict(X)

# Isolation Forest:
#
#  1  = normal
# -1  = anomaly
#

y_pred = np.where(
    raw_prediction == -1,
    1,
    0
)

# =========================================================
# 9. ANOMALY SCORE
# =========================================================

# Higher decision_function = more normal
# Lower decision_function = more anomalous

decision_scores = model.decision_function(X)

# Convert to a simple 0-100 anomaly score.
#
# Lower decision score → higher risk.

risk_score = np.clip(
    (0.5 - decision_scores) * 100,
    0,
    100
)

# =========================================================
# 10. METRICS
# =========================================================

tn, fp, fn, tp = confusion_matrix(
    y,
    y_pred
).ravel()

precision = precision_score(
    y,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    zero_division=0
)

fpr = fp / (fp + tn)

fnr = fn / (fn + tp)

# =========================================================
# 11. RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("ISOLATION FOREST SECURITY EVALUATION")
print("=" * 60)

print(f"\nPrecision      : {precision:.4f}")
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
        y,
        y_pred,
        target_names=[
            "Normal",
            "Suspicious"
        ],
        zero_division=0
    )
)

# =========================================================
# 12. RISK DISTRIBUTION
# =========================================================

results = pd.DataFrame({
    "target": y,
    "prediction": y_pred,
    "risk_score": risk_score
})

print("\nAverage risk score:")

print(
    results.groupby("target")["risk_score"]
    .mean()
)

print("\nRisk score statistics:")

print(
    results.groupby("target")["risk_score"]
    .describe()
)

# =========================================================
# 13. SAVE MODEL
# =========================================================

joblib.dump(
    {
        "model": model,
        "features": features
    },
    "models/isolation_forest.pkl"
)

print("\nModel saved to:")

print(
    "models/isolation_forest.pkl"
)

# =========================================================
# 14. SAVE RESULTS
# =========================================================

output = df.copy()

output["prediction"] = y_pred

output["risk_score"] = (
    risk_score.round(2)
)

output.to_csv(
    "data/anomaly_results.csv",
    index=False
)

print("\nResults saved to:")

print(
    "data/anomaly_results.csv"
)