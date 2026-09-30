import pandas as pd
import numpy as np
import joblib

from event_data import create_event_features, load_event_data


MODEL_FILE = "models/endpoint_rf.pkl"
DATA_FILE = "data/cert_preprocessed_ml_comparison.csv"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    return create_event_features(df)


def risk_level(score):

    if score >= 80:
        return "HIGH"

    elif score >= 50:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# EXPLANATION
# ============================================================

def explain_event(row):

    reasons = []

    if row["is_after_hours"] == 1:
        reasons.append(
            "Activity occurred outside normal working hours"
        )

    if row["is_http"] == 1:
        reasons.append(
            "HTTP activity detected"
        )

    if row["is_email"] == 1:
        reasons.append(
            "Email activity detected"
        )

    if row["is_device"] == 1:
        reasons.append(
            "Device activity detected"
        )

    if row["resource_present"] == 1:
        reasons.append(
            "Resource access detected"
        )

    if row["has_url"] == 1:
        reasons.append(
            "URL-like activity detected"
        )

    if row["activity_has_number"] == 1:
        reasons.append(
            "Activity contains numeric identifiers"
        )

    if row["is_connect"] == 1:
        reasons.append(
            "Network connection activity detected"
        )

    if row["is_disconnect"] == 1:
        reasons.append(
            "Network disconnection activity detected"
        )

    if not reasons:

        reasons.append(
            "No strong behavioral indicators detected"
        )

    return reasons


# ============================================================
# LOAD DATA
# ============================================================

df = load_event_data(DATA_FILE)

print(
    f"Loaded {len(df)} events."
)


# ============================================================
# CREATE FEATURES
# ============================================================

X = create_features(df)


# ============================================================
# PREDICT
# ============================================================

predictions = model.predict(X)

probabilities = model.predict_proba(X)

suspicious_probability = probabilities[:, 1]


# ============================================================
# RISK SCORE
# ============================================================

risk_scores = (
    suspicious_probability * 100
)

risk_scores = np.clip(
    risk_scores,
    0,
    100
)


# ============================================================
# CREATE RESULTS
# ============================================================

results = df.copy()

results["prediction"] = np.where(
    predictions == 1,
    "Suspicious",
    "Normal"
)

results["risk_score"] = (
    risk_scores.round(2)
)

results["risk_level"] = [
    risk_level(score)
    for score in risk_scores
]


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    "data/predictions.csv",
    index=False
)


print("\nPredictions generated.")

print(
    "\nPrediction distribution:"
)

print(
    results["prediction"].value_counts()
)


print(
    "\nRisk distribution:"
)

print(
    results["risk_level"].value_counts()
)


# ============================================================
# SHOW HIGH-RISK EVENTS
# ============================================================

print(
    "\nTop 10 highest-risk events:"
)

display_columns = [
    "timestamp",
    "user",
    "pc",
    "activity",
    "resource",
    "prediction",
    "risk_score",
    "risk_level"
]

print(
    results
    .sort_values(
        "risk_score",
        ascending=False
    )
    [display_columns]
    .head(10)
    .to_string(index=False)
)


print(
    "\nPredictions saved to:"
)

print(
    "data/predictions.csv"
)