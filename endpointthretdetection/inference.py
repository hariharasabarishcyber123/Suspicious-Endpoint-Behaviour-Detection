import joblib
import pandas as pd

from event_data import create_event_features, load_event_data


MODEL_PATH = "models/endpoint_rf.pkl"
DATA_PATH = "data/cert_preprocessed_ml_comparison.csv"

model = joblib.load(MODEL_PATH)
event_data = load_event_data(DATA_PATH)
USER_COUNTS = (
    event_data["user"].fillna("").astype(str).value_counts().to_dict()
)
PC_COUNTS = (
    event_data["pc"].fillna("").astype(str).value_counts().to_dict()
)


def predict_event(event):
    timestamp = pd.to_datetime(event["timestamp"], errors="coerce", format="mixed")

    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp: {event['timestamp']}")

    features = create_event_features(
        pd.DataFrame([event]),
        user_counts=USER_COUNTS,
        pc_counts=PC_COUNTS,
    )

    if hasattr(model, "feature_names_in_"):
        features = features.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    risk_score = round(probability * 100, 2)
    risk_level = (
        "HIGH" if risk_score >= 70
        else "MEDIUM" if risk_score >= 40
        else "LOW"
    )

    return {
        "prediction": prediction,
        "status": "SUSPICIOUS" if prediction == 1 else "NORMAL",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "features": features,
    }


def explain_event(event, result):
    explanations = []
    timestamp = pd.to_datetime(event["timestamp"], errors="coerce", format="mixed")
    activity = str(event.get("activity", "")).strip().lower()
    resource = str(event.get("resource", "")).strip()

    if not pd.isna(timestamp) and (timestamp.hour < 8 or timestamp.hour >= 18):
        explanations.append(
            "Event occurred outside the configured normal working hours."
        )

    activity_explanations = {
        "logon": "User logon activity detected.",
        "logoff": "User logoff activity detected.",
        "connect": "Endpoint connection activity detected.",
        "disconnect": "Endpoint disconnection activity detected.",
        "http": "HTTP/web-related activity detected.",
        "email": "Email-related activity detected.",
        "device": "Device activity detected.",
        "send email": "Email sending activity detected.",
        "consult http": "HTTP consultation activity detected.",
    }

    if activity in activity_explanations:
        explanations.append(activity_explanations[activity])

    if resource:
        explanations.append("A specific resource was associated with this event.")

    if not explanations:
        explanations.append(
            "The machine-learning model classified this event using the "
            "engineered endpoint behaviour features."
        )

    return explanations