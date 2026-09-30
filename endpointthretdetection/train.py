import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# ==================================================
# 1. Load prepared dataset
# ==================================================

DATA_FILE = "data/endpoint_features.csv"
MODEL_FILE = "models/endpoint_rf.pkl"

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

# ==================================================
# 2. Separate features and target
# ==================================================

X = df.drop(columns=["target"])
y = df["target"]

print("\nFeature count:", X.shape[1])

print("\nClass distribution:")
print(y.value_counts())

# ==================================================
# 3. Train / test split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==================================================
# 4. Random Forest
# ==================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training complete.")

# ==================================================
# 5. Predictions
# ==================================================

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# ==================================================
# 6. Security metrics
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

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

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

detection_rate = recall

# ==================================================
# 7. Display results
# ==================================================

print("\n" + "=" * 50)
print("SECURITY MODEL EVALUATION")
print("=" * 50)

print(f"\nAccuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"False Positive : {fpr:.4f}")
print(f"False Negative : {fnr:.4f}")
print(f"Detection Rate : {detection_rate:.4f}")

print("\nConfusion Matrix:")
print(
    f"TN = {tn}"
)
print(
    f"FP = {fp}"
)
print(
    f"FN = {fn}"
)
print(
    f"TP = {tp}"
)

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
# 8. Feature importance
# ==================================================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(
    importance.head(10).to_string(index=False)
)

# ==================================================
# 9. Save model
# ==================================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\nModel saved to:")
print(MODEL_FILE)