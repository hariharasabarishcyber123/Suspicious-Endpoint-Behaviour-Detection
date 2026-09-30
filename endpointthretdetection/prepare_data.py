import pandas as pd

from event_data import create_event_features, load_event_data

INPUT_FILE = "data/cert_preprocessed_ml_comparison.csv"
OUTPUT_FILE = "data/endpoint_features.csv"

# --------------------------------------------------
# Load
# --------------------------------------------------

df = load_event_data(INPUT_FILE)

print("Original shape:", df.shape)

# --------------------------------------------------
# Target
# --------------------------------------------------

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

features = create_event_features(df)

features["target"] = df["target"].values

# --------------------------------------------------
# Save
# --------------------------------------------------

features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPrepared dataset saved to:")
print(OUTPUT_FILE)

print("\nFinal shape:")
print(features.shape)

print("\nFeatures:")
print(list(features.columns))

print("\nTarget distribution:")
print(features["target"].value_counts())

print("\nTarget percentage:")
print(
    (features["target"].value_counts(normalize=True) * 100)
    .round(2)
)