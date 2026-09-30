import pandas as pd

file_path = "data/cert_preprocessed_ml_comparison.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== UNIQUE VALUES ==========")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].nunique())

print("\n========== POSSIBLE LABEL COLUMNS ==========")
for column in df.columns:
    print(column, "=>", df[column].value_counts().head(10))