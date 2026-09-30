import pandas as pd

RAW_FILE = "data/cert_preprocessed_ml_comparison.csv"

df = pd.read_csv(RAW_FILE)

# --------------------------------------------------
# Create target exactly as we did before
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

print("=" * 60)
print("DATASET DIAGNOSTICS")
print("=" * 60)

# --------------------------------------------------
# 1. Target distribution
# --------------------------------------------------

print("\nTARGET DISTRIBUTION")
print(df["target"].value_counts())

# --------------------------------------------------
# 2. Target by hour
# --------------------------------------------------

df["hour"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
).dt.hour

print("\nSUSPICIOUS RATE BY HOUR")

hour_stats = (
    df.groupby("hour")["target"]
    .agg(["count", "sum", "mean"])
    .sort_index()
)

hour_stats["mean"] = (
    hour_stats["mean"] * 100
).round(2)

print(hour_stats)

# --------------------------------------------------
# 3. Target by month
# --------------------------------------------------

df["month"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
).dt.month

print("\nSUSPICIOUS RATE BY MONTH")

month_stats = (
    df.groupby("month")["target"]
    .agg(["count", "sum", "mean"])
)

month_stats["mean"] = (
    month_stats["mean"] * 100
).round(2)

print(month_stats)

# --------------------------------------------------
# 4. Resource presence
# --------------------------------------------------

df["resource_present"] = (
    df["resource"].notna()
).astype(int)

print("\nSUSPICIOUS RATE BY RESOURCE PRESENCE")

print(
    pd.crosstab(
        df["resource_present"],
        df["target"],
        normalize="index"
    ).round(3)
)

# --------------------------------------------------
# 5. File type
# --------------------------------------------------

print("\nSUSPICIOUS RATE BY FILE TYPE")

file_stats = (
    df.groupby("file")["target"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

file_stats["mean"] = (
    file_stats["mean"] * 100
).round(2)

print(file_stats.head(20))

# --------------------------------------------------
# 6. Activity
# --------------------------------------------------

print("\nSUSPICIOUS RATE BY ACTIVITY")

activity_stats = (
    df.groupby("activity")["target"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

activity_stats["mean"] = (
    activity_stats["mean"] * 100
).round(2)

print(activity_stats.head(20))

# --------------------------------------------------
# 7. User
# --------------------------------------------------

print("\nSUSPICIOUS RATE BY USER")

user_stats = (
    df.groupby("user")["target"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

user_stats["mean"] = (
    user_stats["mean"] * 100
).round(2)

print(user_stats)

# --------------------------------------------------
# 8. PC
# --------------------------------------------------

print("\nSUSPICIOUS RATE BY PC")

pc_stats = (
    df.groupby("pc")["target"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

pc_stats["mean"] = (
    pc_stats["mean"] * 100
).round(2)

print(pc_stats.head(30))

# --------------------------------------------------
# 9. Label vs case ID presence
# --------------------------------------------------

print("\nCASE ID PRESENCE")

print(
    df[label_col].notna()
    .value_counts()
)

print("\nCASE ID EXAMPLES")

print(
    df.loc[
        df[label_col].notna(),
        label_col
    ].value_counts().head(20)
)

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)