
from pathlib import Path
import pandas as pd
import numpy as np

file = Path("results/paderborn_labeled_features.csv")
df = pd.read_csv(file)

feature_columns = [
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "kurtosis",
    "crest_factor"
]

print("=" * 55)
print("FEATURE DATASET VALIDATION")
print("=" * 55)

print("Total rows:", len(df))
print("Unique source files:", df["file"].nunique())
print("Unique bearings:", df["bearing_id"].nunique())
print("Missing values:", int(df.isna().sum().sum()))

print("\nNon-numeric or infinite feature values:")
for column in feature_columns:
    values = pd.to_numeric(df[column], errors="coerce")
    invalid = int((~np.isfinite(values)).sum())
    print(f"{column:16s}: {invalid}")

print("\nWindows per bearing:")
print(df.groupby("bearing_id")["window"].count().to_string())

print("\nBearings per condition:")
print(df.groupby("condition")["bearing_id"].nunique().to_string())

print("\nFeature statistics:")
print(df[feature_columns].describe().round(4).to_string())

print("\nValidation checks complete.")
