import pandas as pd

DATA_PATH = "data/raw/twcs.csv"

print("=" * 60)
print("HIVER - TWITTER CUSTOMER SUPPORT DATASET")
print("=" * 60)

# Read only the first 1,000 rows
df = pd.read_csv(DATA_PATH, nrows=1000)

print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")

print("\nSample size:", len(df))

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nInbound / Outbound:")
print(df["inbound"].value_counts(dropna=False))

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)