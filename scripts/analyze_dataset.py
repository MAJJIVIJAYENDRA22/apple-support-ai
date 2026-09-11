import pandas as pd
import re

DATA_PATH = "data/raw/tweets.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("\n==============================")
print("DATASET OVERVIEW")
print("==============================")

print("Total rows:", len(df))
print("Columns:", list(df.columns))

print("\n==============================")
print("INBOUND / OUTBOUND")
print("==============================")

print(df["inbound"].value_counts(dropna=False))

print("\n==============================")
print("MISSING VALUES")
print("==============================")

print(df.isna().sum())

print("\n==============================")
print("SAMPLE TWEETS")
print("==============================")

print(df[["inbound", "text"]].head(10).to_string(index=False))

print("\n==============================")
print("COMMON @HANDLES")
print("==============================")

# Extract Twitter-style handles
handles = []

for text in df["text"].dropna().astype(str):
    found = re.findall(r"@[A-Za-z0-9_]+", text)
    handles.extend(found)

handle_counts = pd.Series(handles).value_counts()

print(handle_counts.head(30))