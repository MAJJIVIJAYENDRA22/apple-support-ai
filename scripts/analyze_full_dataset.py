import pandas as pd
from collections import Counter

DATA_PATH = "data/raw/twcs.csv"
CHUNK_SIZE = 100_000

total_rows = 0
inbound_count = 0
outbound_count = 0

author_counter = Counter()

print("=" * 60)
print("HIVER - FULL DATASET ANALYSIS")
print("=" * 60)

print("\nReading dataset in chunks...")

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1
):

    rows = len(chunk)

    total_rows += rows

    inbound_count += (chunk["inbound"] == True).sum()
    outbound_count += (chunk["inbound"] == False).sum()

    # Count authors
    author_counter.update(
        chunk["author_id"]
        .dropna()
        .astype(str)
    )

    print(
        f"Processed chunk {chunk_number}: "
        f"{rows:,} rows | "
        f"Total: {total_rows:,}"
    )

print("\n" + "=" * 60)
print("FINAL DATASET STATISTICS")
print("=" * 60)

print(f"\nTotal rows: {total_rows:,}")
print(f"Inbound/customer messages: {inbound_count:,}")
print(f"Outbound/brand responses: {outbound_count:,}")

print("\nTop authors:")
for author, count in author_counter.most_common(30):
    print(f"{author}: {count:,}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)