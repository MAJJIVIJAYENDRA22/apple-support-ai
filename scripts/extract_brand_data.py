import pandas as pd
from pathlib import Path

DATA_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/processed/applesupport_tweets.csv"

BRAND = "AppleSupport"
CHUNK_SIZE = 100_000

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

total_processed = 0
total_brand_rows = 0
brand_chunks = []

print("=" * 70)
print("APPLE SUPPORT DATA EXTRACTION")
print("=" * 70)

print(f"\nTarget brand: {BRAND}")
print("Reading the dataset in chunks...\n")

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1
):

    total_processed += len(chunk)

    # AppleSupport's outbound/support messages
    brand_rows = chunk[
        (chunk["inbound"] == False) &
        (chunk["author_id"].astype(str) == BRAND)
    ].copy()

    if not brand_rows.empty:
        brand_chunks.append(brand_rows)
        total_brand_rows += len(brand_rows)

    print(
        f"Chunk {chunk_number:02d} | "
        f"Processed: {total_processed:,} | "
        f"AppleSupport rows: {total_brand_rows:,}"
    )

print("\nCombining AppleSupport data...")

if brand_chunks:

    brand_df = pd.concat(
        brand_chunks,
        ignore_index=True
    )

    brand_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"\nTotal dataset rows scanned: {total_processed:,}")
    print(f"AppleSupport rows extracted: {len(brand_df):,}")

    print("\nSaved to:")
    print(OUTPUT_PATH)

else:

    print("\nERROR: No AppleSupport rows were found.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)