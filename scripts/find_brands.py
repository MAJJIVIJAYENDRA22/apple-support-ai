import pandas as pd
from collections import Counter

DATA_PATH = "data/raw/twcs.csv"
CHUNK_SIZE = 100_000

# Count how often each author sends outbound/support messages
outbound_authors = Counter()

# Count inbound/outbound messages
total_rows = 0
inbound_count = 0
outbound_count = 0

print("=" * 70)
print("HIVER - BRAND DISCOVERY")
print("=" * 70)

print("\nScanning dataset in chunks...")
print("This may take a few minutes.\n")

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1
):
    total_rows += len(chunk)

    inbound_count += int((chunk["inbound"] == True).sum())
    outbound_count += int((chunk["inbound"] == False).sum())

    # Outbound tweets are support/brand responses
    outbound = chunk[chunk["inbound"] == False]

    # Count authors of outbound messages
    authors = (
        outbound["author_id"]
        .dropna()
        .astype(str)
    )

    outbound_authors.update(authors)

    print(
        f"Chunk {chunk_number:02d} | "
        f"Rows processed: {total_rows:,}"
    )

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Total tweets:       {total_rows:,}")
print(f"Inbound tweets:     {inbound_count:,}")
print(f"Outbound tweets:    {outbound_count:,}")

print("\n" + "=" * 70)
print("TOP SUPPORT/BRAND ACCOUNTS")
print("=" * 70)

print(
    f"{'Rank':<6}"
    f"{'Author ID':<20}"
    f"{'Outbound Tweets':<20}"
)

print("-" * 50)

for rank, (author, count) in enumerate(
    outbound_authors.most_common(50),
    start=1
):
    print(
        f"{rank:<6}"
        f"{author:<20}"
        f"{count:<20,}"
    )

print("\n" + "=" * 70)
print("DISCOVERY COMPLETE")
print("=" * 70)