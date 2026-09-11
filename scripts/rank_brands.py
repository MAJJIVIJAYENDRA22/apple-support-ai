import pandas as pd
from collections import defaultdict

DATA_PATH = "data/raw/twcs.csv"
CHUNK_SIZE = 100_000

# ---------------------------------------------------------
# First pass:
# Find all outbound/support accounts
# ---------------------------------------------------------

print("=" * 70)
print("HIVER - BRAND RANKING")
print("=" * 70)

print("\nPASS 1: Finding support accounts...\n")

support_accounts = set()

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1
):

    outbound = chunk[chunk["inbound"] == False]

    support_accounts.update(
        outbound["author_id"]
        .dropna()
        .astype(str)
    )

    print(
        f"Chunk {chunk_number:02d} | "
        f"Support accounts found: {len(support_accounts)}"
    )


print("\nTotal support accounts found:", len(support_accounts))


# ---------------------------------------------------------
# Second pass:
# Calculate statistics for each support account
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PASS 2: Calculating brand statistics...")
print("=" * 70)

stats = defaultdict(
    lambda: {
        "outbound": 0,
        "linked_customer_messages": 0,
        "unique_customers": set(),
        "linked_tweets": set(),
    }
)

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1
):

    outbound = chunk[chunk["inbound"] == False].copy()

    for _, row in outbound.iterrows():

        brand = str(row["author_id"])

        # Ignore accounts that are not identified as support accounts
        if brand not in support_accounts:
            continue

        stats[brand]["outbound"] += 1

        # A brand response normally points back to
        # the customer tweet it is answering.
        parent_id = row["in_response_to_tweet_id"]

        if pd.notna(parent_id):

            stats[brand]["linked_customer_messages"] += 1
            stats[brand]["linked_tweets"].add(str(parent_id))

    print(
        f"Chunk {chunk_number:02d} processed"
    )


# ---------------------------------------------------------
# Build ranking
# ---------------------------------------------------------

results = []

for brand, data in stats.items():

    linked = len(data["linked_tweets"])
    outbound = data["outbound"]

    if outbound == 0:
        continue

    linkage_rate = linked / outbound

    results.append({
        "brand": brand,
        "support_responses": outbound,
        "linked_customer_messages": linked,
        "linkage_rate": linkage_rate,
    })


results.sort(
    key=lambda x: x["linked_customer_messages"],
    reverse=True
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 30 BRAND CANDIDATES")
print("=" * 70)

print(
    f"{'Rank':<6}"
    f"{'Brand':<25}"
    f"{'Responses':<15}"
    f"{'Customer Msgs':<18}"
    f"{'Link %':<10}"
)

print("-" * 75)

for rank, item in enumerate(results[:30], start=1):

    print(
        f"{rank:<6}"
        f"{item['brand']:<25}"
        f"{item['support_responses']:<15,}"
        f"{item['linked_customer_messages']:<18,}"
        f"{item['linkage_rate'] * 100:<10.1f}"
    )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output = pd.DataFrame(results)

output.to_csv(
    "data/processed/brand_ranking.csv",
    index=False
)

print("\nSaved:")
print("data/processed/brand_ranking.csv")

print("\n" + "=" * 70)
print("BRAND RANKING COMPLETE")
print("=" * 70)