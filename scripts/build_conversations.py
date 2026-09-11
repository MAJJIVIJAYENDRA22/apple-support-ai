import pandas as pd
from pathlib import Path

DATA_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/processed/applesupport_conversations.csv"

# IMPORTANT:
# In this dataset, AppleSupport appears directly in author_id
BRAND = "AppleSupport"

CHUNK_SIZE = 100_000


def normalize_id(value):
    """
    Normalize tweet IDs so values such as:
        119236
        119236.0
        '119236'
        '119236.0'

    all become:
        '119236'
    """

    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("APPLE SUPPORT - CONVERSATION RECONSTRUCTION")
print("=" * 70)


# =========================================================
# PASS 1
# Build customer tweet lookup
# =========================================================

print("\nPASS 1: Building customer tweet lookup...")

customer_tweets = {}

total_customer_tweets = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1
):

    # Customer messages
    inbound = chunk[
        chunk["inbound"] == True
    ]

    for _, row in inbound.iterrows():

        tweet_id = normalize_id(row["tweet_id"])

        if tweet_id is None:
            continue

        customer_tweets[tweet_id] = {
            "customer_tweet_id": tweet_id,
            "customer_author_id": normalize_id(
                row["author_id"]
            ),
            "customer_text": str(row["text"]),
            "customer_created_at": row["created_at"]
        }

    total_customer_tweets += len(inbound)

    print(
        f"Chunk {chunk_number:02d} | "
        f"Customer tweets processed: "
        f"{total_customer_tweets:,} | "
        f"Lookup size: {len(customer_tweets):,}"
    )


print("\nPASS 1 COMPLETE")
print(
    f"Customer tweets in lookup: "
    f"{len(customer_tweets):,}"
)


# =========================================================
# PASS 2
# Connect AppleSupport responses
# =========================================================

print("\n" + "=" * 70)
print("PASS 2: Connecting AppleSupport responses...")
print("=" * 70)


conversation_rows = []

apple_responses = 0
responses_with_parent = 0
matched_conversations = 0


for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1
):

    # IMPORTANT:
    # Do NOT convert author_id to numeric.
    # Brand handles are strings such as "AppleSupport".

    apple = chunk[
        (chunk["inbound"] == False) &
        (chunk["author_id"].astype(str).str.strip() == BRAND)
    ]

    apple_responses += len(apple)

    for _, row in apple.iterrows():

        parent_id = normalize_id(
            row["in_response_to_tweet_id"]
        )

        if parent_id is None:
            continue

        responses_with_parent += 1

        customer = customer_tweets.get(parent_id)

        if customer is None:
            continue

        conversation_rows.append({
            "customer_tweet_id":
                customer["customer_tweet_id"],

            "customer_author_id":
                customer["customer_author_id"],

            "customer_text":
                customer["customer_text"],

            "customer_created_at":
                customer["customer_created_at"],

            "brand":
                BRAND,

            "response_tweet_id":
                normalize_id(row["tweet_id"]),

            "brand_response":
                str(row["text"]),

            "brand_created_at":
                row["created_at"]
        })

        matched_conversations += 1

    print(
        f"Chunk {chunk_number:02d} | "
        f"Apple responses: {apple_responses:,} | "
        f"With parent: {responses_with_parent:,} | "
        f"Matched: {matched_conversations:,}"
    )


# =========================================================
# SAVE
# =========================================================

print("\n" + "=" * 70)
print("SAVING CONVERSATIONS")
print("=" * 70)


if conversation_rows:

    conversations = pd.DataFrame(
        conversation_rows
    )

    conversations = conversations.drop_duplicates(
        subset=[
            "customer_tweet_id",
            "response_tweet_id"
        ]
    )

    conversations.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nFinal conversation pairs: "
        f"{len(conversations):,}"
    )

    print("\nColumns:")

    for column in conversations.columns:
        print(f"  - {column}")

    print("\nExample conversations:")

    print(
        conversations[
            [
                "customer_text",
                "brand_response"
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)

else:

    print("\nERROR: No conversation pairs were created.")


print("\n" + "=" * 70)
print("CONVERSATION RECONSTRUCTION COMPLETE")
print("=" * 70)