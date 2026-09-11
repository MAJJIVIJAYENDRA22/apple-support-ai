import pandas as pd
import re
from collections import Counter

DATA_PATH = "data/processed/applesupport_conversations.csv"

print("=" * 70)
print("APPLE SUPPORT - INTENT DISCOVERY")
print("=" * 70)

# ---------------------------------------------------------
# Load conversation data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print(f"\nTotal conversation pairs: {len(df):,}")

print("\nColumns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# Basic text cleaning
# ---------------------------------------------------------

def clean_text(text):
    text = str(text)

    # Remove Twitter usernames
    text = re.sub(r"@\w+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_customer_text"] = df["customer_text"].apply(clean_text)


# ---------------------------------------------------------
# Keyword-based issue exploration
# ---------------------------------------------------------

intent_keywords = {

    "ios_update": [
        "ios",
        "update",
        "upgrade",
        "install update",
        "ios 11",
        "ios 12",
        "ios 13",
        "ios 14",
        "ios 15",
        "ios 16",
        "ios 17"
    ],

    "iphone_ipad_device": [
        "iphone",
        "ipad",
        "ipod",
        "device",
        "phone"
    ],

    "mac_macos": [
        "mac",
        "macbook",
        "macos",
        "imac",
        "apple computer"
    ],

    "icloud": [
        "icloud",
        "icloud storage",
        "icloud account"
    ],

    "apple_id_account": [
        "apple id",
        "account",
        "password",
        "login",
        "sign in",
        "locked"
    ],

    "app_store": [
        "app store",
        "download app",
        "purchase app",
        "app"
    ],

    "itunes_music": [
        "itunes",
        "music",
        "song",
        "album",
        "apple music"
    ],

    "payment_billing": [
        "payment",
        "charged",
        "charge",
        "billing",
        "refund",
        "subscription",
        "money"
    ],

    "technical_issue": [
        "not working",
        "doesn't work",
        "doesnt work",
        "error",
        "problem",
        "issue",
        "broken",
        "crash",
        "crashing",
        "failed"
    ],

    "wifi_connectivity": [
        "wifi",
        "wi-fi",
        "internet",
        "bluetooth",
        "connection",
        "connect"
    ],

    "battery_power": [
        "battery",
        "charging",
        "charge",
        "charger",
        "power"
    ],

    "support_contact": [
        "help",
        "support",
        "contact",
        "customer service",
        "call",
        "dm"
    ]
}


# ---------------------------------------------------------
# Intent keyword coverage
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("INTENT KEYWORD COVERAGE")
print("=" * 70)

intent_counts = {}

for intent, keywords in intent_keywords.items():

    mask = df["clean_customer_text"].str.lower().apply(
        lambda text: any(
            keyword in text
            for keyword in keywords
        )
    )

    count = int(mask.sum())

    intent_counts[intent] = count


# ---------------------------------------------------------
# Sort intents by frequency
# ---------------------------------------------------------

sorted_intents = sorted(
    intent_counts.items(),
    key=lambda x: x[1],
    reverse=True
)


print("\nPotential issue categories:\n")

for rank, (intent, count) in enumerate(
    sorted_intents,
    start=1
):

    percentage = (
        count / len(df) * 100
        if len(df) > 0
        else 0
    )

    print(
        f"{rank:02d}. "
        f"{intent:<25} "
        f"{count:>8,} "
        f"({percentage:>5.1f}%)"
    )


# ---------------------------------------------------------
# Representative examples
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("REPRESENTATIVE CUSTOMER EXAMPLES")
print("=" * 70)


# IMPORTANT:
# sorted_intents contains:
# (intent_name, count)
#
# Therefore we retrieve the keywords from
# intent_keywords using the intent name.

for intent, count in sorted_intents[:10]:

    keywords = intent_keywords[intent]

    mask = df["clean_customer_text"].str.lower().apply(
        lambda text: any(
            keyword in text
            for keyword in keywords
        )
    )

    examples = (
        df.loc[mask, "customer_text"]
        .head(3)
        .tolist()
    )

    print("\n")
    print("-" * 70)
    print(f"INTENT: {intent}")
    print(f"MATCHED ROWS: {count:,}")
    print("-" * 70)

    if not examples:
        print("\nNo examples found.")
        continue

    for i, example in enumerate(
        examples,
        start=1
    ):

        print(f"\nExample {i}:")
        print(example)


# ---------------------------------------------------------
# Most common words
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("COMMON WORDS")
print("=" * 70)


stop_words = {
    "the",
    "and",
    "to",
    "a",
    "of",
    "is",
    "it",
    "in",
    "for",
    "on",
    "my",
    "i",
    "this",
    "that",
    "with",
    "you",
    "me",
    "have",
    "has",
    "was",
    "are",
    "be",
    "but",
    "not",
    "can",
    "do",
    "we",
    "please",
    "apple",
    "support"
}


words = []

for text in df["clean_customer_text"]:

    tokens = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        text.lower()
    )

    words.extend(
        token
        for token in tokens
        if token not in stop_words
    )


word_counts = Counter(words)


print()

for word, count in word_counts.most_common(50):

    print(
        f"{word:<25} {count:>8,}"
    )


# ---------------------------------------------------------
# Completion
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("INTENT DISCOVERY COMPLETE")
print("=" * 70)