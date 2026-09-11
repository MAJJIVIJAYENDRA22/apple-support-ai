import pandas as pd
import re
from pathlib import Path

INPUT_PATH = "data/processed/applesupport_conversations.csv"
OUTPUT_PATH = "data/processed/intent_dataset.csv"

print("=" * 70)
print("APPLE SUPPORT - INTENT DATASET PREPARATION")
print("=" * 70)

# ---------------------------------------------------------
# Load conversations
# ---------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print(f"\nLoaded conversations: {len(df):,}")

# ---------------------------------------------------------
# Validate columns
# ---------------------------------------------------------

required_columns = ["customer_text", "brand_response"]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ---------------------------------------------------------
# Clean text
# ---------------------------------------------------------

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove Twitter usernames
    text = re.sub(r"@\w+", " ", text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["customer_text"] = df["customer_text"].apply(clean_text)
df["brand_response"] = df["brand_response"].apply(clean_text)

# ---------------------------------------------------------
# Remove invalid rows
# ---------------------------------------------------------

df = df[
    (df["customer_text"].str.len() > 5) &
    (df["brand_response"].str.len() > 5)
].copy()

print(f"Valid conversations: {len(df):,}")

# ---------------------------------------------------------
# Intent classification
# ---------------------------------------------------------

intent_keywords = {

    "ios_update": [
        "ios update",
        "ios upgrade",
        "update ios",
        "software update",
        "ios"
    ],

    "device_issue": [
        "iphone",
        "ipad",
        "ipod",
        "device",
        "phone"
    ],

    "mac_macos": [
        "macbook",
        "macos",
        "imac",
        "mac"
    ],

    "icloud": [
        "icloud"
    ],

    "apple_id": [
        "apple id",
        "appleid",
        "password",
        "sign in",
        "login",
        "locked account"
    ],

    "app_store": [
        "app store",
        "download app",
        "purchase app"
    ],

    "itunes_music": [
        "itunes",
        "apple music",
        "music",
        "song",
        "album"
    ],

    "payment_billing": [
        "payment",
        "charged",
        "charge",
        "billing",
        "refund",
        "subscription"
    ],

    "wifi_connectivity": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "internet",
        "connection",
        "connect"
    ],

    "battery_charging": [
        "battery",
        "charging",
        "charger",
        "power"
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
        "failed"
    ],

    "general_support": [
        "help",
        "support",
        "contact",
        "customer service",
        "call",
        "dm"
    ]
}


def classify_intent(text):

    text = text.lower()

    scores = {}

    for intent, keywords in intent_keywords.items():

        score = 0

        for keyword in keywords:

            if keyword in text:
                score += 1

        scores[intent] = score

    best_intent = max(
        scores,
        key=scores.get
    )

    # No keyword matched
    if scores[best_intent] == 0:
        return "other"

    return best_intent


df["intent"] = df["customer_text"].apply(
    classify_intent
)

# ---------------------------------------------------------
# Show distribution
# ---------------------------------------------------------

print("\nIntent distribution:")

print(
    df["intent"]
    .value_counts()
    .to_string()
)

# ---------------------------------------------------------
# Select final columns
# ---------------------------------------------------------

final_df = df[
    [
        "customer_text",
        "brand_response",
        "intent"
    ]
].copy()

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

final_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("INTENT DATASET PREPARATION COMPLETE")
print("=" * 70)