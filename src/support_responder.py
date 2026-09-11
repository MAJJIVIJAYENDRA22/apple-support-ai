import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "models/intent_classifier.joblib"

CONVERSATION_PATH = (
    "data/processed/applesupport_conversations.csv"
)


# =========================================================
# LOAD INTENT MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Intent model not found: {MODEL_PATH}"
    )

intent_model = joblib.load(MODEL_PATH)


# =========================================================
# LOAD CONVERSATIONS
# =========================================================

if not os.path.exists(CONVERSATION_PATH):
    raise FileNotFoundError(
        f"Conversation dataset not found: "
        f"{CONVERSATION_PATH}"
    )

df = pd.read_csv(CONVERSATION_PATH)


# =========================================================
# VALIDATE COLUMNS
# =========================================================

required_columns = [
    "customer_text",
    "brand_response"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna(
    subset=[
        "customer_text",
        "brand_response"
    ]
).copy()


df["customer_text"] = (
    df["customer_text"]
    .astype(str)
    .str.strip()
)

df["brand_response"] = (
    df["brand_response"]
    .astype(str)
    .str.strip()
)


# Remove empty records

df = df[
    (df["customer_text"] != "") &
    (df["brand_response"] != "")
].reset_index(drop=True)


print(
    f"Loaded {len(df):,} Apple Support conversations."
)


# =========================================================
# BUILD SEARCH INDEX
# =========================================================

print("Building response search index...")


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)


conversation_matrix = vectorizer.fit_transform(
    df["customer_text"]
)


print("Response search index ready.")


# =========================================================
# RESPONSE FUNCTION
# =========================================================

def generate_response(message, top_k=3):

    message = str(message).strip()

    if not message:
        raise ValueError(
            "Customer message cannot be empty."
        )


    # -----------------------------------------------------
    # STEP 1: Predict intent
    # -----------------------------------------------------

    predicted_intent = intent_model.predict(
        [message]
    )[0]


    probabilities = intent_model.predict_proba(
        [message]
    )[0]


    confidence = float(
        probabilities.max()
    )


    # -----------------------------------------------------
    # STEP 2: Convert message to TF-IDF
    # -----------------------------------------------------

    message_vector = vectorizer.transform(
        [message]
    )


    # -----------------------------------------------------
    # STEP 3: Calculate similarity
    # -----------------------------------------------------

    similarities = cosine_similarity(
        message_vector,
        conversation_matrix
    ).flatten()


    # -----------------------------------------------------
    # STEP 4: Get best matches
    # -----------------------------------------------------

    top_indices = similarities.argsort()[
        -top_k:
    ][::-1]


    matches = []


    for index in top_indices:

        score = float(
            similarities[index]
        )

        matches.append(
            {
                "customer_text": df.iloc[index][
                    "customer_text"
                ],

                "brand_response": df.iloc[index][
                    "brand_response"
                ],

                "similarity": round(
                    score,
                    4
                )
            }
        )


    # -----------------------------------------------------
    # STEP 5: Select best response
    # -----------------------------------------------------

    best_match = matches[0]


    return {
        "message": message,

        "predicted_intent": predicted_intent,

        "intent_confidence": round(
            confidence,
            4
        ),

        "response": best_match[
            "brand_response"
        ],

        "similarity": best_match[
            "similarity"
        ],

        "matches": matches
    }


# =========================================================
# COMMAND LINE TEST
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("APPLE SUPPORT - RESPONSE ENGINE TEST")
    print("=" * 70)


    test_messages = [

        "My iPhone battery is draining very quickly",

        "I cannot connect my iPhone to WiFi",

        "I forgot my Apple ID password",

        "I was charged twice for the same purchase",

        "My iCloud account is not working"

    ]


    for message in test_messages:

        print("\n" + "-" * 70)

        print(
            f"Customer: {message}"
        )


        result = generate_response(
            message
        )


        print(
            f"\nIntent: "
            f"{result['predicted_intent']}"
        )


        print(
            f"Intent confidence: "
            f"{result['intent_confidence']:.2%}"
        )


        print(
            f"Similarity: "
            f"{result['similarity']:.2%}"
        )


        print("\nRecommended response:")

        print(
            result["response"]
        )


    print("\n" + "=" * 70)
    print("RESPONSE ENGINE TEST COMPLETE")
    print("=" * 70)