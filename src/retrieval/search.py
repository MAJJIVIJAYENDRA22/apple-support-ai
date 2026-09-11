import os
import re
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# =========================================================
# MODEL PATHS
# =========================================================

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR,
    "intent_classifier.joblib"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "retrieval_vectorizer.joblib"
)

MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "retrieval_matrix.joblib"
)

DATA_PATH = os.path.join(
    MODEL_DIR,
    "retrieval_data.joblib"
)


# =========================================================
# CHECK REQUIRED FILES
# =========================================================

required_files = {
    "Intent classifier": CLASSIFIER_PATH,
    "Retrieval vectorizer": VECTORIZER_PATH,
    "Retrieval matrix": MATRIX_PATH,
    "Retrieval data": DATA_PATH,
}

for name, path in required_files.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


# =========================================================
# LOAD MODELS
# =========================================================

# IMPORTANT:
# The intent classifier is already a complete Pipeline.
#
# Pipeline:
#     TF-IDF Vectorizer
#          ↓
#     Logistic Regression
#
# Therefore we pass raw text directly to:
#
#     intent_classifier.predict([query])
#
# We DO NOT manually transform the query first.

intent_classifier = joblib.load(
    CLASSIFIER_PATH
)


# Retrieval uses its own vectorizer and matrix.

retrieval_vectorizer = joblib.load(
    VECTORIZER_PATH
)

retrieval_matrix = joblib.load(
    MATRIX_PATH
)

retrieval_data = joblib.load(
    DATA_PATH
)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_query(text):

    text = str(text)

    # Remove Twitter usernames
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# INTENT CLASSIFICATION
# =========================================================

def classify_intent(query):

    query = clean_query(query)

    if not query:
        return None

    # IMPORTANT:
    # intent_classifier is a complete sklearn Pipeline.
    # Pass raw text directly.

    prediction = intent_classifier.predict(
        [query]
    )[0]

    return str(prediction)


# =========================================================
# RETRIEVAL SEARCH
# =========================================================

def search_similar(
    query,
    top_k=5,
    min_similarity=0.10
):

    query = clean_query(query)

    if not query:
        return []


    # -----------------------------------------------------
    # CLASSIFY INTENT
    # -----------------------------------------------------

    intent = classify_intent(
        query
    )


    # -----------------------------------------------------
    # CONVERT QUERY TO RETRIEVAL TF-IDF
    # -----------------------------------------------------

    query_vector = retrieval_vectorizer.transform(
        [query]
    )


    # -----------------------------------------------------
    # CALCULATE COSINE SIMILARITY
    # -----------------------------------------------------

    scores = cosine_similarity(
        query_vector,
        retrieval_matrix
    )[0]


    # -----------------------------------------------------
    # RANK RESULTS
    # -----------------------------------------------------

    ranked_indices = scores.argsort()[::-1]


    results = []


    for index in ranked_indices:

        similarity = float(
            scores[index]
        )

        if similarity < min_similarity:
            break


        row = retrieval_data.iloc[index]


        results.append(
            {
                "customer_text": str(
                    row["customer_text"]
                ),

                "brand_response": str(
                    row["brand_response"]
                ),

                "similarity": round(
                    similarity,
                    4
                ),

                "intent": intent,

                "intent_source": "intent_classifier"
            }
        )


        if len(results) >= top_k:
            break


    return results


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "APPLE SUPPORT - RETRIEVAL SEARCH TEST"
    )

    print("=" * 70)


    test_queries = [

        "My iPhone cannot connect to WiFi",

        "I was charged twice",

        "My iCloud account is not working",

        "The App Store is not working",

        "I need help with an iOS update"

    ]


    for query in test_queries:

        print(
            "\n" + "-" * 70
        )

        print(
            f"Query:\n{query}"
        )


        # -------------------------------------------------
        # INTENT
        # -------------------------------------------------

        intent = classify_intent(
            query
        )

        print(
            f"\nPredicted Intent: {intent}"
        )


        # -------------------------------------------------
        # RETRIEVAL
        # -------------------------------------------------

        results = search_similar(
            query,
            top_k=3
        )


        if not results:

            print(
                "\nNo sufficiently similar results found."
            )

            continue


        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRank {rank}"
            )

            print(
                f"Similarity: "
                f"{result['similarity']:.2%}"
            )

            print(
                f"Intent: "
                f"{result['intent']}"
            )

            print(
                f"Customer:\n"
                f"{result['customer_text']}"
            )

            print(
                f"\nResponse:\n"
                f"{result['brand_response']}"
            )


    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVAL SEARCH TEST COMPLETE"
    )

    print(
        "=" * 70
    )