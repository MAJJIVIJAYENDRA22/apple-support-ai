import os
import sys
import joblib


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# Make sure project root is available for imports
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# Import after fixing Python path
from src.retrieval.search import search_similar


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "intent_classifier.joblib"
)

DEFAULT_TOP_K = 5

RETRIEVAL_WEIGHT = 0.70
INTENT_WEIGHT = 0.30


# =========================================================
# LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Intent model not found:\n{MODEL_PATH}\n\n"
        "Please train the intent model first."
    )


try:

    model = joblib.load(
        MODEL_PATH
    )

except Exception as error:

    raise RuntimeError(
        f"Failed to load intent model: {error}"
    )


# =========================================================
# INTENT PREDICTION
# =========================================================

def predict_intent(message):

    """
    Predict the customer support intent.

    Returns:
        tuple:
            predicted_intent
            confidence
    """

    if not isinstance(
        message,
        str
    ):

        raise TypeError(
            "Message must be a string."
        )


    message = message.strip()


    if not message:

        raise ValueError(
            "Message cannot be empty."
        )


    try:

        prediction = model.predict(
            [message]
        )[0]

        probabilities = model.predict_proba(
            [message]
        )[0]

        confidence = float(
            probabilities.max()
        )

    except Exception as error:

        raise RuntimeError(
            f"Intent prediction failed: {error}"
        )


    return (
        str(prediction),
        confidence
    )


# =========================================================
# RESPONSE RANKING
# =========================================================

def rank_responses(
    message,
    top_k=DEFAULT_TOP_K
):

    """
    Predict intent, retrieve similar conversations,
    and rank the retrieved responses.

    Ranking formula:

        Final Score =
            70% Retrieval Similarity
            +
            30% Intent Confidence
    """


    # -----------------------------------------------------
    # Validate message
    # -----------------------------------------------------

    if not isinstance(
        message,
        str
    ):

        raise TypeError(
            "Message must be a string."
        )


    message = message.strip()


    if not message:

        raise ValueError(
            "Message cannot be empty."
        )


    # -----------------------------------------------------
    # Validate top_k
    # -----------------------------------------------------

    try:

        top_k = int(
            top_k
        )

    except (TypeError, ValueError):

        top_k = DEFAULT_TOP_K


    top_k = max(
        1,
        min(
            top_k,
            20
        )
    )


    # -----------------------------------------------------
    # Predict intent
    # -----------------------------------------------------

    predicted_intent, intent_confidence = (
        predict_intent(
            message
        )
    )


    # -----------------------------------------------------
    # Retrieve similar conversations
    # -----------------------------------------------------

    try:

        retrieved = search_similar(
            message,
            top_k=top_k
        )

    except Exception as error:

        raise RuntimeError(
            f"Response retrieval failed: {error}"
        )


    # -----------------------------------------------------
    # Handle no results
    # -----------------------------------------------------

    if not retrieved:

        return {

            "message": message,

            "intent": predicted_intent,

            "intent_confidence":
                round(
                    intent_confidence,
                    4
                ),

            "results": []
        }


    # -----------------------------------------------------
    # Score retrieved responses
    # -----------------------------------------------------

    ranked = []


    for item in retrieved:

        # Safely read values
        customer_text = str(
            item.get(
                "customer_text",
                ""
            )
        )

        brand_response = str(
            item.get(
                "brand_response",
                ""
            )
        )


        # -----------------------------------------------
        # Retrieval similarity
        # -----------------------------------------------

        try:

            similarity = float(
                item.get(
                    "similarity",
                    0.0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            similarity = 0.0


        # Keep similarity inside valid range
        similarity = max(
            0.0,
            min(
                similarity,
                1.0
            )
        )


        # -----------------------------------------------
        # Combined ranking score
        # -----------------------------------------------

        combined_score = (
            similarity
            * RETRIEVAL_WEIGHT
        ) + (
            intent_confidence
            * INTENT_WEIGHT
        )


        # -----------------------------------------------
        # Create ranked result
        # -----------------------------------------------

        ranked.append(
            {
                "customer_text":
                    customer_text,

                "brand_response":
                    brand_response,

                "similarity":
                    round(
                        similarity,
                        4
                    ),

                "intent":
                    predicted_intent,

                "intent_confidence":
                    round(
                        intent_confidence,
                        4
                    ),

                "score":
                    round(
                        combined_score,
                        4
                    )
            }
        )


    # -----------------------------------------------------
    # Sort highest score first
    # -----------------------------------------------------

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    # -----------------------------------------------------
    # Return final result
    # -----------------------------------------------------

    return {

        "message":
            message,

        "intent":
            predicted_intent,

        "intent_confidence":
            round(
                intent_confidence,
                4
            ),

        "results":
            ranked
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("APPLE SUPPORT - RESPONSE RANKING TEST")
    print("=" * 70)


    test_queries = [

        "My iPhone cannot connect to WiFi",

        "I was charged twice",

        "My iCloud account is not working",

        "The App Store is not working",

        "I need help with an iOS update"

    ]


    for message in test_queries:

        print("\n" + "=" * 70)

        print(
            f"Customer:\n{message}"
        )


        try:

            result = rank_responses(
                message,
                top_k=3
            )


            print(
                f"\nPredicted intent: "
                f"{result['intent']}"
            )


            print(
                f"Intent confidence: "
                f"{result['intent_confidence']:.2%}"
            )


            print(
                "\nRanked responses:"
            )


            if not result["results"]:

                print(
                    "\nNo similar responses found."
                )

                continue


            for rank, item in enumerate(
                result["results"],
                start=1
            ):

                print(
                    "\n" + "-" * 60
                )


                print(
                    f"Rank: {rank}"
                )


                print(
                    f"Similarity: "
                    f"{item['similarity']:.2%}"
                )


                print(
                    f"Intent confidence: "
                    f"{item['intent_confidence']:.2%}"
                )


                print(
                    f"Combined score: "
                    f"{item['score']:.2%}"
                )


                print(
                    f"\nCustomer example:\n"
                    f"{item['customer_text']}"
                )


                print(
                    f"\nRecommended response:\n"
                    f"{item['brand_response']}"
                )


        except Exception as error:

            print(
                f"\nERROR: {error}"
            )


    print(
        "\n" + "=" * 70
    )

    print(
        "RESPONSE RANKING TEST COMPLETE"
    )

    print(
        "=" * 70
    )