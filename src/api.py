import os
import sys
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# =========================================================
# IMPORT RETRIEVAL
# =========================================================

try:
    from src.retrieval.search import search_similar
except Exception as error:
    raise RuntimeError(
        f"Failed to import retrieval module: {error}"
    )


# =========================================================
# CONFIGURATION
# =========================================================

CONFIDENCE_THRESHOLD = 0.70
SIMILARITY_THRESHOLD = 0.50

TOP_K = 10

RETRIEVAL_WEIGHT = 0.60
INTENT_WEIGHT = 0.40

# IMPORTANT:
# Use 8001 consistently because 8000 was already occupied.
API_HOST = "127.0.0.1"
API_PORT = 8001


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "intent_classifier.joblib"
)


# =========================================================
# LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Intent model not found:\n"
        f"{MODEL_PATH}\n\n"
        "Please run:\n"
        "python scripts/train_intent_model.py"
    )


try:
    model = joblib.load(MODEL_PATH)

except Exception as error:
    raise RuntimeError(
        f"Failed to load intent model: {error}"
    )


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Apple Support AI API",
    description=(
        "Apple customer support AI API with "
        "intent classification, semantic retrieval, "
        "response ranking and confidence-based escalation."
    ),
    version="1.0.0"
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",

        "http://localhost:3000",
        "http://127.0.0.1:3000",

        "http://localhost:5173",
        "http://127.0.0.1:5173",

        "http://localhost:5174",
        "http://127.0.0.1:5174",

        # Required when HTML is opened directly
        "null"
    ],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"]
)


# =========================================================
# REQUEST MODEL
# =========================================================

class PredictionRequest(BaseModel):

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Customer support message"
    )


# =========================================================
# ROOT
# =========================================================

@app.get("/", tags=["System"])
def root():

    return {
        "status": "running",
        "service": "Apple Support AI API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "number_of_intents": len(model.classes_),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "top_k": TOP_K,
        "api_url": f"http://{API_HOST}:{API_PORT}"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health", tags=["System"])
def health():

    model_loaded = model is not None

    return {
        "status": (
            "healthy"
            if model_loaded
            else "unhealthy"
        ),

        "model_loaded": model_loaded,

        "number_of_intents": (
            len(model.classes_)
            if model_loaded
            else 0
        ),

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "similarity_threshold":
            SIMILARITY_THRESHOLD,

        "top_k":
            TOP_K,

        "api_url":
            f"http://{API_HOST}:{API_PORT}"
    }


# =========================================================
# INTENT PREDICTION
# =========================================================

def get_intent_prediction(message):

    try:

        prediction = model.predict(
            [message]
        )[0]

        probabilities = model.predict_proba(
            [message]
        )[0]

    except Exception as error:

        raise RuntimeError(
            f"Intent prediction failed: {error}"
        )

    confidence = float(
        probabilities.max()
    )

    # -----------------------------------------------------
    # Rank predictions
    # -----------------------------------------------------

    ranked_predictions = sorted(
        zip(
            model.classes_,
            probabilities
        ),
        key=lambda item: item[1],
        reverse=True
    )

    # -----------------------------------------------------
    # Top 3 predictions
    # -----------------------------------------------------

    top_3 = []

    for intent, probability in ranked_predictions[:3]:

        top_3.append(
            {
                "intent": str(intent),

                "confidence": round(
                    float(probability),
                    4
                )
            }
        )

    return (
        str(prediction),
        confidence,
        top_3
    )


# =========================================================
# RETRIEVE + RANK
# =========================================================

def retrieve_and_rank(
    message,
    predicted_intent,
    intent_confidence
):

    try:

        retrieved = search_similar(
            message,
            top_k=TOP_K
        )

    except Exception as error:

        raise RuntimeError(
            f"Response retrieval failed: {error}"
        )

    if not retrieved:
        return []

    ranked = []

    for item in retrieved:

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

        item_intent = str(
            item.get(
                "intent",
                ""
            )
        )

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

        # Keep similarity between 0 and 1

        similarity = max(
            0.0,
            min(
                similarity,
                1.0
            )
        )

        # -------------------------------------------------
        # Intent match
        # -------------------------------------------------

        intent_match = (
            bool(item_intent)
            and
            item_intent.strip().lower()
            ==
            predicted_intent.strip().lower()
        )

        # -------------------------------------------------
        # Intent score
        # -------------------------------------------------

        if intent_match:
            intent_score = intent_confidence
        else:
            intent_score = 0.0

        # -------------------------------------------------
        # Final ranking score
        # -------------------------------------------------

        ranking_score = (

            similarity
            * RETRIEVAL_WEIGHT

        ) + (

            intent_score
            * INTENT_WEIGHT

        )

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
                    (
                        item_intent
                        if item_intent
                        else predicted_intent
                    ),

                "predicted_intent":
                    predicted_intent,

                "intent_match":
                    intent_match,

                "intent_confidence":
                    round(
                        intent_confidence,
                        4
                    ),

                "score":
                    round(
                        ranking_score,
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

    return ranked


# =========================================================
# PREDICT ENDPOINT
# =========================================================

@app.post(
    "/predict",
    tags=["Prediction"]
)
def predict(request: PredictionRequest):

    # =====================================================
    # 1. VALIDATE MESSAGE
    # =====================================================

    message = request.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # =====================================================
    # 2. INTENT PREDICTION
    # =====================================================

    try:

        (
            predicted_intent,
            confidence,
            top_3
        ) = get_intent_prediction(
            message
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    # =====================================================
    # 3. CONFIDENCE CHECK
    # =====================================================

    if confidence < CONFIDENCE_THRESHOLD:

        return {

            "message":
                message,

            "predicted_intent":
                predicted_intent,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "top_predictions":
                top_3,

            "status":
                "escalation_required",

            "recommended_response":
                None,

            "similarity":
                None,

            "ranking_score":
                None,

            "ranked_results":
                [],

            "escalation":
                {
                    "required":
                        True,

                    "reason":
                        (
                            "Intent confidence is below "
                            "the required threshold."
                        ),

                    "confidence_threshold":
                        CONFIDENCE_THRESHOLD,

                    "similarity_threshold":
                        SIMILARITY_THRESHOLD
                }
        }

    # =====================================================
    # 4. RETRIEVE + RANK
    # =====================================================

    try:

        ranked_results = retrieve_and_rank(
            message,
            predicted_intent,
            confidence
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    # =====================================================
    # 5. NO RETRIEVAL RESULTS
    # =====================================================

    if not ranked_results:

        return {

            "message":
                message,

            "predicted_intent":
                predicted_intent,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "top_predictions":
                top_3,

            "status":
                "escalation_required",

            "recommended_response":
                None,

            "similarity":
                None,

            "ranking_score":
                None,

            "ranked_results":
                [],

            "escalation":
                {
                    "required":
                        True,

                    "reason":
                        (
                            "No similar support "
                            "conversation was found."
                        ),

                    "confidence_threshold":
                        CONFIDENCE_THRESHOLD,

                    "similarity_threshold":
                        SIMILARITY_THRESHOLD
                }
        }

    # =====================================================
    # 6. BEST RESULT
    # =====================================================

    best_response = ranked_results[0]

    similarity = float(
        best_response.get(
            "similarity",
            0.0
        )
    )

    ranking_score = float(
        best_response.get(
            "score",
            0.0
        )
    )

    recommended_response = (
        best_response.get(
            "brand_response",
            ""
        )
    )

    # =====================================================
    # 7. SIMILARITY CHECK
    # =====================================================

    if similarity < SIMILARITY_THRESHOLD:

        return {

            "message":
                message,

            "predicted_intent":
                predicted_intent,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "top_predictions":
                top_3,

            "status":
                "escalation_required",

            "recommended_response":
                None,

            "similarity":
                round(
                    similarity,
                    4
                ),

            "ranking_score":
                round(
                    ranking_score,
                    4
                ),

            "ranked_results":
                ranked_results,

            "escalation":
                {
                    "required":
                        True,

                    "reason":
                        (
                            "The best retrieved support "
                            "example does not meet the "
                            "minimum similarity threshold."
                        ),

                    "confidence_threshold":
                        CONFIDENCE_THRESHOLD,

                    "similarity_threshold":
                        SIMILARITY_THRESHOLD
                }
        }

    # =====================================================
    # 8. SUCCESS
    # =====================================================

    return {

        "message":
            message,

        "predicted_intent":
            predicted_intent,

        "confidence":
            round(
                confidence,
                4
            ),

        "top_predictions":
            top_3,

        "status":
            "response_generated",

        "recommended_response":
            recommended_response,

        "similarity":
            round(
                similarity,
                4
            ),

        "ranking_score":
            round(
                ranking_score,
                4
            ),

        "ranked_results":
            ranked_results,

        "escalation":
            {
                "required":
                    False,

                "confidence_threshold":
                    CONFIDENCE_THRESHOLD,

                "similarity_threshold":
                    SIMILARITY_THRESHOLD
            }
    }


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False
    )