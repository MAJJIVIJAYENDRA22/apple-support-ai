import os
import sys
import joblib


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "models/intent_classifier.joblib"


# =========================================================
# LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# =========================================================
# GET CUSTOMER MESSAGE
# =========================================================

if len(sys.argv) < 2:
    print("Usage:")
    print(
        'python scripts/predict_intent.py "your customer message"'
    )
    sys.exit(1)


message = " ".join(sys.argv[1:]).strip()


if not message:
    print("ERROR: Customer message is empty.")
    sys.exit(1)


# =========================================================
# PREDICT
# =========================================================

prediction = model.predict(
    [message]
)[0]


# =========================================================
# CONFIDENCE
# =========================================================

probabilities = model.predict_proba(
    [message]
)[0]

classes = model.classes_


# Sort probabilities from highest to lowest
ranked = sorted(
    zip(classes, probabilities),
    key=lambda x: x[1],
    reverse=True
)


# =========================================================
# OUTPUT
# =========================================================

print("=" * 70)
print("APPLE SUPPORT - INTENT PREDICTION")
print("=" * 70)

print("\nCustomer message:")
print(message)

print("\nPredicted intent:")
print(prediction)

print(
    f"\nConfidence: {probabilities.max():.2%}"
)


print("\nTop 3 predictions:")

for rank, (intent, probability) in enumerate(
    ranked[:3],
    start=1
):

    print(
        f"{rank}. "
        f"{intent:<25} "
        f"{probability:.2%}"
    )


print("\n" + "=" * 70)