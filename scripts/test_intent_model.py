import os
import joblib


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "models/intent_classifier.joblib"


print("=" * 70)
print("APPLE SUPPORT - INTENT MODEL TEST")
print("=" * 70)


# =========================================================
# LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("\nModel loaded successfully.")


# =========================================================
# TEST EXAMPLES
# =========================================================

test_messages = [
    "My iPhone is not connecting to WiFi",
    "My battery is draining very quickly",
    "I was charged twice for my purchase",
    "I forgot my Apple ID password",
    "I cannot access my iCloud account",
    "I cannot download an app from the App Store",
    "My internet connection keeps disconnecting",
    "My iPhone won't charge",
    "I need help with an iOS update",
    "The App Store is not working"
]


# =========================================================
# PREDICTION
# =========================================================

print("\n" + "=" * 70)
print("PREDICTIONS")
print("=" * 70)

predictions = model.predict(test_messages)

for message, prediction in zip(
    test_messages,
    predictions
):
    print("\nCustomer:")
    print(message)

    print("Predicted intent:")
    print(prediction)


# =========================================================
# CONFIDENCE SCORES
# =========================================================

print("\n" + "=" * 70)
print("PREDICTIONS WITH CONFIDENCE")
print("=" * 70)


if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(
        test_messages
    )

    classes = model.classes_

    for message, probs in zip(
        test_messages,
        probabilities
    ):

        best_index = probs.argmax()

        intent = classes[best_index]
        confidence = probs[best_index]

        print("\nCustomer:")
        print(message)

        print(
            f"Intent: {intent}"
        )

        print(
            f"Confidence: {confidence:.2%}"
        )


print("\n" + "=" * 70)
print("MODEL TEST COMPLETE")
print("=" * 70)