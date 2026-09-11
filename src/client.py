import requests


API_URL = "http://127.0.0.1:8000/predict"


def ask_support(message):

    if not isinstance(message, str):
        raise TypeError("Message must be a string.")

    message = message.strip()

    if not message:
        raise ValueError("Message cannot be empty.")

    try:

        response = requests.post(
            API_URL,
            json={
                "message": message
            },
            timeout=30
        )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Could not connect to the Apple Support AI API. "
            "Make sure FastAPI is running."
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "API request timed out."
        )

    if response.status_code != 200:

        try:
            detail = response.json().get(
                "detail",
                response.text
            )
        except Exception:
            detail = response.text

        raise RuntimeError(
            f"API error ({response.status_code}): {detail}"
        )

    return response.json()


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("APPLE SUPPORT AI - API CLIENT TEST")
    print("=" * 70)

    test_messages = [

        "My iPhone cannot connect to WiFi",

        "I was charged twice",

        "My iCloud account is not working",

        "The App Store is not working",

        "I need help with an iOS update",

        "I have a problem with something"
    ]

    for message in test_messages:

        print("\n" + "=" * 70)
        print(f"Customer: {message}")

        try:

            result = ask_support(message)

            print(
                f"\nIntent: "
                f"{result.get('predicted_intent')}"
            )

            print(
                f"Confidence: "
                f"{result.get('confidence', 0):.2%}"
            )

            print(
                f"Status: "
                f"{result.get('status')}"
            )

            print(
                f"Similarity: "
                f"{result.get('similarity')}"
            )

            print(
                "\nRecommended Response:"
            )

            print(
                result.get(
                    "recommended_response"
                )
            )

            print(
                "\nEscalation:"
            )

            print(
                result.get(
                    "escalation"
                )
            )

        except Exception as error:

            print(
                f"\nERROR: {error}"
            )

    print("\n" + "=" * 70)
    print("API CLIENT TEST COMPLETE")
    print("=" * 70)