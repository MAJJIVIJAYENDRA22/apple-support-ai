import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/processed/intent_dataset.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "intent_classifier.joblib"
)


print("=" * 70)
print("APPLE SUPPORT - INTENT CLASSIFIER TRAINING")
print("=" * 70)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print(f"\nDataset rows: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")


# =========================================================
# VALIDATE DATA
# =========================================================

required_columns = [
    "customer_text",
    "intent"
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
# REMOVE MISSING VALUES
# =========================================================

df = df.dropna(
    subset=["customer_text", "intent"]
).copy()


# =========================================================
# CONVERT TO STRING
# =========================================================

df["customer_text"] = df["customer_text"].astype(str)
df["intent"] = df["intent"].astype(str)


# =========================================================
# REMOVE EMPTY CUSTOMER MESSAGES
# =========================================================

df = df[
    df["customer_text"].str.strip().str.len() > 0
]


print(f"\nUsable rows: {len(df):,}")
print(
    f"Number of intents: "
    f"{df['intent'].nunique()}"
)


# =========================================================
# INTENT DISTRIBUTION
# =========================================================

print("\nIntent distribution:\n")

print(
    df["intent"]
    .value_counts()
    .to_string()
)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X = df["customer_text"]
y = df["intent"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# BUILD MODEL
# =========================================================

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]
)


# =========================================================
# TRAIN MODEL
# =========================================================

print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)

model.fit(
    X_train,
    y_train
)

print("\nTraining complete.")


# =========================================================
# EVALUATE MODEL
# =========================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)


predictions = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    f"\nAccuracy: {accuracy:.4f}"
)


print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    model,
    MODEL_PATH
)


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    f"\nSaved to:\n{MODEL_PATH}"
)


print("\n" + "=" * 70)
print("INTENT MODEL TRAINING COMPLETE")
print("=" * 70)