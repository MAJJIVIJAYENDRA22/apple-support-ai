import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "applesupport_conversations.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "retrieval_vectorizer.joblib"
)

MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "retrieval_matrix.joblib"
)

DATA_INDEX_PATH = os.path.join(
    MODEL_DIR,
    "retrieval_data.joblib"
)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 70)
print("APPLE SUPPORT - RETRIEVAL INDEX")
print("=" * 70)

print("\nLoading conversation dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(
    f"Loaded {len(df):,} conversation records."
)


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

print("\nCleaning data...")

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

df = df[
    (df["customer_text"] != "") &
    (df["brand_response"] != "")
].reset_index(drop=True)


print(
    f"Usable records: {len(df):,}"
)


# =========================================================
# BUILD TF-IDF VECTORISER
# =========================================================

print("\nBuilding TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


# =========================================================
# CREATE SEARCH MATRIX
# =========================================================

print("Creating retrieval matrix...")

matrix = vectorizer.fit_transform(
    df["customer_text"]
)

print(
    f"Matrix shape: {matrix.shape}"
)


# =========================================================
# CREATE MODEL DIRECTORY
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# SAVE VECTORIZER
# =========================================================

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

print(
    f"\nVectorizer saved:\n{VECTORIZER_PATH}"
)


# =========================================================
# SAVE MATRIX
# =========================================================

joblib.dump(
    matrix,
    MATRIX_PATH
)

print(
    f"Matrix saved:\n{MATRIX_PATH}"
)


# =========================================================
# SAVE RETRIEVAL DATA
# =========================================================

joblib.dump(
    df,
    DATA_INDEX_PATH
)

print(
    f"Retrieval data saved:\n{DATA_INDEX_PATH}"
)


# =========================================================
# TEST RETRIEVAL
# =========================================================

print("\n" + "=" * 70)
print("TESTING RETRIEVAL")
print("=" * 70)

test_query = "My iPhone is not connecting to WiFi"

query_vector = vectorizer.transform(
    [test_query]
)

scores = cosine_similarity(
    query_vector,
    matrix
)[0]

top_indices = scores.argsort()[-3:][::-1]


print(
    f"\nTest query:\n{test_query}"
)

print("\nTop 3 matches:\n")

for rank, index in enumerate(
    top_indices,
    start=1
):

    print("-" * 70)

    print(
        f"Rank: {rank}"
    )

    print(
        f"Similarity: {scores[index]:.4f}"
    )

    print(
        f"Customer:\n{df.iloc[index]['customer_text']}"
    )

    print(
        f"\nResponse:\n{df.iloc[index]['brand_response']}"
    )


print("\n" + "=" * 70)
print("RETRIEVAL INDEX BUILD COMPLETE")
print("=" * 70)