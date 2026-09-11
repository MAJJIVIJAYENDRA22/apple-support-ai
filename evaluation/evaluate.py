import os
import sys
import json
import warnings

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "golden",
    "golden_set.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "intent_classifier.joblib"
)

RESULT_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "evaluation_results.json"
)


# =========================================================
# CONFIGURATION
# =========================================================

CONFIDENCE_THRESHOLD = 0.70
SIMILARITY_THRESHOLD = 0.60

RANDOM_SEED = 42

TEST_SIZE = 0.25


# =========================================================
# IMPORT RETRIEVAL
# =========================================================

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:

    from src.retrieval.search import search_similar

except Exception as error:

    search_similar = None

    print(
        f"[WARNING] Retrieval module unavailable: {error}"
    )


# =========================================================
# LOAD GOLDEN SET
# =========================================================

def load_golden_set():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Golden set not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\nGolden Set Columns:")
    print(df.columns.tolist())

    # -----------------------------------------------------
    # Detect customer message column
    # -----------------------------------------------------

    message_candidates = [
        "customer_message",
        "customer_text",
        "message",
        "text",
        "query",
        "customer_query"
    ]

    message_column = next(
        (
            col
            for col in message_candidates
            if col in df.columns
        ),
        None
    )

    if message_column is None:
        raise ValueError(
            "Could not find customer message column.\n"
            f"Expected one of: {message_candidates}\n"
            f"Found: {df.columns.tolist()}"
        )

    # -----------------------------------------------------
    # Detect human-labelled intent column
    # -----------------------------------------------------

    intent_candidates = [
        "true_intent",
        "golden_intent",
        "intent",
        "label",
        "category",
        "true_label"
    ]

    intent_column = next(
        (
            col
            for col in intent_candidates
            if col in df.columns
        ),
        None
    )

    if intent_column is None:
        raise ValueError(
            "Could not find intent label column.\n"
            f"Expected one of: {intent_candidates}\n"
            f"Found: {df.columns.tolist()}\n\n"
            "Your golden_set.csv must contain the "
            "human-labelled intent for every example."
        )

    # -----------------------------------------------------
    # Standardize column names
    # -----------------------------------------------------

    df = df.rename(
        columns={
            message_column: "customer_message",
            intent_column: "true_intent"
        }
    )

    # -----------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------

    required_columns = [
        "customer_message",
        "true_intent"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # -----------------------------------------------------
    # Clean data
    # -----------------------------------------------------

    df = df.dropna(
        subset=[
            "customer_message",
            "true_intent"
        ]
    ).copy()

    df["customer_message"] = (
        df["customer_message"]
        .astype(str)
        .str.strip()
    )

    df["true_intent"] = (
        df["true_intent"]
        .astype(str)
        .str.strip()
    )

    # Remove empty examples
    df = df[
        (df["customer_message"] != "") &
        (df["true_intent"] != "")
    ].copy()

    # -----------------------------------------------------
    # Normalize escalation column if available
    # -----------------------------------------------------

    if "should_escalate" in df.columns:

        df["should_escalate"] = (
            df["should_escalate"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin([
                "true",
                "1",
                "yes",
                "y"
            ])
        )

    print(
        f"\nLoaded {len(df)} golden evaluation examples."
    )

    print(
        f"Intent column: {intent_column}"
    )

    print(
        f"Message column: {message_column}"
    )

    print(
        f"Number of intents: "
        f"{df['true_intent'].nunique()}"
    )

    return df
    # -----------------------------------------------------
    # Find intent column
    # -----------------------------------------------------

    intent_candidates = [
        "true_intent",
        "intent",
        "label",
        "category",
        "true_label"
    ]

    intent_column = None

    for column in intent_candidates:

        if column in df.columns:

            intent_column = column
            break

    if intent_column is None:

        raise ValueError(
            "\nCould not find intent label column.\n"
            "Expected one of:\n"
            f"{intent_candidates}\n\n"
            f"Found:\n{list(df.columns)}\n\n"
            "Your golden_set.csv must contain the "
            "human-labelled intent for every example."
        )

    # -----------------------------------------------------
    # Rename to standard names
    # -----------------------------------------------------

    df = df.rename(
        columns={
            message_column: "customer_message",
            intent_column: "true_intent"
        }
    )

    # -----------------------------------------------------
    # Clean data
    # -----------------------------------------------------

    df = df.dropna(
        subset=[
            "customer_message",
            "true_intent"
        ]
    )

    df["customer_message"] = (
        df["customer_message"]
        .astype(str)
        .str.strip()
    )

    df["true_intent"] = (
        df["true_intent"]
        .astype(str)
        .str.strip()
    )

    # Remove empty messages
    df = df[
        df["customer_message"] != ""
    ]

    # Remove empty intents
    df = df[
        df["true_intent"] != ""
    ]

    df = df.reset_index(
        drop=True
    )

    print(
        f"\nLoaded {len(df)} valid golden examples."
    )

    print(
        f"Intent classes: {df['true_intent'].nunique()}"
    )

    return df


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    y_true,
    y_pred
):

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {

        "accuracy": round(
            accuracy_score(
                y_true,
                y_pred
            ),
            4
        ),

        "macro_precision": round(
            precision,
            4
        ),

        "macro_recall": round(
            recall,
            4
        ),

        "macro_f1": round(
            f1,
            4
        ),

        "weighted_f1": round(
            weighted_f1,
            4
        )
    }


# =========================================================
# PRIMARY MODEL
# =========================================================

def evaluate_primary_model(
    df
):

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nIntent model not found:\n"
            f"{MODEL_PATH}\n\n"
            f"Train the model first."
        )

    model = joblib.load(
        MODEL_PATH
    )

    X = df[
        "customer_message"
    ]

    y_true = df[
        "true_intent"
    ]

    y_pred = model.predict(
        X
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    print("\n")
    print("=" * 60)
    print("PRIMARY MODEL")
    print("=" * 60)

    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )

    return model, metrics


# =========================================================
# MAJORITY BASELINE
# =========================================================

def evaluate_majority_baseline(
    train_df,
    test_df
):

    y_train = train_df[
        "true_intent"
    ]

    y_test = test_df[
        "true_intent"
    ]

    majority_class = (
        y_train
        .value_counts()
        .idxmax()
    )

    y_pred = [
        majority_class
        for _ in y_test
    ]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    result = {

        "accuracy": round(
            accuracy,
            4
        ),

        "macro_f1": round(
            macro_f1,
            4
        ),

        "majority_class": str(
            majority_class
        )
    }

    print("\n")
    print("=" * 60)
    print("MAJORITY CLASS BASELINE")
    print("=" * 60)

    print(
        f"Majority class: {majority_class}"
    )

    print(
        f"Accuracy: {result['accuracy']}"
    )

    print(
        f"Macro F1: {result['macro_f1']}"
    )

    return result


# =========================================================
# TF-IDF BASELINE
# =========================================================

def evaluate_tfidf_baseline(
    train_df,
    test_df
):

    X_train = train_df[
        "customer_message"
    ]

    y_train = train_df[
        "true_intent"
    ]

    X_test = test_df[
        "customer_message"
    ]

    y_test = test_df[
        "true_intent"
    ]

    model = Pipeline(
        [

            (
                "tfidf",

                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=10000
                )
            ),

            (
                "classifier",

                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        y_pred
    )

    print("\n")
    print("=" * 60)
    print("TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 60)

    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )

    return model, metrics


# =========================================================
# RETRIEVAL EVALUATION
# =========================================================

def evaluate_retrieval(
    df
):

    if search_similar is None:

        print(
            "\n[WARNING] Retrieval evaluation skipped."
        )

        return {

            "recall_at_1": None,
            "recall_at_3": None,
            "recall_at_5": None,
            "recall_at_10": None,
            "mean_reciprocal_rank": None
        }

    hits = {

        1: 0,
        3: 0,
        5: 0,
        10: 0
    }

    reciprocal_ranks = []

    total = len(df)

    print("\n")
    print("=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    for index, row in df.iterrows():

        message = row[
            "customer_message"
        ]

        true_intent = str(
            row["true_intent"]
        ).strip().lower()

        try:

            results = search_similar(
                message,
                top_k=10
            )

        except Exception:

            results = []

        found_rank = None

        for rank, item in enumerate(
            results,
            start=1
        ):

            item_intent = str(
                item.get(
                    "intent",
                    ""
                )
            ).strip().lower()

            if item_intent == true_intent:

                found_rank = rank
                break

        if found_rank is not None:

            reciprocal_ranks.append(
                1 / found_rank
            )

            for k in hits:

                if found_rank <= k:

                    hits[k] += 1

        else:

            reciprocal_ranks.append(
                0
            )

    if total == 0:

        return {
            "recall_at_1": 0,
            "recall_at_3": 0,
            "recall_at_5": 0,
            "recall_at_10": 0,
            "mean_reciprocal_rank": 0
        }

    result = {

        "recall_at_1": round(
            hits[1] / total,
            4
        ),

        "recall_at_3": round(
            hits[3] / total,
            4
        ),

        "recall_at_5": round(
            hits[5] / total,
            4
        ),

        "recall_at_10": round(
            hits[10] / total,
            4
        ),

        "mean_reciprocal_rank": round(
            sum(reciprocal_ranks) / total,
            4
        )
    }

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    return result


# =========================================================
# ESCALATION EVALUATION
# =========================================================

def evaluate_escalation(
    df,
    model
):

    auto_handled = 0
    escalated = 0

    escalation_labels = []
    escalation_predictions = []

    has_escalation_labels = (
        "should_escalate"
        in df.columns
    )

    print("\n")
    print("=" * 60)
    print("ESCALATION EVALUATION")
    print("=" * 60)

    for _, row in df.iterrows():

        message = row[
            "customer_message"
        ]

        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        try:

            probabilities = (
                model.predict_proba(
                    [message]
                )[0]
            )

            confidence = float(
                probabilities.max()
            )

        except Exception:

            confidence = 0.0

        # -------------------------------------------------
        # Similarity
        # -------------------------------------------------

        best_similarity = 0.0

        if search_similar is not None:

            try:

                results = search_similar(
                    message,
                    top_k=10
                )

                if results:

                    best_similarity = float(
                        results[0].get(
                            "similarity",
                            0.0
                        )
                    )

                    best_similarity = max(
                        0.0,
                        min(
                            best_similarity,
                            1.0
                        )
                    )

            except Exception:

                best_similarity = 0.0

        # -------------------------------------------------
        # Same decision logic as API
        # -------------------------------------------------

        predicted_escalation = (

            confidence
            < CONFIDENCE_THRESHOLD

            or

            best_similarity
            < SIMILARITY_THRESHOLD
        )

        if predicted_escalation:

            escalated += 1

        else:

            auto_handled += 1

        # -------------------------------------------------
        # Human label
        # -------------------------------------------------

        if has_escalation_labels:

            actual = str(
                row[
                    "should_escalate"
                ]
            ).strip().lower()

            actual = actual in [
                "true",
                "1",
                "yes",
                "y"
            ]

            escalation_labels.append(
                actual
            )

            escalation_predictions.append(
                predicted_escalation
            )

    result = {

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "similarity_threshold":
            SIMILARITY_THRESHOLD,

        "total_examples":
            len(df),

        "auto_handled":
            auto_handled,

        "escalated":
            escalated
    }

    # -----------------------------------------------------
    # Calculate actual escalation metrics
    # -----------------------------------------------------

    if has_escalation_labels:

        precision, recall, _, _ = (
            precision_recall_fscore_support(
                escalation_labels,
                escalation_predictions,
                average="binary",
                zero_division=0
            )
        )

        f1 = f1_score(
            escalation_labels,
            escalation_predictions,
            zero_division=0
        )

        result[
            "precision"
        ] = round(
            precision,
            4
        )

        result[
            "recall"
        ] = round(
            recall,
            4
        )

        result[
            "f1"
        ] = round(
            f1,
            4
        )

    else:

        result[
            "precision"
        ] = None

        result[
            "recall"
        ] = None

        result[
            "f1"
        ] = None

        print(
            "\n[INFO] No should_escalate column found."
        )

        print(
            "Escalation counts calculated, "
            "but precision/recall/F1 cannot "
            "be measured without human labels."
        )

    print(
        f"Auto handled: {auto_handled}"
    )

    print(
        f"Escalated: {escalated}"
    )

    print(
        f"Escalation F1: {result['f1']}"
    )

    return result


# =========================================================
# UPDATE JSON
# =========================================================

def update_results(
    primary_metrics,
    majority_metrics,
    tfidf_metrics,
    retrieval_metrics,
    escalation_metrics,
    test_size
):

    if os.path.exists(
        RESULT_PATH
    ):

        try:

            with open(
                RESULT_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                results = json.load(
                    file
                )

        except json.JSONDecodeError:

            print(
                "[WARNING] Existing evaluation_results.json "
                "is invalid. Creating a new one."
            )

            results = {}

    else:

        results = {}

    # -----------------------------------------------------
    # Project
    # -----------------------------------------------------

    results.setdefault(
        "project",
        {}
    )

    results[
        "project"
    ][
        "golden_set_size"
    ] = test_size

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    results.setdefault(
        "evaluation",
        {}
    )

    results[
        "evaluation"
    ][
        "random_seed"
    ] = RANDOM_SEED

    results[
        "evaluation"
    ][
        "test_set_size"
    ] = test_size

    # -----------------------------------------------------
    # Intent Classification
    # -----------------------------------------------------

    results.setdefault(
        "intent_classification",
        {}
    )

    results[
        "intent_classification"
    ][
        "primary_model"
    ] = primary_metrics

    results[
        "intent_classification"
    ].setdefault(
        "baselines",
        {}
    )

    results[
        "intent_classification"
    ][
        "baselines"
    ][
        "majority_class"
    ] = {

        "accuracy":
            majority_metrics[
                "accuracy"
            ],

        "macro_f1":
            majority_metrics[
                "macro_f1"
            ]
    }

    results[
        "intent_classification"
    ][
        "baselines"
    ][
        "tfidf_logistic_regression"
    ] = tfidf_metrics

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    results[
        "retrieval"
    ] = retrieval_metrics

    # -----------------------------------------------------
    # Escalation
    # -----------------------------------------------------

    results[
        "escalation"
    ] = escalation_metrics

    # -----------------------------------------------------
    # System Comparison
    # -----------------------------------------------------

    results[
        "system_comparison"
    ] = {

        "majority_baseline": {

            "accuracy":
                majority_metrics[
                    "accuracy"
                ],

            "macro_f1":
                majority_metrics[
                    "macro_f1"
                ]
        },

        "tfidf_baseline": {

            "accuracy":
                tfidf_metrics[
                    "accuracy"
                ],

            "macro_f1":
                tfidf_metrics[
                    "macro_f1"
                ],

            "retrieval_recall_at_5":
                None
        },

        "apple_support_ai": {

            "accuracy":
                primary_metrics[
                    "accuracy"
                ],

            "macro_f1":
                primary_metrics[
                    "macro_f1"
                ],

            "retrieval_recall_at_5":
                retrieval_metrics[
                    "recall_at_5"
                ],

            "response_quality_mean":
                results.get(
                    "response_quality",
                    {}
                ).get(
                    "mean_score"
                ),

            "escalation_f1":
                escalation_metrics[
                    "f1"
                ]
        }
    }

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    results[
        "reproducibility"
    ] = {

        "seed":
            RANDOM_SEED,

        "golden_set":
            "data/golden_set.csv",

        "evaluation_script":
            "evaluation/evaluate.py",

        "command":
            "python evaluation/evaluate.py"
    }

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    os.makedirs(
        os.path.dirname(
            RESULT_PATH
        ),
        exist_ok=True
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    print(
        f"\nEvaluation results saved to:"
        f"\n{RESULT_PATH}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("APPLE SUPPORT AI - EVALUATION HARNESS")
    print("=" * 60)

    # -----------------------------------------------------
    # Load golden set
    # -----------------------------------------------------

    df = load_golden_set()

    if len(df) < 5:

        raise ValueError(
            "\nGolden set is too small."
        )

    # -----------------------------------------------------
    # Train/test split for baselines
    # -----------------------------------------------------

    try:

        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=df["true_intent"]
        )

    except ValueError:

        print(
            "\n[WARNING] Stratified split failed."
        )

        print(
            "Using random split instead."
        )

        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED
        )

    print(
        f"\nGolden set total: {len(df)}"
    )

    print(
        f"Baseline training examples: {len(train_df)}"
    )

    print(
        f"Baseline test examples: {len(test_df)}"
    )

    # -----------------------------------------------------
    # Primary model
    # -----------------------------------------------------

    primary_model, primary_metrics = (
        evaluate_primary_model(
            df
        )
    )

    # -----------------------------------------------------
    # Majority baseline
    # -----------------------------------------------------

    majority_metrics = (
        evaluate_majority_baseline(
            train_df,
            test_df
        )
    )

    # -----------------------------------------------------
    # TF-IDF baseline
    # -----------------------------------------------------

    _, tfidf_metrics = (
        evaluate_tfidf_baseline(
            train_df,
            test_df
        )
    )

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    retrieval_metrics = (
        evaluate_retrieval(
            df
        )
    )

    # -----------------------------------------------------
    # Escalation
    # -----------------------------------------------------

    escalation_metrics = (
        evaluate_escalation(
            df,
            primary_model
        )
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    update_results(
        primary_metrics,
        majority_metrics,
        tfidf_metrics,
        retrieval_metrics,
        escalation_metrics,
        len(df)
    )

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        "\nResults:"
    )

    print(
        f"Primary Accuracy: "
        f"{primary_metrics['accuracy']}"
    )

    print(
        f"Primary Macro F1: "
        f"{primary_metrics['macro_f1']}"
    )

    print(
        f"TF-IDF Accuracy: "
        f"{tfidf_metrics['accuracy']}"
    )

    print(
        f"TF-IDF Macro F1: "
        f"{tfidf_metrics['macro_f1']}"
    )

    print(
        f"Retrieval Recall@5: "
        f"{retrieval_metrics['recall_at_5']}"
    )

    print(
        f"Escalation F1: "
        f"{escalation_metrics['f1']}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()