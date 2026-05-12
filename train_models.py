"""
Main training script for SMS phishing detection.

Run this file from the project root:

    python src/train_models.py
"""

from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from preprocess import clean_text, add_engineered_features
from evaluate import evaluate_model, print_report, save_confusion_matrix
from visualize import (
    save_model_comparison,
    save_label_distribution,
    save_message_length_distribution,
)


DATA_PATHS = [
    Path("data/SMSSpamCollection"),
    Path("data/SMSSpamCollection.txt"),
    Path("data/sms_spam.csv"),
]


def load_dataset() -> pd.DataFrame:
    """
    Load the SMS Spam Collection dataset from the data folder.

    Returns:
        DataFrame with columns: label, message.
    """
    available_path = None
    for path in DATA_PATHS:
        if path.exists():
            available_path = path
            break

    if available_path is None:
        raise FileNotFoundError(
            "Dataset not found. Place the SMS Spam Collection dataset in the data folder as "
            "'SMSSpamCollection', 'SMSSpamCollection.txt', or 'sms_spam.csv'."
        )

    if available_path.suffix == ".csv":
        df = pd.read_csv(available_path)
        if "label" not in df.columns or "message" not in df.columns:
            raise ValueError("CSV file must contain 'label' and 'message' columns.")
        return df[["label", "message"]]

    # Original UCI format is usually tab-separated: label<TAB>message
    df = pd.read_csv(
        available_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the SMS dataset.

    Args:
        df: Raw dataset.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    df = df.dropna()
    df = df.drop_duplicates()

    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])]

    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    df = add_engineered_features(df)
    df["clean_message"] = df["message"].apply(clean_text)

    return df


def train_and_evaluate(df: pd.DataFrame):
    """
    Train and evaluate Naive Bayes, Logistic Regression, and SVM.

    Args:
        df: Preprocessed DataFrame.
    """
    X = df["clean_message"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": LinearSVC(random_state=42),
    }

    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_model(model_name, y_test, y_pred)
        results.append(metrics)

        print_report(model_name, y_test, y_pred)
        save_confusion_matrix(model_name, y_test, y_pred)

        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=5,
            scoring=make_scorer(f1_score, pos_label=1),
        )
        print(f"{model_name} 5-fold CV F1-score: {cv_scores.mean():.4f}")

    results_df = pd.DataFrame(results)
    Path("results").mkdir(exist_ok=True)
    results_df.to_csv("results/model_results.csv", index=False)

    save_model_comparison(results_df)

    print("\n===== Model Results Summary =====")
    print(results_df)


def main():
    """Run the full project pipeline."""
    df = load_dataset()
    df = prepare_data(df)

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    save_label_distribution(df)
    save_message_length_distribution(df)

    print(f"Dataset size after cleaning: {len(df)}")
    print(df["label"].value_counts())

    train_and_evaluate(df)


if __name__ == "__main__":
    main()
