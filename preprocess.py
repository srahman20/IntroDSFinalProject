"""
Preprocessing utilities for SMS phishing detection.

This file cleans SMS messages before vectorization and modeling.
"""

import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def download_nltk_resources() -> None:
    """Download required NLTK stopwords if they are not already available."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def clean_text(text: str) -> str:
    """
    Clean a single SMS message.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove punctuation
    4. Remove numbers
    5. Remove stopwords
    6. Apply stemming

    Args:
        text: Raw SMS message.

    Returns:
        Cleaned SMS message as a string.
    """
    download_nltk_resources()

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " urltoken ", text)
    text = re.sub(r"\d+", " numbertoken ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    cleaned_words: List[str] = []
    for word in words:
        if word not in stop_words:
            cleaned_words.append(stemmer.stem(word))

    return " ".join(cleaned_words)


def add_engineered_features(df):
    """
    Add simple engineered features often useful for phishing detection.

    Features:
    - message_length
    - num_exclamation
    - num_digits
    - has_url

    Args:
        df: DataFrame with a 'message' column.

    Returns:
        DataFrame with new feature columns.
    """
    df = df.copy()
    df["message_length"] = df["message"].astype(str).apply(len)
    df["num_exclamation"] = df["message"].astype(str).apply(lambda x: x.count("!"))
    df["num_digits"] = df["message"].astype(str).apply(lambda x: sum(ch.isdigit() for ch in x))
    df["has_url"] = df["message"].astype(str).str.contains(r"http|www|\.com", case=False, regex=True).astype(int)
    return df
