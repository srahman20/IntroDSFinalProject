"""
Visualization utilities for SMS phishing detection.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_model_comparison(results_df: pd.DataFrame, output_path="results/figures/model_comparison.png"):
    """
    Save a bar chart comparing model F1-scores.

    Args:
        results_df: DataFrame containing model evaluation metrics.
        output_path: Output file path for the figure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="model", y="f1_score")
    plt.title("Model Comparison by F1-Score")
    plt.xlabel("Model")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_label_distribution(df: pd.DataFrame, output_path="results/figures/label_distribution.png"):
    """
    Save a plot showing class distribution between ham and spam.

    Args:
        df: DataFrame containing a 'label' column.
        output_path: Output file path for the figure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="label")
    plt.title("Distribution of SMS Labels")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_message_length_distribution(df: pd.DataFrame, output_path="results/figures/message_length_distribution.png"):
    """
    Save a histogram showing message length distribution by label.

    Args:
        df: DataFrame containing 'label' and 'message_length' columns.
        output_path: Output file path for the figure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="message_length", hue="label", bins=40, kde=True)
    plt.title("Message Length Distribution")
    plt.xlabel("Message Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
