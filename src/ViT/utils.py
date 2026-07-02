"""
Logging setup and evaluation metrics.
"""

import logging
import sys

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def setup_logger(log_file="training.log", level=logging.INFO):
    """Configures a logger that writes to both console and a file."""
    logger = logging.getLogger("ViT")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        # Avoid duplicate handlers if setup_logger is called more than once.
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def compute_metrics(y_true, y_pred, class_names=None):
    """
    Computes accuracy, precision, recall, F1 (macro-averaged for
    multi-class), and the confusion matrix.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    return metrics


def format_metrics(metrics, prefix=""):
    """Human-readable one-line summary for logging."""
    return (
        f"{prefix}acc={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )
