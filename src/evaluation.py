"""Evaluation metrics and comparison utilities.

Provides MAE, RMSE calculation and model comparison table generation.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import TABLES_DIR, ensure_directories

logger = logging.getLogger(__name__)


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error.

    Parameters
    ----------
    actual : np.ndarray
        Ground truth values.
    predicted : np.ndarray
        Predicted values.

    Returns
    -------
    float
        MAE value.
    """
    return float(np.mean(np.abs(actual - predicted)))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error.

    Parameters
    ----------
    actual : np.ndarray
        Ground truth values.
    predicted : np.ndarray
        Predicted values.

    Returns
    -------
    float
        RMSE value.
    """
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def build_comparison_table(
    all_results: list[dict],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build a comparison table from all experiment results.

    Parameters
    ----------
    all_results : list of dict
        Results from run_single_series for each keyword-country.
    output_path : Path, optional
        Where to save the CSV. Defaults to results/tables/model_comparison.csv.

    Returns
    -------
    pd.DataFrame
        Comparison table with MAE and RMSE for each model, keyword, country.
    """
    if output_path is None:
        ensure_directories()
        output_path = TABLES_DIR / "model_comparison.csv"

    rows = []
    for r in all_results:
        if r is None:
            continue
        for model_name, model_key in [
            ("Seasonal Naive", "seasonal_naive"),
            ("Standard LSTM", "standard_lstm"),
            ("Seasonality-Aware LSTM", "seasonal_lstm"),
        ]:
            rows.append({
                "keyword": r["keyword"],
                "country": r["country"],
                "model": model_name,
                "n_train": r["n_train"],
                "n_val": r["n_val"],
                "n_test": r["n_test"],
                "MAE": round(r[model_key]["mae"], 4),
                "RMSE": round(r[model_key]["rmse"], 4),
            })

    comparison_df = pd.DataFrame(rows)

    if not comparison_df.empty:
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Model comparison table saved to {output_path}")

    return comparison_df
