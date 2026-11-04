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


# ── Predictions, seed-level and sensitivity tables ───────────────────

MODEL_KEYS = [("Seasonal Naive", "seasonal_naive"), ("Standard LSTM", "standard_lstm"),
              ("Seasonality-Aware LSTM", "seasonal_lstm")]
# Month where all 8 series equal 100 simultaneously (see docs/data_verification.md)
ANOMALY_MONTH = "2026-02-01"


def build_predictions_table(all_results: list[dict], output_path: Optional[Path] = None) -> pd.DataFrame:
    """Long table of test-period actuals and predictions for every model."""
    rows = []
    for r in all_results:
        for i, d in enumerate(pd.to_datetime(r["test_dates"])):
            row = {"keyword": r["keyword"], "country": r["country"], "date": d.strftime("%Y-%m-%d"),
                   "actual": r["test_actual"][i]}
            for name, key in MODEL_KEYS:
                row[key] = round(float(r[key]["predictions"][i]), 4)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_path or TABLES_DIR / "predictions.csv", index=False)
    return out


def build_seed_table(all_results: list[dict], output_path: Optional[Path] = None) -> pd.DataFrame:
    """Per-seed LSTM metrics plus seed mean/std (ensemble metrics are in model_comparison.csv)."""
    rows = []
    for r in all_results:
        for name, key in MODEL_KEYS[1:]:
            m, q = np.array(r[key]["seed_mae"]), np.array(r[key]["seed_rmse"])
            rows.append({"keyword": r["keyword"], "country": r["country"], "model": name,
                         "n_seeds": len(m), "MAE_mean": m.mean(), "MAE_std": m.std(ddof=1),
                         "RMSE_mean": q.mean(), "RMSE_std": q.std(ddof=1)})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(output_path or TABLES_DIR / "seed_variability.csv", index=False)
    return out


def build_sensitivity_table(pred_df: pd.DataFrame, output_path: Optional[Path] = None) -> pd.DataFrame:
    """Test metrics with the anomalous month excluded from scoring only (inputs unchanged)."""
    sub = pred_df[pred_df["date"] != ANOMALY_MONTH]
    rows = []
    for (kw, co), g in sub.groupby(["keyword", "country"]):
        for name, key in MODEL_KEYS:
            rows.append({"keyword": kw, "country": co, "model": name, "n_test": len(g),
                         "MAE": calculate_mae(g["actual"].values, g[key].values),
                         "RMSE": calculate_rmse(g["actual"].values, g[key].values)})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(output_path or TABLES_DIR / "model_comparison_excl_2026-02.csv", index=False)
    return out


def summarize_by_model(comp: pd.DataFrame, output_path: Optional[Path] = None) -> pd.DataFrame:
    """Mean MAE/RMSE per model (overall and by country) and per-series win counts."""
    g = comp.groupby(["country", "model"])[["MAE", "RMSE"]].mean()
    allm = comp.groupby("model")[["MAE", "RMSE"]].mean()
    allm.index = pd.MultiIndex.from_product([["All"], allm.index])
    out = pd.concat([allm, g])
    best = comp.loc[comp.groupby(["keyword", "country"])["MAE"].idxmin()]["model"].value_counts()
    out["MAE_wins"] = [best.get(m, 0) if c == "All" else np.nan for c, m in out.index]
    out = out.round(4)
    out.to_csv(output_path or TABLES_DIR / "summary_by_model.csv")
    return out
