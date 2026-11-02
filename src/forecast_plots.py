"""Forecast visualization functions.

Generates actual-vs-predicted plots, model comparison bar charts,
and cross-country pattern comparisons.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.config import (
    COLOR_PALETTE,
    FIGURE_DPI,
    FIGURE_SIZE,
    FIGURES_DIR,
    ensure_directories,
)

logger = logging.getLogger(__name__)


def plot_actual_vs_predicted(
    result: dict,
    save_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Plot actual vs predicted for all three models on a single series.

    Parameters
    ----------
    result : dict
        Output from models.run_single_series().
    save_dir : Path, optional
        Directory to save figures.

    Returns
    -------
    Path or None
        Path to saved figure.
    """
    if save_dir is None:
        save_dir = FIGURES_DIR
    ensure_directories()

    keyword = result["keyword"]
    country = result["country"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    test_dates = pd.to_datetime(result["test_dates"])
    train_dates = pd.to_datetime(result["train_dates"])
    val_dates = pd.to_datetime(result["val_dates"])
    actual = result["test_actual"]

    models = [
        ("Seasonal Naive", "seasonal_naive", "#2ca02c"),
        ("Standard LSTM", "standard_lstm", "#1f77b4"),
        ("Seasonality-Aware LSTM", "seasonal_lstm", "#d62728"),
    ]

    for ax, (name, key, color) in zip(axes, models):
        preds = result[key]["predictions"]
        mae = result[key]["mae"]
        rmse = result[key]["rmse"]

        # Plot training and validation as context
        ax.plot(train_dates, result["train_actual"],
                color="#999999", linewidth=0.8, alpha=0.5, label="Training data")
        ax.plot(val_dates, result["val_actual"],
                color="#666666", linewidth=0.8, alpha=0.5, label="Validation data")

        # Actual test values
        ax.plot(test_dates, actual,
                color="black", linewidth=1.5, marker="o", markersize=3,
                label="Actual")

        # Predicted values
        ax.plot(test_dates, preds,
                color=color, linewidth=1.5, linestyle="--", marker="s",
                markersize=3, label=f"{name}")

        ax.set_title(f"{name}  |  MAE={mae:.2f}  RMSE={rmse:.2f}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Search Interest (0-100)", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date", fontsize=10)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

    fig.suptitle(
        f"Forecast Comparison: '{keyword}' in {country}\n"
        f"(Test period: {test_dates.min().strftime('%Y-%m')} to {test_dates.max().strftime('%Y-%m')})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout()

    safe_kw = keyword.replace(" ", "_").lower()
    safe_co = country.replace(" ", "_").lower()
    filename = f"forecast_{safe_kw}_{safe_co}.png"
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info(f"Saved forecast plot: {filepath}")
    plt.close(fig)
    return filepath


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    save_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate bar chart comparing MAE and RMSE across models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from evaluation.build_comparison_table().
    save_dir : Path, optional
        Directory to save figures.

    Returns
    -------
    list of Path
        Paths to saved figures.
    """
    if save_dir is None:
        save_dir = FIGURES_DIR
    ensure_directories()

    paths = []

    for metric in ["MAE", "RMSE"]:
        fig, ax = plt.subplots(figsize=(14, 6))

        # Create grouped bar chart
        pivot = comparison_df.pivot_table(
            index=["keyword", "country"],
            columns="model",
            values=metric,
        )

        # Ensure consistent column order
        col_order = ["Seasonal Naive", "Standard LSTM", "Seasonality-Aware LSTM"]
        pivot = pivot[[c for c in col_order if c in pivot.columns]]

        pivot.plot(kind="bar", ax=ax, width=0.75,
                   color=["#2ca02c", "#1f77b4", "#d62728"])

        ax.set_title(f"Model Comparison: {metric} by Keyword and Country",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("(Keyword, Country)", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.legend(title="Model", fontsize=9)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", fontsize=7, padding=2)

        plt.tight_layout()

        filename = f"model_comparison_{metric.lower()}.png"
        filepath = save_dir / filename
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Saved comparison chart: {filepath}")
        plt.close(fig)
        paths.append(filepath)

    return paths


def plot_cross_country_patterns(
    all_results: list[dict],
    save_dir: Optional[Path] = None,
) -> list[Path]:
    """Plot India vs France temporal patterns per keyword.

    Shows the actual test-period values side by side to compare
    temporal dynamics (not absolute magnitudes).

    Parameters
    ----------
    all_results : list of dict
        Results from all series runs.
    save_dir : Path, optional
        Directory to save figures.

    Returns
    -------
    list of Path
        Paths to saved figures.
    """
    if save_dir is None:
        save_dir = FIGURES_DIR
    ensure_directories()

    paths = []

    # Group results by keyword
    by_keyword = {}
    for r in all_results:
        if r is None:
            continue
        kw = r["keyword"]
        if kw not in by_keyword:
            by_keyword[kw] = {}
        by_keyword[kw][r["country"]] = r

    for keyword, country_results in by_keyword.items():
        if len(country_results) < 2:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        for ax_idx, (metric_label, metric_key) in enumerate([("Actual", "test_actual"), ("Best Model Predictions", None)]):
            ax = axes[ax_idx]

            for country, r in country_results.items():
                dates = pd.to_datetime(r["test_dates"])
                color = COLOR_PALETTE.get(country, "#333333")

                if metric_key is not None:
                    values = r[metric_key]
                    ax.plot(dates, values, color=color, linewidth=1.5,
                            marker="o", markersize=3, label=country)
                else:
                    # Find best model (lowest MAE)
                    best_model = min(
                        ["seasonal_naive", "standard_lstm", "seasonal_lstm"],
                        key=lambda m: r[m]["mae"],
                    )
                    best_name = {
                        "seasonal_naive": "Naive",
                        "standard_lstm": "Std LSTM",
                        "seasonal_lstm": "Sea LSTM",
                    }[best_model]
                    values = r[best_model]["predictions"]
                    ax.plot(dates, values, color=color, linewidth=1.5,
                            marker="s", markersize=3,
                            label=f"{country} ({best_name})")

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_ylabel("Search Interest (0-100)", fontsize=9)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        fig.suptitle(
            f"Cross-Country Pattern Comparison: '{keyword}'\n"
            f"(Temporal dynamics only — magnitudes are not comparable across countries)",
            fontsize=12, fontweight="bold",
        )

        plt.tight_layout()

        safe_kw = keyword.replace(" ", "_").lower()
        filename = f"cross_country_{safe_kw}.png"
        filepath = save_dir / filename
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Saved cross-country plot: {filepath}")
        plt.close(fig)
        paths.append(filepath)

    return paths
