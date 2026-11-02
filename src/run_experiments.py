"""Main experiment runner.

Executes the full forecasting pipeline:
1. Load processed data
2. Run all three models on each keyword-country series
3. Generate comparison table
4. Generate all forecast visualizations
5. Print summary

Usage:
    python -m src.run_experiments
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import (
    COUNTRIES,
    KEYWORDS,
    PROCESSED_DATA_DIR,
    PROCESSED_DATASET_FILENAME,
    FIGURES_DIR,
    TABLES_DIR,
    ensure_directories,
)
from src.evaluation import build_comparison_table
from src.models import run_single_series
from src.forecast_plots import (
    plot_actual_vs_predicted,
    plot_model_comparison_bar,
    plot_cross_country_patterns,
)

logger = logging.getLogger(__name__)


def main():
    """Run the complete experiment pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ensure_directories()

    # Suppress TensorFlow info/warning messages
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # ── Step 1: Load processed data ──────────────────────────────
    processed_path = PROCESSED_DATA_DIR / PROCESSED_DATASET_FILENAME
    if not processed_path.exists():
        logger.error(
            f"Processed dataset not found at {processed_path}. "
            "Run preprocessing first: python -m src.preprocessing"
        )
        sys.exit(1)

    df = pd.read_csv(processed_path)
    logger.info(f"Loaded {len(df)} observations from {processed_path}")

    # ── Step 2: Run models on each keyword-country series ────────
    all_results = []
    for keyword in KEYWORDS:
        for country_code, country_name in COUNTRIES.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Running models: '{keyword}' in {country_name}")
            logger.info(f"{'='*60}")

            result = run_single_series(
                df=df,
                keyword=keyword,
                country=country_name,
                epochs=100,
                seed=42,
            )
            all_results.append(result)

    # Filter out None results
    valid_results = [r for r in all_results if r is not None]

    if not valid_results:
        logger.error("No models ran successfully. Check data availability.")
        sys.exit(1)

    # ── Step 3: Build comparison table ───────────────────────────
    comparison_df = build_comparison_table(valid_results)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print("=" * 70)

    # ── Step 4: Generate visualizations ──────────────────────────
    logger.info("\nGenerating forecast visualizations...")

    generated_plots = []

    # Actual vs Predicted plots
    for r in valid_results:
        path = plot_actual_vs_predicted(r)
        if path:
            generated_plots.append(path)

    # Model comparison bar charts
    bar_paths = plot_model_comparison_bar(comparison_df)
    generated_plots.extend(bar_paths)

    # Cross-country pattern comparisons
    cross_paths = plot_cross_country_patterns(valid_results)
    generated_plots.extend(cross_paths)

    # ── Step 5: Print summary ────────────────────────────────────
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Series processed: {len(valid_results)}/{len(all_results)}")
    print(f"Plots generated:  {len(generated_plots)}")
    print(f"Results table:    {TABLES_DIR / 'model_comparison.csv'}")
    print(f"Figures directory: {FIGURES_DIR}")

    print(f"\n--- Per-Series Best Model ---")
    for r in valid_results:
        models = {
            "Seasonal Naive": r["seasonal_naive"]["mae"],
            "Std LSTM": r["standard_lstm"]["mae"],
            "Sea LSTM": r["seasonal_lstm"]["mae"],
        }
        best = min(models, key=models.get)
        print(
            f"  {r['keyword']:25s} | {r['country']:8s} | "
            f"Best: {best} (MAE={models[best]:.2f})"
        )

    print(f"\n{'='*70}")
    print("Experiment complete.")

    return valid_results, comparison_df


if __name__ == "__main__":
    main()
