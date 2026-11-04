# Experiment Log

This file records all experiments, data collection runs, and significant pipeline executions for the Fashion Trend Forecasting project.

**Purpose:** Ensure reproducibility by documenting datasets, configurations, and results for each experiment.

**Last updated:** 2026-11-02

---

## Log Format

Each entry should include:

| Field | Description |
|---|---|
| **Date** | Date of the experiment |
| **Phase** | Project phase (data collection, preprocessing, model training, evaluation) |
| **Description** | What was done |
| **Dataset** | Files and date ranges used |
| **Configuration** | Relevant settings (e.g., keywords, countries, model hyperparameters) |
| **Random Seed** | If applicable |
| **Status** | Completed / Failed / Partial |
| **Results** | Key outputs or metrics |
| **Errors/Limitations** | Any issues encountered |
| **Notes** | Additional observations |

---

## Experiments

### EXP-001: Project Setup and Initial Data Collection

- **Date:** 2026-10-20
- **Phase:** Data Collection
- **Description:** Initial project structure created. Automated data collection via pytrends executed successfully. All 8 keyword-country series collected.
- **Dataset:** Monthly search interest, Jan 2016 – Sep 2026 (129 observations per series)
- **Configuration:**
  - Keywords: streetwear, oversized clothing, sustainable fashion, traditional clothing
  - Countries: India (IN), France (FR)
  - Timeframe: 2016-01-01 to 2026-09-30
  - Collection method: pytrends 4.9.2 (automated)
  - Request delay: 15 seconds between queries
  - Retries: 3 (none needed)
- **Status:** COMPLETED
- **Results:**
  - 8/8 queries successful
  - 129 monthly observations per series
  - 1032 total observations in processed dataset
  - 0 missing values, 0 time-series gaps, 0 duplicates
  - All data validation checks passed
  - 15 exploratory figures generated
- **Data Summary (mean search interest per series):**

  | Keyword | France | India |
  |---|---|---|
  | oversized clothing | 1.15 | 12.91 |
  | streetwear | 29.44 | 10.51 |
  | sustainable fashion | 4.57 | 11.57 |
  | traditional clothing | 11.98 | 37.43 |

- **Errors/Limitations:**
  - Exit code 1 reported by PowerShell due to stderr logging (not an actual Python error)
  - One minor matplotlib deprecation warning for `get_cmap` (fixed)
  - Google Trends normalization means cross-query absolute comparisons are not valid
- **Notes:**
  - All raw CSVs and metadata JSONs saved in `data/raw/`
  - Processed dataset at `data/processed/fashion_trends_processed.csv`
  - Data quality report at `results/tables/data_quality_report.txt`
  - 15 figures saved in `results/figures/`
  - Tests: 15/15 passed

---

### EXP-002: Data re-verification
- **Date:** 2026-11-02. **Status:** COMPLETED. Re-checked dates, NaNs, duplicates, frequency, provenance (see `data_verification.md`).
- **Finding:** all 8 series peak at 100 in 2026-02 and jump in 2025-11; `oversized clothing`-France is 98% zeros. Data NOT altered.

### EXP-003: Three-model comparison (primary experiment)
- **Date:** 2026-11-02. **Command:** `python -m src.run_experiments` then `python -m src.supplementary` (about 12 min on 1 CPU core).
- **Environment:** Python 3.12.3, TensorFlow(-cpu) 2.21.0, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0.
- **Config:** see `methodology.md` (lookback 12, LSTM(50), 5 seeds 42-46, 70/15/15 split, walk-forward one-step test).
- **Status:** COMPLETED; all 3 models ran on all 8 series (24 model-series results; 80 LSTM trainings). Tests: 34/34 pass.
- **Results (mean over 8 series, test 2025-02..2026-09):**

  | Model | MAE | RMSE | MAE wins |
  |---|---|---|---|
  | Seasonal naive | 12.82 | 23.44 | 1 |
  | Standard LSTM | 11.00 | 21.21 | 4 |
  | Seasonality-aware LSTM | 10.80 | 21.36 | 3 |

  Excluding 2026-02 from scoring: MAE 9.12 / 7.50 / 7.20; RMSE 14.01 / 11.74 / 11.58 (naive / standard / seasonal).
  Persistence diagnostic: MAE 11.45 (full), 8.69 (excl. 2026-02).
- **Interpretation:** LSTMs beat seasonal naive in 7/8 series; the seasonal-feature gain over the standard LSTM
  (about 0.2 MAE) is smaller than seed-to-seed SD (about 0.5) and has mixed sign across series, so it is **not established**.
- **Limitations:** single split, 20 test months dominated by a regime shift and the anomalous 2026-02 spike.
- **Code changes:** fair walk-forward naive; target-month seasonal features; multi-seed training; batched prediction;
  new tables (`predictions.csv`, `seed_variability.csv`, `summary_by_model*.csv`, sensitivity, persistence).

### EXP-004: Presentation-quality figures and paper revision (no model rerun)
- **Date:** 2026-11-02. `python -m src.supplementary` re-plots `test_period_grid.png` and new `mae_by_series.png` from the saved `predictions.csv` / `model_comparison.csv`; `supplementary_persistence.csv` verified byte-identical. No results changed.
- Paper retitled "...Indian and French Markets..."; contribution statement, writing and discussion revised; all numbers re-checked against `results/tables/`. India naive mean MAE corrected to 13.72 (rounding of unrounded values; was 13.73 in an earlier draft).
