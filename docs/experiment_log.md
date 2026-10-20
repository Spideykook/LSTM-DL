# Experiment Log

This file records all experiments, data collection runs, and significant pipeline executions for the Fashion Trend Forecasting project.

**Purpose:** Ensure reproducibility by documenting datasets, configurations, and results for each experiment.

**Last updated:** 2026-10-20

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

*No further experiments have been conducted yet. This file will be updated as the project progresses.*
