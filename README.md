# Comparative Fashion Trend Forecasting Across Indian and International Markets Using LSTM Networks

## Evaluating Seasonal Effects on Search-Interest Prediction

### Research Objective

This project investigates fashion trend forecasting using historical Google Trends search-interest data. It compares forecasting performance between **India** and **France** using:

1. A standard **LSTM** forecasting model
2. A **seasonality-aware LSTM** with explicit seasonal features
3. A **seasonal naive baseline** for benchmark comparison

The study examines whether explicitly incorporating seasonal information improves forecasting accuracy, and whether results differ between Indian and French fashion markets.

> **Important:** Google Trends measures *normalized search interest* (0–100 scale), not absolute search volume or actual fashion sales. All forecasts and comparisons in this project must be interpreted accordingly. See [Google Trends Limitations](docs/google_trends_limitations.md) for details. *[S1, S2]*

### Project Status

- [x] Project structure and configuration
- [x] Data collection pipeline (pytrends + manual CSV import)
- [x] Data preprocessing and quality validation
- [x] Exploratory visualization
- [x] Documentation and research source registry
- [ ] LSTM model implementation
- [ ] Seasonality-aware LSTM model
- [ ] Seasonal naive baseline
- [ ] Model evaluation and comparison
- [ ] Cross-country analysis
- [ ] Final report

### Setup

#### Prerequisites

- Python 3.9+
- pip

#### Installation

```bash
# Navigate to the project directory
# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Data Collection

**Automated (via pytrends):**

```bash
python -m src.data_collection
```

This will attempt to collect monthly Google Trends data for all configured keywords and countries. Rate limiting and retries are handled automatically.

**Manual fallback:**

If pytrends is blocked or rate-limited, you can export data manually from [Google Trends](https://trends.google.com/trends/explore):

```bash
# Print manual export instructions
python -c "from src.data_collection import print_manual_export_instructions; print_manual_export_instructions()"
```

See [Manual Data Export](#manual-data-export) below for details.

#### 2. Preprocessing

```bash
python -m src.preprocessing
```

This loads raw CSV files, validates the data, removes duplicates, detects missing values, and produces:
- A processed dataset at `data/processed/fashion_trends_processed.csv`
- A data quality report at `results/tables/data_quality_report.txt`

#### 3. Visualization

```bash
python -m src.visualization
```

Generates exploratory plots saved to `results/figures/`.

#### 4. Tests

```bash
python -m pytest tests/ -v
```

### Data

#### Keywords

| Keyword | Rationale |
|---|---|
| streetwear | Urban fashion trend with global and regional variation |
| oversized clothing | Recurring silhouette trend in fashion cycles |
| sustainable fashion | Growing ethical fashion movement |
| traditional clothing | Cultural fashion with potential seasonal patterns |

#### Countries

| Country | Code | Rationale |
|---|---|---|
| India | IN | Large, diverse fashion market with distinct seasonal patterns |
| France | FR | Established fashion capital with different climate and fashion cycles |

#### Data Source

Google Trends (https://trends.google.com). Monthly data, January 2016 – September 2026.

See `docs/research_sources.md` for full source documentation and `docs/google_trends_limitations.md` for methodology notes.

### Manual Data Export

If automated collection is unavailable:

1. Visit https://trends.google.com/trends/explore
2. Set the search term to one keyword
3. Set the time range to "2016-01-01 to 2026-09-30" (or closest available)
4. Set the region to the target country (India or France)
5. Download the CSV using the export button
6. Rename the file following the convention: `gtrends_<keyword>_<country_code>.csv`
7. Place files in `data/raw/`
8. Run preprocessing: `python -m src.preprocessing`

**Critical:** Export each keyword-country pair **separately** to maintain consistent normalization. Do not combine multiple keywords in a single Google Trends query, as Google normalizes them jointly. *[S1]*

### Project Structure

```
DL/
├── data/
│   ├── raw/              # Raw Google Trends CSV files
│   └── processed/        # Cleaned, validated datasets
├── docs/
│   ├── research_sources.md        # Bibliography and source registry
│   ├── experiment_log.md          # Experiment tracking
│   └── google_trends_limitations.md  # Data source methodology notes
├── src/
│   ├── __init__.py
│   ├── config.py         # Central configuration
│   ├── data_collection.py  # Google Trends data collection
│   ├── preprocessing.py  # Data cleaning and validation
│   └── visualization.py  # Exploratory plotting
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py
├── results/
│   ├── figures/          # Generated charts
│   └── tables/           # Data quality reports
├── requirements.txt
├── README.md
└── .gitignore
```

### Limitations

1. **Google Trends normalization:** Search interest values are normalized 0–100 per query. Cross-query absolute comparisons are not valid. *[S1]*
2. **Sampling variability:** Google Trends uses sampled data; results may vary between collection runs. *[S1]*
3. **Cross-country magnitude:** A score of 80 in India and 80 in France does NOT indicate the same absolute search volume. Our analysis compares temporal patterns, not magnitudes.
4. **pytrends reliability:** The pytrends library is unofficial and may be rate-limited or blocked. *[S3]*
5. **Keyword coverage:** The selected keywords represent a subset of fashion trends. Findings may not generalize to all fashion categories.

### Research Integrity

- All external sources are documented in `docs/research_sources.md`.
- No synthetic data is used in research experiments.
- All experimental results are tracked in `docs/experiment_log.md`.
- Claims are distinguished as: established facts (with citations), project design decisions, hypotheses (to be tested), or empirical findings (from our experiments).
- Inline citations (e.g., *[S1]*) reference entries in `docs/research_sources.md`.

### Citation Format

References marked *[S1]*, *[S2]*, etc. correspond to entries in [`docs/research_sources.md`](docs/research_sources.md).

### Next Steps

The next implementation phase will include:

1. **Verify data collection** — confirm that Google Trends data has been successfully collected or manually imported.
2. **Implement LSTM model** — standard LSTM for time-series forecasting of search interest.
3. **Implement seasonality-aware LSTM** — add explicit seasonal features (month, quarter, year).
4. **Implement seasonal naive baseline** — use same-month-last-year as the forecast.
5. **Define evaluation framework** — chronological train/validation/test split with MAE, RMSE, and MAPE metrics.
6. **Run comparative experiments** — evaluate all models across both countries and all keywords.

### License

Academic research project. Not for commercial distribution.
