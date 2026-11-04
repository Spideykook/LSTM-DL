# Comparative Fashion Trend Forecasting Across Indian and International Markets Using LSTM Networks

## Evaluating Seasonal Effects on Search-Interest Prediction

### Research Objective

This project investigates fashion trend forecasting using historical Google Trends search-interest data. It compares forecasting performance between **India** and **France** using:

1. A **seasonal naive baseline** for benchmark comparison
2. A standard **LSTM** forecasting model
3. A **seasonality-aware LSTM** with explicit monthly sine/cosine features

The study examines whether explicitly incorporating seasonal information improves forecasting accuracy, and whether results differ between Indian and French fashion markets.

> **Important:** Google Trends measures *normalized search interest* (0–100 scale), not absolute search volume or actual fashion sales. All forecasts and comparisons in this project must be interpreted accordingly. See [Google Trends Limitations](docs/google_trends_limitations.md) for details. *[S1, S2]*

### Project Status

- [x] Project structure and configuration
- [x] Data collection pipeline (pytrends + manual CSV import)
- [x] Data preprocessing and quality validation
- [x] Exploratory visualization
- [x] Documentation and research source registry
- [x] Seasonal naive baseline
- [x] Standard LSTM model
- [x] Seasonality-aware LSTM model
- [x] Model evaluation and comparison
- [x] Cross-country analysis
- [ ] Final research paper

### Experimental Results Summary

All three models were evaluated on 8 keyword-country series (4 keywords × 2 countries). Split: 70% train / 15% validation / 15% test (chronological). Scalers fit on training data only.

| Keyword | Country | Best Model | MAE | RMSE |
|---|---|---|---|---|
| streetwear | India | Standard LSTM | 8.53 | 18.59 |
| streetwear | France | Seasonality-Aware LSTM | 9.79 | 17.39 |
| oversized clothing | India | Standard LSTM | 8.10 | 17.69 |
| oversized clothing | France | Seasonal Naive | 7.40 | 23.67 |
| sustainable fashion | India | Seasonality-Aware LSTM | 13.80 | 26.11 |
| sustainable fashion | France | Seasonality-Aware LSTM | 10.35 | 22.34 |
| traditional clothing | India | Seasonality-Aware LSTM | 12.26 | 17.74 |
| traditional clothing | France | Standard LSTM | 13.16 | 22.55 |

Full results: `results/tables/model_comparison.csv`

> **Key finding:** No single model dominates. The Seasonality-Aware LSTM won 4/8 series, Standard LSTM won 3/8, and Seasonal Naive won 1/8. LSTM models do not uniformly outperform the baseline.

### Setup

#### Prerequisites

- Python 3.9+
- pip

#### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### 1. Data Collection

```bash
python -m src.data_collection
```

Or export manually from [Google Trends](https://trends.google.com/trends/explore) — run `python -c "from src.data_collection import print_manual_export_instructions; print_manual_export_instructions()"` for instructions.

#### 2. Preprocessing

```bash
python -m src.preprocessing
```

Outputs: `data/processed/fashion_trends_processed.csv`, `results/tables/data_quality_report.txt`

#### 3. Exploratory Visualization

```bash
python -m src.visualization
```

#### 4. Run Forecasting Experiments

```bash
python -m src.run_experiments
```

Trains all three models on each keyword-country series, generates forecast plots and comparison charts, and saves metrics to `results/tables/model_comparison.csv`.

#### 5. Tests

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

Google Trends (https://trends.google.com). Monthly data, January 2016 – September 2026. 129 observations per series, 1032 total.

### Project Structure

```
DL/
├── data/
│   ├── raw/                  # Raw Google Trends CSV files + metadata JSON
│   └── processed/            # Cleaned, validated datasets
├── docs/
│   ├── research_sources.md   # Bibliography and source registry
│   ├── experiment_log.md     # Experiment tracking
│   └── google_trends_limitations.md
├── src/
│   ├── __init__.py
│   ├── config.py             # Central configuration
│   ├── data_collection.py    # Google Trends data collection
│   ├── preprocessing.py      # Data cleaning and validation
│   ├── visualization.py      # Exploratory plotting
│   ├── models.py             # Seasonal Naive, Standard LSTM, Seasonality-Aware LSTM
│   ├── evaluation.py         # MAE, RMSE, comparison table
│   ├── forecast_plots.py     # Forecast visualizations
│   └── run_experiments.py    # Main experiment runner
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_forecasting.py
├── results/
│   ├── figures/              # All generated charts
│   └── tables/               # model_comparison.csv, data quality reports
├── requirements.txt
├── README.md
└── .gitignore
```

### Methodology

- **Seasonal Naive Baseline:** Predicts using the value from the same month in the previous year. *[S5]*
- **Standard LSTM:** Single-layer LSTM (50 units) with 12-month lookback, MinMaxScaler, early stopping. *[S4]*
- **Seasonality-Aware LSTM:** Same LSTM architecture with additional sine/cosine month-of-year features (3 input features instead of 1).
- **Evaluation:** Chronological 70/15/15 split. Walk-forward (rolling) test predictions. MAE and RMSE metrics.
- **Data leakage prevention:** Scalers fit on training data only. No future information in sequences.

### Limitations

1. **Google Trends normalization:** Values are 0–100 per query; cross-query absolute comparisons are not valid. *[S1]*
2. **Sampling variability:** Google Trends uses sampled data; results may vary between runs. *[S1]*
3. **Cross-country magnitude:** Equal scores across countries do NOT indicate equal absolute search volume. Analysis compares temporal patterns only.
4. **Small dataset:** 129 monthly observations per series limits deep learning effectiveness.
5. **CPU training:** Models trained on CPU; no GPU acceleration used.
6. **Keyword coverage:** Selected keywords represent a subset of fashion trends.

### Research Integrity

- All sources documented in `docs/research_sources.md`.
- No synthetic data in research experiments. Test files use clearly labelled synthetic fixtures.
- All results in `docs/experiment_log.md` are from actual model runs.
- No assumption that LSTM outperforms baselines — results reported as observed.

### Next Steps

1. Write the final research paper based on actual experimental results.
2. Explore hyperparameter tuning and additional architectures.
3. Add confidence intervals or multiple random seed runs.

### License

Academic research project. Not for commercial distribution.
