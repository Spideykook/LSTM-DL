# Comparative Fashion Trend Forecasting Across Indian and French Markets Using LSTM Networks

## Evaluating Seasonal Effects on Search-Interest Prediction

### Research Objective

This project investigates fashion trend forecasting using historical Google Trends search-interest data. It compares forecasting performance between **India** and **France** using:

1. A standard **LSTM** forecasting model
2. A **seasonality-aware LSTM** with explicit seasonal features
3. A **seasonal naive baseline** for benchmark comparison

The study examines whether explicitly incorporating seasonal information improves forecasting accuracy, and whether results differ between Indian and French fashion markets.

> **Important:** Google Trends measures *normalized search interest* (0–100 scale), not absolute search volume or actual fashion sales. All forecasts and comparisons in this project must be interpreted accordingly. See [Google Trends Limitations](docs/google_trends_limitations.md) for details. *[S1, S2]*

### Project Status

All three models were run on all 8 keyword-country series (see `docs/experiment_log.md`, EXP-003).

- [x] Data collection, preprocessing, independent re-verification (`docs/data_verification.md`)
- [x] Seasonal naive baseline, standard LSTM, seasonality-aware LSTM
- [x] MAE/RMSE evaluation per country and keyword; sensitivity analysis; seed variability
- [x] Forecast figures and tables in `results/`
- [x] Paper (`paper/`) and presentation (`presentation/`)

### Headline results (mean over 8 series, test 2025-02 to 2026-09; index points)

| Model | MAE | RMSE | Series won (MAE) |
|---|---|---|---|
| Seasonal naive | 12.82 | 23.44 | 1 |
| Standard LSTM | 11.00 | 21.21 | 4 |
| Seasonality-aware LSTM | 10.80 | 21.36 | 3 |

Both LSTMs beat the seasonal naive baseline on average, but **seasonal sin/cos features do not give a reliable
improvement over the standard LSTM** (difference smaller than seed-to-seed noise, mixed sign across series).
**Data caveat:** all 8 series peak at exactly 100 in 2026-02 (unexplained; see `docs/data_verification.md`);
results are reported with and without that month. Full tables: `results/tables/`.

### Setup

Python 3.9+ (developed on 3.12.3; TensorFlow 2.21.0, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0).

```bash
pip install -r requirements.txt
```

### Reproduce the experiments (about 12 min on one CPU core)

```bash
python -m src.preprocessing          # optional: rebuild processed data + quality report from data/raw
python -m pytest tests/ -q           # 34 tests
python -m src.run_experiments        # trains all models, writes results/tables/*.csv and results/figures/*.png
python -m src.supplementary          # test-period grid figure + persistence diagnostic
```

Seeds are fixed (42-46), but TensorFlow results can differ slightly across hardware/versions.
Optional extras: `python -m src.data_collection` (re-collect via pytrends; values may differ because Google samples data)
and `python -m src.visualization` (exploratory plots).

### Outputs

| Path | Content |
|---|---|
| `results/tables/model_comparison.csv` | MAE/RMSE per model, keyword, country (primary) |
| `results/tables/model_comparison_excl_2026-02.csv` | Same, 2026-02 excluded from scoring (sensitivity) |
| `results/tables/predictions.csv` | Test actuals and predictions of all models |
| `results/tables/seed_variability.csv` | Per-seed LSTM MAE/RMSE mean and SD |
| `results/tables/summary_by_model*.csv`, `supplementary_persistence.csv` | Aggregates, persistence diagnostic |
| `results/figures/` | Per-series forecast plots, bar charts, `test_period_grid.png` (exploratory plots: `python -m src.visualization`) |
| `docs/methodology.md`, `docs/data_verification.md`, `docs/experiment_log.md` | Method, data checks, log |
| `paper/`, `presentation/` | Paper and slides (editable + PDF) |

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

Google Trends (https://trends.google.com). Monthly data, January 2016 – September 2026 (129 observations per series).

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
data/raw, data/processed     Google Trends CSVs, metadata, processed dataset
docs/                        methodology, data verification, experiment log, sources, GT limitations
src/                         config, data_collection, preprocessing, models, evaluation, run_experiments,
                             forecast_plots, supplementary, visualization
tests/                       34 unit tests (synthetic data, test-only)
results/{tables,figures}     experiment outputs
paper/, presentation/        deliverables
```

### Limitations

1. **Normalized indices only:** Google Trends gives relative search interest (0-100, each query scaled to its own peak), not
   search volume or sales. Series are normalized independently, so levels and errors are not comparable across series or countries;
   cross-country conclusions concern dynamics only. *[S1, S2]*
2. **Unexplained anomaly:** all series hit 100 in 2026-02 and shift upward in 2025-11, inside the test period. Provenance could not be re-verified
   (Google Trends not reachable from the analysis sandbox). Results are reported with and without 2026-02.
3. **Degenerate series:** `oversized clothing`-France is 98% zeros (training segment all zero); its results are uninformative.
4. **Small data:** 129 monthly points per series (90 for training), 20 test months, one chronological split, 8 series: no significance testing is attempted.
5. **Walk-forward one-step forecasts only;** multi-step horizons not evaluated. No hyperparameter tuning.
6. **Sampling:** Google Trends is sampled; re-collection may give different values. pytrends is unofficial. *[S1, S3]*
7. **Keyword coverage:** four English queries; not representative of all fashion categories or of Hindi/French-language searches.

### Research Integrity

- All external sources are documented in `docs/research_sources.md`.
- No synthetic data is used in research experiments.
- All experimental results are tracked in `docs/experiment_log.md`.
- Claims are distinguished as: established facts (with citations), project design decisions, hypotheses (to be tested), or empirical findings (from our experiments).
- Inline citations (e.g., *[S1]*) reference entries in `docs/research_sources.md`.

### Citation Format

References marked *[S1]*, *[S2]*, etc. correspond to entries in [`docs/research_sources.md`](docs/research_sources.md).

### Future Work

Re-collect data and investigate the 2025-11 / 2026-02 behaviour; rolling-origin evaluation and multi-step horizons;
repeated queries to average Google's sampling noise; a persistence/ETS/SARIMA baseline set; a contemporaneous-month
feature ablation; a pooled (global) LSTM across series; statistical testing on more series.

### License

Academic research project. Not for commercial distribution.
