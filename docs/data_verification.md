# Data Verification (performed 2026-11-02)

Independent re-check of `data/raw/*.csv` and `data/processed/fashion_trends_processed.csv`
(the repository's own quality report was not simply trusted).

| Check | Result |
|---|---|
| Series / rows | 8 series x 129 monthly rows (2016-01-01 to 2026-09-01); 1,032 processed rows |
| Duplicated dates | 0 |
| Missing values | 0 |
| Frequency | Strictly monthly (month start), no gaps |
| Range | 0-100 integers, every series has max = 100 |
| Raw vs processed | Identical row counts, no duplicate rows |

## Provenance
`data/raw/metadata_*.json` records `collection_method: pytrends`, collected 2026-10-20, one query per
keyword-country pair. **I could not re-collect or independently confirm these values**: the analysis sandbox
cannot reach trends.google.com. Provenance therefore rests on the repository metadata. Values were used exactly
as supplied; nothing was imputed, smoothed, replaced or removed.

## Anomalies found (flagged, not corrected)
1. **Simultaneous peak:** all 8 series equal exactly 100 in **2026-02**, i.e. each series' maximum falls in the
   same month across unrelated keywords and two countries. This is unusual for independent organic series and may
   reflect a Google Trends sampling/data artifact (or an event affecting every query). Cause is unknown.
2. **Common level shift:** all series jump upward in **2025-11** (e.g. streetwear-India 14 -> 37).
3. **Sparse/degenerate series:** `oversized clothing`-France has 126/129 zeros (non-zero only 2025-11, 2026-02, 2026-03);
   its training segment (2016-01 to 2023-06) is constant 0. Google reports 0 for volume below its threshold,
   so this is "below threshold", not "no searches". Its scores are uninformative.
4. Because the 100-peak is the series maximum, Google's per-query rescaling compresses all earlier values
   (training-period values are low relative to 100).

Anomalies 1-2 fall in the **test period** (2025-02 to 2026-09) and dominate test error. To make this visible,
every result is reported twice: on the full test set (primary, pre-specified) and with 2026-02 excluded from
**scoring only** (sensitivity; model inputs unchanged).

## Independent normalisation
Each series is normalised by Google separately (its own peak = 100). Levels are not comparable across keywords or
countries; only within-series dynamics are. MAE/RMSE are in index points of each series' own scale, so averages
across series are descriptive summaries, not statistically meaningful pooled errors.
