# Research Sources Registry

This file serves as the central bibliography and evidence registry for the Fashion Trend Forecasting project. Every external source used to justify technical decisions, explain methodology, or support research claims is documented here.

**Last updated:** 2026-10-20

---

## Source Categories

- **[OD]** Official Documentation
- **[PR]** Peer-Reviewed Paper
- **[AP]** Academic Preprint
- **[OT]** Other (blog, tutorial, technical report)

---

## S1. Google Trends Data Documentation

- **Title:** Google Trends Help – How Trends data is adjusted
- **Author/Organization:** Google
- **Publication Date:** Continuously updated
- **Source Type:** [OD] Official Documentation
- **URL:** https://support.google.com/trends/answer/4365533
- **Date Accessed:** 2026-10-20
- **Verification Status:** UNVERIFIED — URL accessibility could not be confirmed at time of project setup due to environment constraints. The URL is the canonical help page referenced in Google Trends' own interface.
- **Claims Supported:**
  - Google Trends normalizes search data to a 0–100 scale relative to the peak within the selected timeframe and geography.
  - Each data point is divided by the total searches of the geography and time range to produce relative popularity.
  - A value of 100 represents peak popularity for the term; 50 means half as popular.
  - Results from different queries are independently normalized and should not be compared in absolute terms across separate queries.
- **Limitations:**
  - Google does not disclose the exact sampling methodology.
  - Results may vary between collection runs due to sampling.
  - Low-volume terms may be reported as 0 or suppressed entirely.

## S2. Google Trends FAQ

- **Title:** Google Trends FAQ
- **Author/Organization:** Google
- **Source Type:** [OD] Official Documentation
- **URL:** https://support.google.com/trends/answer/4359550
- **Date Accessed:** 2026-10-20
- **Verification Status:** UNVERIFIED — standard FAQ URL for Google Trends.
- **Claims Supported:**
  - Google Trends data reflects the proportion of searches, not absolute search count.
  - Repeated searches by the same user over a short period are excluded.

## S3. pytrends Library Documentation

- **Title:** pytrends – Unofficial API for Google Trends
- **Author/Organization:** General Mills (original); community maintained
- **Source Type:** [OD] Official library documentation (open source)
- **URL:** https://github.com/GeneralMills/pytrends
- **Date Accessed:** 2026-10-20
- **Verification Status:** UNVERIFIED — GitHub repository URL is well-known but not accessed during setup.
- **Claims Supported:**
  - pytrends provides a Python interface to query Google Trends data programmatically.
  - It supports specifying geography, timeframe, and keyword parameters.
  - The library returns pandas DataFrames with normalized interest values.
- **Limitations:**
  - Unofficial library; may break if Google changes their Trends interface.
  - Subject to rate limiting by Google.
  - Not guaranteed to return identical results to the Google Trends web interface.

## S4. LSTM for Time Series Forecasting

- **Title:** Long Short-Term Memory
- **Authors:** Sepp Hochreiter, Jürgen Schmidhuber
- **Publication Date:** 1997
- **Source Type:** [PR] Peer-Reviewed Paper
- **Journal:** Neural Computation, 9(8), 1735–1780
- **DOI:** 10.1162/neco.1997.9.8.1735
- **Verification Status:** UNVERIFIED — this is the seminal LSTM paper; DOI is well-known but not accessed during setup.
- **Claims Supported:**
  - LSTM networks address the vanishing gradient problem in recurrent neural networks.
  - LSTMs use gating mechanisms (input, forget, output gates) to learn long-term dependencies.
- **Relevance to Project:** Foundation architecture for our forecasting models. Actual performance must be empirically tested, not assumed.

## S5. Seasonal Naive Baseline and Forecasting Principles

- **Title:** Forecasting: Principles and Practice (3rd edition)
- **Authors:** Rob J. Hyndman, George Athanasopoulos
- **Publication Date:** 2021
- **Source Type:** [OT] Open-access textbook
- **URL:** https://otexts.com/fpp3/
- **Verification Status:** UNVERIFIED — well-known forecasting textbook; URL not accessed.
- **Claims Supported:**
  - Seasonal patterns are recurring fluctuations that repeat at fixed intervals (e.g., annually).
  - Seasonal naive forecasting uses the observation from the same season in the previous year as the forecast.
  - It is a standard baseline for evaluating seasonal time-series models.
- **Relevance to Project:** Justifies the use of a seasonal naive baseline and the concept of seasonality in fashion search interest.

---

## Unverified Claims and Limitations

The following claims used in the project documentation are based on widely accepted knowledge but have not been independently verified against primary sources during this project setup:

1. **Google Trends normalization methodology** — based on Google's own help pages (S1, S2). Exact sampling details are not publicly disclosed.
2. **pytrends reliability** — based on community documentation (S3). No formal reliability assessment conducted.
3. **LSTM suitability for time series** — LSTMs are widely used for sequential data (S4), but their superiority over simpler methods for our specific dataset is a hypothesis to be tested, not an established fact.
4. **Seasonal naive baseline** — described in standard forecasting literature (S5), but its appropriateness depends on whether our data exhibits genuine seasonality.

---

## Notes on Source Verification

- All URLs listed above are canonical references from well-known sources.
- Due to environment constraints during initial project setup, direct web access to verify these URLs was not performed.
- Sources should be re-verified when internet access is available.
- No fabricated sources, authors, DOIs, or claims are included in this registry.
- If a source is later found to be inaccurate or unavailable, it will be updated with a correction note.
