# Google Trends: Methodology and Limitations

This document describes the characteristics and limitations of Google Trends data as used in this project.

## What Google Trends Measures

Google Trends reports **normalized search interest**, not absolute search volume. Each data point represents the relative popularity of a search term within the specified geography and time range.

- A value of **100** indicates peak popularity for the term in that query.
- A value of **50** means the term was half as popular as the peak.
- A value of **0** means insufficient data for the term.

*Source: [S1] in docs/research_sources.md*

## Normalization

- Each query is independently normalized by Google. The same term queried over different time ranges or geographies may yield different normalized values.
- Combining data from separate queries requires acknowledging that the scales may not be directly comparable.
- Our collection strategy queries each (keyword, country) pair individually over a consistent timeframe to maintain internal consistency within each series.

## Sampling

- Google Trends uses a sample of search data, not the complete search log.
- Results may vary slightly between collection runs due to sampling variability.
- Google does not publicly disclose the exact sampling methodology.

## Missing and Suppressed Data

- Search terms with very low volume in a region may return 0 or no data.
- A value of 0 does not necessarily mean zero searches occurred — it may indicate volume below the reporting threshold.
- Some terms may have no data in certain countries if search volume is negligible.

## Cross-Country Comparisons

Google Trends normalizes each query relative to its own geography's search patterns. Therefore:

- A score of 80 in India and 80 in France does **NOT** mean the same absolute number of searches.
- Cross-country comparisons can indicate relative trends and seasonal patterns within each country, but cannot establish which country has higher absolute search interest.
- Our analysis compares **pattern shapes and temporal dynamics**, not absolute magnitudes.

## Temporal Resolution

- Monthly data is available for queries spanning more than ~5 years.
- Weekly data is available for queries spanning less than ~5 years.
- Daily data is available for queries spanning less than ~9 months.
- Our project uses monthly data for the full 2016–2026 period.

## Reproducibility Considerations

- Data collected via pytrends may differ slightly from data exported through the Google Trends web interface.
- Collection date should be recorded as results may change over time.
- We store raw data alongside metadata documenting collection parameters.

## References

See `docs/research_sources.md` for full source citations.
