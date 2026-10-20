"""Preprocessing module for fashion trend forecasting.

This module provides functions to:
- Load raw Google Trends CSV files
- Validate data schema and types
- Detect missing values and time-series gaps
- Remove duplicate observations
- Combine individual keyword-country files into a unified processed dataset
- Generate a data quality report

All preprocessing is separated from model training to ensure reproducibility.
Missing values are reported but NOT silently interpolated or filled with zeros.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    COUNTRIES,
    DATA_QUALITY_REPORT_FILENAME,
    DATE_FORMAT,
    KEYWORDS,
    PROCESSED_COLUMNS,
    PROCESSED_DATA_DIR,
    PROCESSED_DATASET_FILENAME,
    RAW_DATA_DIR,
    TABLES_DIR,
    ensure_directories,
    raw_csv_filename,
)

logger = logging.getLogger(__name__)


# ── Loading functions ────────────────────────────────────────────────

def load_raw_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """Load a single raw Google Trends CSV file.

    Parameters
    ----------
    filepath : Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ['date', 'search_interest'], or None if loading fails.
    """
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath.name}")
        return df
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None


def load_all_raw_data(
    keywords: list[str] = None,
    countries: dict[str, str] = None,
    raw_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load all raw CSV files and combine into a single DataFrame.

    Each raw file is loaded, tagged with keyword and country labels,
    and concatenated into a unified DataFrame.

    Parameters
    ----------
    keywords : list of str, optional
        Keywords to load. Defaults to config.KEYWORDS.
    countries : dict, optional
        Country code to name mapping. Defaults to config.COUNTRIES.
    raw_dir : Path, optional
        Directory containing raw CSV files. Defaults to RAW_DATA_DIR.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns: date, keyword, country, search_interest.
        May be empty if no files are found.
    """
    if keywords is None:
        keywords = KEYWORDS
    if countries is None:
        countries = COUNTRIES
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR

    all_frames = []
    files_found = 0
    files_missing = 0

    for keyword in keywords:
        for country_code, country_name in countries.items():
            fname = raw_csv_filename(keyword, country_code)
            fpath = raw_dir / fname

            df = load_raw_csv(fpath)
            if df is not None:
                df["keyword"] = keyword
                df["country"] = country_name
                all_frames.append(df)
                files_found += 1
            else:
                files_missing += 1

    logger.info(f"Loaded {files_found} files, {files_missing} missing")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        return combined
    else:
        logger.warning("No raw data files found. Returning empty DataFrame.")
        return pd.DataFrame(columns=PROCESSED_COLUMNS)


# ── Validation functions ─────────────────────────────────────────────

def validate_columns(df: pd.DataFrame) -> list[str]:
    """Validate that the DataFrame contains the expected columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    list of str
        List of validation issues found. Empty list means all checks passed.
    """
    issues = []
    for col in PROCESSED_COLUMNS:
        if col not in df.columns:
            issues.append(f"Missing expected column: '{col}'")
    for col in df.columns:
        if col not in PROCESSED_COLUMNS:
            issues.append(f"Unexpected column: '{col}'")
    return issues


def validate_data_types(df: pd.DataFrame) -> list[str]:
    """Validate data types of the processed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    list of str
        List of type validation issues.
    """
    issues = []

    if "date" in df.columns:
        try:
            pd.to_datetime(df["date"])
        except Exception:
            issues.append("Column 'date' contains non-parseable date values.")

    if "search_interest" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["search_interest"]):
            issues.append("Column 'search_interest' is not numeric.")
        else:
            if (df["search_interest"].dropna() < 0).any():
                issues.append("Column 'search_interest' contains negative values.")
            if (df["search_interest"].dropna() > 100).any():
                issues.append(
                    "Column 'search_interest' contains values > 100. "
                    "Google Trends scores should be 0-100."
                )

    if "keyword" in df.columns and df["keyword"].isna().any():
        issues.append("Column 'keyword' contains missing values.")

    if "country" in df.columns and df["country"].isna().any():
        issues.append("Column 'country' contains missing values.")

    return issues


# ── Cleaning functions ───────────────────────────────────────────────

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the 'date' column to datetime and format as ISO date string.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'date' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed and formatted dates.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["date"] = df["date"].dt.strftime(DATE_FORMAT)
    return df


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the DataFrame by date, keyword, and country.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sort.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame.
    """
    df = df.copy()
    sort_cols = [c for c in ["date", "keyword", "country"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate observations based on date, keyword, and country.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame potentially containing duplicates.

    Returns
    -------
    tuple of (pd.DataFrame, int)
        Cleaned DataFrame and number of duplicates removed.
    """
    key_cols = [c for c in ["date", "keyword", "country"] if c in df.columns]
    n_before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} duplicate observations.")
    return df, n_removed


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and report missing values in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.

    Returns
    -------
    pd.DataFrame
        Summary of missing values per column.
    """
    missing = df.isna().sum()
    missing_pct = (df.isna().sum() / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": missing_pct,
        "total_rows": len(df),
    })
    return summary


def detect_time_series_gaps(
    df: pd.DataFrame,
    freq: str = "MS",
) -> pd.DataFrame:
    """Detect gaps in the monthly time series for each keyword-country group.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame with 'date', 'keyword', 'country' columns.
    freq : str
        Expected frequency. Default is 'MS' (month start).

    Returns
    -------
    pd.DataFrame
        DataFrame listing missing months per keyword-country group.
    """
    if df.empty:
        return pd.DataFrame(columns=["keyword", "country", "missing_date"])

    df_temp = df.copy()
    df_temp["date"] = pd.to_datetime(df_temp["date"])

    gap_records = []

    for (keyword, country), group in df_temp.groupby(["keyword", "country"]):
        dates = group["date"].sort_values()
        if len(dates) < 2:
            continue

        expected = pd.date_range(
            start=dates.min(),
            end=dates.max(),
            freq=freq,
        )

        actual_months = dates.dt.to_period("M").unique()
        expected_months = expected.to_period("M").unique()

        missing_months = expected_months.difference(actual_months)

        for m in missing_months:
            gap_records.append({
                "keyword": keyword,
                "country": country,
                "missing_date": str(m),
            })

    gaps_df = pd.DataFrame(gap_records)
    if not gaps_df.empty:
        logger.warning(f"Found {len(gaps_df)} missing months in the time series.")
    else:
        logger.info("No time-series gaps detected.")

    return gaps_df


# ── Data quality report ──────────────────────────────────────────────

def generate_data_quality_report(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> str:
    """Generate a comprehensive data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        The processed dataset.
    output_dir : Path, optional
        Where to save the report. Defaults to TABLES_DIR.

    Returns
    -------
    str
        The report as a text string.
    """
    if output_dir is None:
        output_dir = TABLES_DIR

    ensure_directories()

    lines = []
    lines.append("=" * 70)
    lines.append("DATA QUALITY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # 1. Overview
    lines.append("\n-- OVERVIEW --")
    lines.append(f"Total observations: {len(df)}")
    if not df.empty:
        lines.append(f"Columns: {list(df.columns)}")
        lines.append(
            f"Keywords: {sorted(df['keyword'].unique()) if 'keyword' in df.columns else 'N/A'}"
        )
        lines.append(
            f"Countries: {sorted(df['country'].unique()) if 'country' in df.columns else 'N/A'}"
        )
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            lines.append(
                f"Date range: {dates.min().strftime(DATE_FORMAT)} to "
                f"{dates.max().strftime(DATE_FORMAT)}"
            )

    # 2. Column validation
    lines.append("\n-- COLUMN VALIDATION --")
    col_issues = validate_columns(df)
    if col_issues:
        for issue in col_issues:
            lines.append(f"  WARNING: {issue}")
    else:
        lines.append("  OK: All expected columns present.")

    # 3. Data type validation
    lines.append("\n-- DATA TYPE VALIDATION --")
    type_issues = validate_data_types(df)
    if type_issues:
        for issue in type_issues:
            lines.append(f"  WARNING: {issue}")
    else:
        lines.append("  OK: All data types valid.")

    # 4. Missing values
    lines.append("\n-- MISSING VALUES --")
    missing_summary = detect_missing_values(df)
    for col_name, row in missing_summary.iterrows():
        status = "WARNING" if row["missing_count"] > 0 else "OK"
        lines.append(
            f"  {status} {col_name}: {int(row['missing_count'])} missing "
            f"({row['missing_percent']:.1f}%)"
        )

    # 5. Time series gaps
    lines.append("\n-- TIME SERIES GAPS --")
    if not df.empty and "date" in df.columns:
        gaps = detect_time_series_gaps(df)
        if gaps.empty:
            lines.append("  OK: No gaps detected in monthly time series.")
        else:
            lines.append(f"  WARNING: Found {len(gaps)} missing months:")
            for _, row in gaps.iterrows():
                lines.append(
                    f"    - {row['keyword']} / {row['country']}: {row['missing_date']}"
                )
    else:
        lines.append("  No data available for gap analysis.")

    # 6. Per-group summary
    lines.append("\n-- OBSERVATIONS PER KEYWORD-COUNTRY --")
    if not df.empty and "keyword" in df.columns and "country" in df.columns:
        summary = df.groupby(["keyword", "country"]).agg(
            n_obs=("search_interest", "count"),
            n_missing=("search_interest", lambda x: x.isna().sum()),
            mean_interest=("search_interest", "mean"),
            min_interest=("search_interest", "min"),
            max_interest=("search_interest", "max"),
        ).round(2)
        lines.append(summary.to_string())

        summary_path = output_dir / "observation_summary.csv"
        summary.to_csv(summary_path)
        lines.append(f"\n  Summary table saved to: {summary_path}")
    else:
        lines.append("  No data available for summary.")

    # 7. Search interest distribution
    lines.append("\n-- SEARCH INTEREST DISTRIBUTION --")
    if not df.empty and "search_interest" in df.columns:
        si = df["search_interest"].dropna()
        lines.append(f"  Count (non-null): {len(si)}")
        lines.append(f"  Mean: {si.mean():.2f}")
        lines.append(f"  Std:  {si.std():.2f}")
        lines.append(f"  Min:  {si.min():.2f}")
        lines.append(f"  25%:  {si.quantile(0.25):.2f}")
        lines.append(f"  50%:  {si.quantile(0.50):.2f}")
        lines.append(f"  75%:  {si.quantile(0.75):.2f}")
        lines.append(f"  Max:  {si.max():.2f}")
        lines.append("\n  NOTE: These are normalized Google Trends scores (0-100),")
        lines.append("  NOT absolute search volumes. Cross-query comparisons")
        lines.append("  of absolute magnitude are not methodologically valid.")

    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    report_path = output_dir / DATA_QUALITY_REPORT_FILENAME
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Data quality report saved to {report_path}")

    return report_text


# ── Main preprocessing pipeline ─────────────────────────────────────

def run_preprocessing_pipeline(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Run the complete preprocessing pipeline.

    Steps:
    1. Load all raw CSV files
    2. Validate columns and types
    3. Parse dates
    4. Sort chronologically
    5. Remove duplicates
    6. Detect missing values
    7. Detect time-series gaps
    8. Save processed dataset
    9. Generate data quality report

    Parameters
    ----------
    raw_dir : Path, optional
        Directory containing raw CSV files.
    output_dir : Path, optional
        Directory to save the processed dataset.

    Returns
    -------
    pd.DataFrame or None
        The processed dataset, or None if no data is available.
    """
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR

    ensure_directories()
    logger.info("Starting preprocessing pipeline...")

    # Step 1: Load raw data
    logger.info("Step 1: Loading raw data...")
    df = load_all_raw_data(raw_dir=raw_dir)

    if df.empty:
        logger.warning(
            "No raw data files found. Please run data collection first "
            "or place manually exported CSV files in data/raw/."
        )
        return None

    # Step 2: Validate columns
    logger.info("Step 2: Validating columns...")
    col_issues = validate_columns(df)
    for issue in col_issues:
        logger.warning(f"Column validation: {issue}")

    # Step 3: Validate data types
    logger.info("Step 3: Validating data types...")
    type_issues = validate_data_types(df)
    for issue in type_issues:
        logger.warning(f"Type validation: {issue}")

    # Step 4: Parse dates
    logger.info("Step 4: Parsing dates...")
    df = parse_dates(df)

    # Step 5: Sort chronologically
    logger.info("Step 5: Sorting chronologically...")
    df = sort_chronologically(df)

    # Step 6: Remove duplicates
    logger.info("Step 6: Removing duplicates...")
    df, n_dupes = remove_duplicates(df)

    # Step 7: Detect missing values
    logger.info("Step 7: Checking missing values...")
    missing = detect_missing_values(df)
    for col_name, row in missing.iterrows():
        if row["missing_count"] > 0:
            logger.warning(
                f"Column '{col_name}': {int(row['missing_count'])} missing values "
                f"({row['missing_percent']:.1f}%)"
            )

    # Step 8: Detect time-series gaps
    logger.info("Step 8: Detecting time-series gaps...")
    gaps = detect_time_series_gaps(df)

    # Step 9: Save processed dataset
    logger.info("Step 9: Saving processed dataset...")
    processed_path = output_dir / PROCESSED_DATASET_FILENAME
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed dataset saved to {processed_path} ({len(df)} rows)")

    # Step 10: Generate data quality report
    logger.info("Step 10: Generating data quality report...")
    report = generate_data_quality_report(df)
    print("\n" + report)

    return df


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = run_preprocessing_pipeline()
    if result is not None:
        print(f"\nPreprocessing complete. {len(result)} observations processed.")
    else:
        print("\nPreprocessing failed. No data available.")
