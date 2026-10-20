"""Data collection module for Google Trends fashion search-interest data.

This module provides functions to collect monthly search-interest data from
Google Trends using the pytrends library. It implements rate limiting,
error handling, and metadata logging.

IMPORTANT METHODOLOGICAL NOTES:
- Google Trends returns normalized search-interest values (0-100), not
  absolute search volumes.
- Each query is independently normalized by Google to its own peak within
  the requested timeframe and geography.
- Scores from different queries or geographies are NOT directly comparable
  in absolute terms.
- This pipeline collects each keyword-country pair separately. Cross-keyword
  and cross-country comparisons must account for independent normalization.

Collection strategy:
  - Each (keyword, country) pair is queried individually over the full
    timeframe to obtain a consistently normalized series.
  - Metadata is saved alongside each raw CSV to document provenance.
  - If pytrends is unavailable or blocked, the module provides instructions
    for manual CSV export from Google Trends.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    COUNTRIES,
    KEYWORDS,
    PYTRENDS_BACKOFF_FACTOR,
    PYTRENDS_HOST_LANGUAGE,
    PYTRENDS_REQUEST_DELAY,
    PYTRENDS_RETRIES,
    PYTRENDS_TIMEZONE,
    RAW_DATA_DIR,
    TIMEFRAME,
    ensure_directories,
    metadata_filename,
    raw_csv_filename,
)

logger = logging.getLogger(__name__)


# ── pytrends availability check ──────────────────────────────────────

def _check_pytrends_available() -> bool:
    """Check whether the pytrends library is installed and importable."""
    try:
        import pytrends  # noqa: F401
        return True
    except ImportError:
        return False


PYTRENDS_AVAILABLE = _check_pytrends_available()


# ── Core collection functions ────────────────────────────────────────

def collect_single_keyword(
    keyword: str,
    country_code: str,
    timeframe: str = TIMEFRAME,
    output_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Collect Google Trends data for a single keyword and country.

    Parameters
    ----------
    keyword : str
        The search term to query.
    country_code : str
        ISO 3166-1 alpha-2 country code (e.g., 'IN', 'FR').
    timeframe : str
        Google Trends timeframe string (e.g., '2016-01-01 2026-09-30').
    output_dir : Path, optional
        Directory to save the raw CSV and metadata. Defaults to RAW_DATA_DIR.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ['date', 'search_interest'] if successful,
        None if collection failed.

    Notes
    -----
    Each keyword-country query is independently normalized by Google Trends.
    The returned search_interest values (0-100) represent relative interest
    within that specific query, NOT absolute search volume.
    """
    if not PYTRENDS_AVAILABLE:
        logger.error(
            "pytrends is not installed. Install with: pip install pytrends\n"
            "Alternatively, export CSV manually from https://trends.google.com"
        )
        return None

    from pytrends.request import TrendReq

    if output_dir is None:
        output_dir = RAW_DATA_DIR

    country_name = COUNTRIES.get(country_code, country_code)
    logger.info(f"Collecting data for '{keyword}' in {country_name} ({country_code})")
    logger.info(f"Timeframe: {timeframe}")

    df = None
    last_error = None

    for attempt in range(1, PYTRENDS_RETRIES + 1):
        try:
            pytrends_client = TrendReq(
                hl=PYTRENDS_HOST_LANGUAGE,
                tz=PYTRENDS_TIMEZONE,
            )

            pytrends_client.build_payload(
                kw_list=[keyword],
                timeframe=timeframe,
                geo=country_code,
            )

            raw_df = pytrends_client.interest_over_time()

            if raw_df is None or raw_df.empty:
                logger.warning(
                    f"No data returned for '{keyword}' in {country_name}. "
                    f"The term may have insufficient search volume in this region."
                )
                _save_metadata(
                    keyword=keyword,
                    country_code=country_code,
                    country_name=country_name,
                    timeframe=timeframe,
                    output_dir=output_dir,
                    success=False,
                    note="No data returned; insufficient search volume or term not recognized.",
                )
                return None

            # Process the raw response
            if "isPartial" in raw_df.columns:
                raw_df = raw_df.drop(columns=["isPartial"])

            # Rename the keyword column to 'search_interest'
            if keyword in raw_df.columns:
                df = raw_df[[keyword]].copy()
                df = df.rename(columns={keyword: "search_interest"})
            else:
                df = raw_df.iloc[:, 0:1].copy()
                df.columns = ["search_interest"]

            df.index.name = "date"
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

            logger.info(
                f"Successfully collected {len(df)} observations for "
                f"'{keyword}' in {country_name}"
            )
            break

        except Exception as e:
            last_error = e
            wait_time = PYTRENDS_BACKOFF_FACTOR ** attempt
            logger.warning(
                f"Attempt {attempt}/{PYTRENDS_RETRIES} failed for '{keyword}' "
                f"in {country_name}: {type(e).__name__}: {e}"
            )
            if attempt < PYTRENDS_RETRIES:
                logger.info(f"Retrying in {wait_time:.0f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(
                    f"All {PYTRENDS_RETRIES} attempts failed for '{keyword}' "
                    f"in {country_name}. Last error: {e}"
                )

    if df is not None:
        csv_path = output_dir / raw_csv_filename(keyword, country_code)
        df.to_csv(csv_path, index=False)
        logger.info(f"Raw data saved to {csv_path}")

        _save_metadata(
            keyword=keyword,
            country_code=country_code,
            country_name=country_name,
            timeframe=timeframe,
            output_dir=output_dir,
            success=True,
            note=f"Collected {len(df)} monthly observations via pytrends.",
            n_observations=len(df),
            date_range_actual=(
                f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}"
                if len(df) > 0 else "N/A"
            ),
        )
        return df
    else:
        _save_metadata(
            keyword=keyword,
            country_code=country_code,
            country_name=country_name,
            timeframe=timeframe,
            output_dir=output_dir,
            success=False,
            note=f"Collection failed after {PYTRENDS_RETRIES} attempts. Last error: {last_error}",
        )
        return None


def collect_all_data(
    keywords: list[str] = None,
    countries: dict[str, str] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Collect Google Trends data for all keyword-country combinations.

    Parameters
    ----------
    keywords : list of str, optional
        Keywords to query. Defaults to config.KEYWORDS.
    countries : dict, optional
        Country code to name mapping. Defaults to config.COUNTRIES.
    output_dir : Path, optional
        Output directory. Defaults to RAW_DATA_DIR.

    Returns
    -------
    dict
        Results summary with keys:
        - 'successful': list of (keyword, country_code) tuples collected
        - 'failed': list of (keyword, country_code) tuples that failed
        - 'dataframes': dict mapping (keyword, country_code) to DataFrames
    """
    if keywords is None:
        keywords = KEYWORDS
    if countries is None:
        countries = COUNTRIES
    if output_dir is None:
        output_dir = RAW_DATA_DIR

    ensure_directories()

    results = {
        "successful": [],
        "failed": [],
        "dataframes": {},
    }

    total_queries = len(keywords) * len(countries)
    current_query = 0

    for keyword in keywords:
        for country_code, country_name in countries.items():
            current_query += 1
            logger.info(
                f"\n{'='*60}\n"
                f"Query {current_query}/{total_queries}: '{keyword}' in {country_name}\n"
                f"{'='*60}"
            )

            df = collect_single_keyword(
                keyword=keyword,
                country_code=country_code,
                output_dir=output_dir,
            )

            if df is not None:
                results["successful"].append((keyword, country_code))
                results["dataframes"][(keyword, country_code)] = df
            else:
                results["failed"].append((keyword, country_code))

            # Rate limiting between requests
            if current_query < total_queries:
                logger.info(f"Waiting {PYTRENDS_REQUEST_DELAY}s before next request...")
                time.sleep(PYTRENDS_REQUEST_DELAY)

    logger.info(f"\n{'='*60}")
    logger.info(f"Collection complete: {len(results['successful'])}/{total_queries} successful")
    if results["failed"]:
        logger.warning(f"Failed queries: {results['failed']}")
    logger.info(f"{'='*60}")

    return results


# ── Metadata helpers ─────────────────────────────────────────────────

def _save_metadata(
    keyword: str,
    country_code: str,
    country_name: str,
    timeframe: str,
    output_dir: Path,
    success: bool,
    note: str,
    n_observations: int = 0,
    date_range_actual: str = "N/A",
) -> None:
    """Save collection metadata as a JSON file alongside the raw data."""
    meta = {
        "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "collection_method": "pytrends" if PYTRENDS_AVAILABLE else "manual_export",
        "country_code": country_code,
        "country_name": country_name,
        "keyword": keyword,
        "date_range_requested": timeframe,
        "date_range_actual": date_range_actual,
        "n_observations": n_observations,
        "success": success,
        "normalization_note": (
            "Google Trends normalizes search interest to a 0-100 scale relative "
            "to the peak within the queried timeframe and geography. Values from "
            "different queries are independently normalized and not directly "
            "comparable in absolute terms."
        ),
        "limitations": [
            "Values represent relative search interest, not absolute search volume.",
            "Each query is independently normalized by Google.",
            "Low-volume search terms may return zero or no data.",
            "Google may sample data differently across requests.",
            "Results may vary slightly between collection runs.",
        ],
        "note": note,
    }

    meta_path = output_dir / metadata_filename(keyword, country_code)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to {meta_path}")


# ── Manual export instructions ───────────────────────────────────────

def print_manual_export_instructions() -> None:
    """Print instructions for manually exporting data from Google Trends.

    Use this when pytrends is unavailable, blocked, or rate-limited.
    """
    instructions = """
========================================================================
          MANUAL GOOGLE TRENDS DATA EXPORT INSTRUCTIONS
========================================================================

  If pytrends is unavailable or blocked, you can manually export
  data from Google Trends:

  1. Go to https://trends.google.com/trends/explore

  2. For EACH keyword-country combination, configure:
     - Search term: enter one keyword at a time
     - Time range: Jan 2016 - Sep 2026 (or latest available)
     - Region: select the specific country (India or France)
     - Category: leave as 'All categories'
     - Search type: 'Web Search'

  3. Click the download button to export CSV.

  4. Rename the downloaded file following this convention:
     gtrends_<keyword>_<country_code>.csv
     Examples:
       gtrends_streetwear_in.csv
       gtrends_streetwear_fr.csv
       gtrends_oversized_clothing_in.csv
       gtrends_sustainable_fashion_fr.csv
       gtrends_traditional_clothing_in.csv

  5. Place all CSV files in: data/raw/

  6. The Google Trends CSV format typically has:
     - Header rows (skip first 1-2 rows)
     - A 'Month' or 'Week' column with dates
     - A column with the keyword name containing interest values

  7. After placing files, run the preprocessing pipeline:
     python -m src.preprocessing

  IMPORTANT NOTES:
  - Export each keyword-country pair SEPARATELY to maintain
    consistent normalization within each series.
  - Do NOT combine multiple keywords in a single query, as
    Google normalizes them jointly, changing individual values.
  - Document the exact date you exported the data.
  - Values are normalized 0-100 relative to peak interest.

========================================================================
"""
    print(instructions)

    print("\nExpected files:")
    for keyword in KEYWORDS:
        for code in COUNTRIES:
            fname = raw_csv_filename(keyword, code)
            print(f"  data/raw/{fname}")


# ── Manual CSV import ────────────────────────────────────────────────

def import_manual_csv(
    filepath: Path,
    keyword: str,
    country_code: str,
    output_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Import a manually exported Google Trends CSV file.

    Google Trends CSVs typically have 1-3 header rows before the actual data.
    This function attempts to detect and parse the format automatically.

    Parameters
    ----------
    filepath : Path
        Path to the manually exported CSV file.
    keyword : str
        The search keyword this data represents.
    country_code : str
        ISO country code (e.g., 'IN', 'FR').
    output_dir : Path, optional
        Where to save the standardized CSV. Defaults to RAW_DATA_DIR.

    Returns
    -------
    pd.DataFrame or None
        Standardized DataFrame with ['date', 'search_interest'] columns.
    """
    if output_dir is None:
        output_dir = RAW_DATA_DIR

    country_name = COUNTRIES.get(country_code, country_code)
    logger.info(f"Importing manual CSV for '{keyword}' in {country_name}: {filepath}")

    if not Path(filepath).exists():
        logger.error(f"File not found: {filepath}")
        return None

    try:
        # Try to auto-detect the header row
        df = None
        for skiprows in [0, 1, 2, 3]:
            try:
                candidate = pd.read_csv(filepath, skiprows=skiprows)
                first_col = candidate.columns[0]
                if len(candidate) > 0:
                    try:
                        pd.to_datetime(candidate[first_col].iloc[0])
                        df = candidate
                        logger.info(f"Detected data starting at row {skiprows + 1}")
                        break
                    except (ValueError, TypeError):
                        continue
            except Exception:
                continue

        if df is None:
            logger.error(f"Could not parse date column in {filepath}")
            return None

        # Standardize columns
        date_col = df.columns[0]
        value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        result = pd.DataFrame()
        result["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

        # Handle '<1' values that Google Trends sometimes returns
        values = df[value_col].copy()
        if values.dtype == object:
            values = values.replace("<1", "0.5")
            values = pd.to_numeric(values, errors="coerce")
        result["search_interest"] = values

        # Save standardized version
        csv_path = output_dir / raw_csv_filename(keyword, country_code)
        result.to_csv(csv_path, index=False)
        logger.info(f"Standardized data saved to {csv_path} ({len(result)} observations)")

        _save_metadata(
            keyword=keyword,
            country_code=country_code,
            country_name=country_name,
            timeframe=(
                f"{result['date'].iloc[0]} to {result['date'].iloc[-1]}"
                if len(result) > 0 else "N/A"
            ),
            output_dir=output_dir,
            success=True,
            note=f"Imported from manual CSV export: {Path(filepath).name}",
            n_observations=len(result),
            date_range_actual=(
                f"{result['date'].iloc[0]} to {result['date'].iloc[-1]}"
                if len(result) > 0 else "N/A"
            ),
        )

        return result

    except Exception as e:
        logger.error(f"Failed to import {filepath}: {e}")
        return None


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ensure_directories()

    if not PYTRENDS_AVAILABLE:
        logger.error("pytrends is not installed.")
        print_manual_export_instructions()
    else:
        logger.info("Starting automated data collection via pytrends...")
        results = collect_all_data()

        if results["failed"]:
            logger.warning(
                f"\n{len(results['failed'])} queries failed. "
                f"You may need to export these manually from Google Trends."
            )
            print_manual_export_instructions()

        if results["successful"]:
            logger.info(
                f"\nSuccessfully collected data for {len(results['successful'])} "
                f"keyword-country combinations."
            )
