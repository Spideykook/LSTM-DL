"""Tests for the preprocessing module.

These tests use synthetic data created specifically for testing.
The synthetic data is clearly labelled and is NOT used in research experiments.
"""

import pandas as pd
import pytest

from src.preprocessing import (
    detect_missing_values,
    detect_time_series_gaps,
    parse_dates,
    remove_duplicates,
    sort_chronologically,
    validate_columns,
    validate_data_types,
)


# ── Test fixtures (synthetic data for testing only) ──────────────────


@pytest.fixture
def valid_df():
    """Create a valid synthetic DataFrame for testing. NOT research data."""
    return pd.DataFrame({
        "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
        "keyword": ["streetwear"] * 4,
        "country": ["India"] * 4,
        "search_interest": [45, 52, 38, 61],
    })


@pytest.fixture
def df_with_duplicates():
    """Synthetic DataFrame with duplicate rows for testing."""
    return pd.DataFrame({
        "date": ["2023-01-01", "2023-01-01", "2023-02-01"],
        "keyword": ["streetwear"] * 3,
        "country": ["India"] * 3,
        "search_interest": [45, 45, 52],
    })


@pytest.fixture
def df_with_missing():
    """Synthetic DataFrame with missing values for testing."""
    return pd.DataFrame({
        "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
        "keyword": ["streetwear"] * 3,
        "country": ["India"] * 3,
        "search_interest": [45, None, 38],
    })


@pytest.fixture
def df_with_gap():
    """Synthetic DataFrame with a gap in the monthly series."""
    return pd.DataFrame({
        "date": ["2023-01-01", "2023-02-01", "2023-04-01"],  # March missing
        "keyword": ["streetwear"] * 3,
        "country": ["India"] * 3,
        "search_interest": [45, 52, 61],
    })


# ── Tests ────────────────────────────────────────────────────────────


class TestValidateColumns:
    def test_valid_columns(self, valid_df):
        issues = validate_columns(valid_df)
        assert issues == []

    def test_missing_column(self):
        df = pd.DataFrame({"date": [], "keyword": []})
        issues = validate_columns(df)
        assert any("country" in i for i in issues)
        assert any("search_interest" in i for i in issues)

    def test_extra_column(self, valid_df):
        df = valid_df.copy()
        df["extra_col"] = 0
        issues = validate_columns(df)
        assert any("extra_col" in i for i in issues)


class TestValidateDataTypes:
    def test_valid_types(self, valid_df):
        issues = validate_data_types(valid_df)
        assert issues == []

    def test_negative_interest(self):
        df = pd.DataFrame({
            "date": ["2023-01-01"],
            "keyword": ["test"],
            "country": ["India"],
            "search_interest": [-5],
        })
        issues = validate_data_types(df)
        assert any("negative" in i for i in issues)

    def test_over_100_interest(self):
        df = pd.DataFrame({
            "date": ["2023-01-01"],
            "keyword": ["test"],
            "country": ["India"],
            "search_interest": [150],
        })
        issues = validate_data_types(df)
        assert any("100" in i for i in issues)


class TestParseDates:
    def test_parse_iso_dates(self, valid_df):
        result = parse_dates(valid_df)
        assert result["date"].iloc[0] == "2023-01-01"

    def test_parse_various_formats(self):
        df = pd.DataFrame({
            "date": ["2023-01-15", "2023/02/15"],
            "keyword": ["test", "test"],
            "country": ["India", "India"],
            "search_interest": [50, 60],
        })
        result = parse_dates(df)
        assert len(result) == 2


class TestSortChronologically:
    def test_sorts_by_date(self):
        df = pd.DataFrame({
            "date": ["2023-03-01", "2023-01-01", "2023-02-01"],
            "keyword": ["a"] * 3,
            "country": ["India"] * 3,
            "search_interest": [1, 2, 3],
        })
        result = sort_chronologically(df)
        assert list(result["date"]) == ["2023-01-01", "2023-02-01", "2023-03-01"]


class TestRemoveDuplicates:
    def test_removes_duplicates(self, df_with_duplicates):
        result, n_removed = remove_duplicates(df_with_duplicates)
        assert n_removed == 1
        assert len(result) == 2

    def test_no_duplicates(self, valid_df):
        result, n_removed = remove_duplicates(valid_df)
        assert n_removed == 0
        assert len(result) == len(valid_df)


class TestDetectMissingValues:
    def test_no_missing(self, valid_df):
        summary = detect_missing_values(valid_df)
        assert summary["missing_count"].sum() == 0

    def test_with_missing(self, df_with_missing):
        summary = detect_missing_values(df_with_missing)
        assert summary.loc["search_interest", "missing_count"] == 1


class TestDetectTimeSeriesGaps:
    def test_no_gaps(self, valid_df):
        gaps = detect_time_series_gaps(valid_df)
        assert len(gaps) == 0

    def test_with_gap(self, df_with_gap):
        gaps = detect_time_series_gaps(df_with_gap)
        assert len(gaps) == 1
        assert "2023-03" in gaps["missing_date"].iloc[0]
