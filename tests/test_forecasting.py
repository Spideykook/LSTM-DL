"""Tests for the forecasting and evaluation pipeline.

Uses synthetic data clearly labelled as test-only. NOT research data.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import calculate_mae, calculate_rmse
from src.models import (
    add_seasonal_features,
    chronological_split,
    create_sequences,
    prepare_series,
    seasonal_naive_forecast,
    seasonal_naive_walkforward,
)


# ── Test fixtures ────────────────────────────────────────────────────

@pytest.fixture
def synthetic_series():
    """Create a synthetic monthly time series for testing. NOT research data."""
    n = 60  # 5 years of monthly data
    dates = pd.date_range("2019-01-01", periods=n, freq="MS")
    # Simple seasonal pattern + trend
    values = 50 + 10 * np.sin(2 * np.pi * np.arange(n) / 12) + np.random.RandomState(42).randn(n) * 3
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "keyword": ["test_keyword"] * n,
        "country": ["TestCountry"] * n,
        "search_interest": values.clip(0, 100),
    })


# ── Evaluation metric tests ─────────────────────────────────────────

class TestMetrics:
    def test_mae_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert calculate_mae(actual, actual) == 0.0

    def test_mae_known(self):
        actual = np.array([1.0, 2.0, 3.0])
        pred = np.array([2.0, 3.0, 4.0])
        assert calculate_mae(actual, pred) == 1.0

    def test_rmse_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert calculate_rmse(actual, actual) == 0.0

    def test_rmse_known(self):
        actual = np.array([1.0, 2.0, 3.0])
        pred = np.array([2.0, 3.0, 4.0])
        assert calculate_rmse(actual, pred) == 1.0

    def test_rmse_ge_mae(self):
        actual = np.array([1.0, 5.0, 3.0, 8.0])
        pred = np.array([2.0, 3.0, 5.0, 6.0])
        assert calculate_rmse(actual, pred) >= calculate_mae(actual, pred)


# ── Seasonal naive tests ────────────────────────────────────────────

class TestSeasonalNaive:
    def test_forecast_length(self):
        train = np.arange(24, dtype=float)
        preds = seasonal_naive_forecast(train, n_forecast=6)
        assert len(preds) == 6

    def test_forecast_uses_last_season(self):
        # 24 months of data, forecast should use months 12-23
        train = np.arange(24, dtype=float)
        preds = seasonal_naive_forecast(train, n_forecast=12, seasonal_period=12)
        expected = train[12:24]
        np.testing.assert_array_equal(preds, expected)

    def test_short_train(self):
        train = np.array([1.0, 2.0, 3.0])
        preds = seasonal_naive_forecast(train, n_forecast=2, seasonal_period=12)
        assert len(preds) == 2  # Should not crash


# ── Sequence creation tests ─────────────────────────────────────────

class TestCreateSequences:
    def test_output_shapes(self):
        data = np.random.rand(20, 1)
        X, y = create_sequences(data, lookback=5)
        assert X.shape == (15, 5, 1)
        assert y.shape == (15,)

    def test_multivariate(self):
        data = np.random.rand(20, 3)
        X, y = create_sequences(data, lookback=5)
        assert X.shape == (15, 5, 3)
        assert y.shape == (15,)

    def test_lookback_equals_length(self):
        data = np.random.rand(5, 1)
        X, y = create_sequences(data, lookback=5)
        assert len(X) == 0


# ── Seasonal features tests ─────────────────────────────────────────

class TestSeasonalFeatures:
    def test_output_columns(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=12, freq="MS"))
        result = add_seasonal_features(dates)
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns
        assert len(result) == 12

    def test_periodicity(self):
        dates = pd.Series(pd.date_range("2023-01-01", periods=24, freq="MS"))
        result = add_seasonal_features(dates)
        # January values should repeat every 12 months
        np.testing.assert_almost_equal(
            result["month_sin"].iloc[0],
            result["month_sin"].iloc[12],
        )


# ── Chronological split tests ───────────────────────────────────────

class TestChronologicalSplit:
    def test_split_sizes(self, synthetic_series):
        train, val, test = chronological_split(synthetic_series)
        assert len(train) + len(val) + len(test) == len(synthetic_series)
        assert len(train) == 42  # 70% of 60
        assert len(val) == 9    # 15% of 60 = 9 (int(60*0.85) - int(60*0.7))
        assert len(test) == 9   # remaining

    def test_no_overlap(self, synthetic_series):
        train, val, test = chronological_split(synthetic_series)
        train_dates = set(train["date"])
        val_dates = set(val["date"])
        test_dates = set(test["date"])
        assert train_dates.isdisjoint(val_dates)
        assert train_dates.isdisjoint(test_dates)
        assert val_dates.isdisjoint(test_dates)


# ── Prepare series tests ────────────────────────────────────────────

class TestPrepareSeries:
    def test_basic_extraction(self, synthetic_series):
        result = prepare_series(synthetic_series, "test_keyword", "TestCountry")
        assert result is not None
        assert len(result) == 60

    def test_missing_keyword(self, synthetic_series):
        result = prepare_series(synthetic_series, "nonexistent", "TestCountry")
        assert result is None

    def test_sorted_output(self, synthetic_series):
        # Shuffle and verify sorting
        shuffled = synthetic_series.sample(frac=1, random_state=42)
        result = prepare_series(shuffled, "test_keyword", "TestCountry")
        dates = result["date"].values
        assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1))


class TestWalkForwardNaive:
    def test_uses_actual_lag12(self):
        hist, test = np.arange(24, dtype=float), np.arange(100, 106, dtype=float)
        p = seasonal_naive_walkforward(hist, test)
        np.testing.assert_array_equal(p[:6], hist[12:18])
        p2 = seasonal_naive_walkforward(hist[:12], np.arange(100, 124, dtype=float))
        assert p2[12] == 100.0  # second cycle uses revealed test actuals
