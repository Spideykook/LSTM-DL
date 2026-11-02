"""Forecasting models for fashion trend prediction.

Implements three approaches:
1. Seasonal Naive Baseline - uses same-month-last-year as forecast
2. Standard LSTM - vanilla LSTM on raw search interest sequences
3. Seasonality-Aware LSTM - LSTM with month sine/cosine features

All models operate on individual (keyword, country) time series.
Scalers are fit on training data only to prevent data leakage.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Window size for LSTM input sequences (months of history)
LOOKBACK = 12
# Seasonal period for naive baseline (months)
SEASONAL_PERIOD = 12


# ── Seasonal Naive Baseline ─────────────────────────────────────────

def seasonal_naive_forecast(
    train: np.ndarray,
    n_forecast: int,
    seasonal_period: int = SEASONAL_PERIOD,
) -> np.ndarray:
    """Produce seasonal naive forecasts.

    For each forecast step, predicts the value from the same month
    in the most recent complete seasonal cycle available in training data.

    Parameters
    ----------
    train : np.ndarray
        Training series values (1-D).
    n_forecast : int
        Number of steps to forecast.
    seasonal_period : int
        Length of one seasonal cycle (default 12 for monthly).

    Returns
    -------
    np.ndarray
        Array of length n_forecast with predicted values.
    """
    forecasts = []
    for i in range(n_forecast):
        # Look back exactly one seasonal period from the end of training
        idx = len(train) - seasonal_period + (i % seasonal_period)
        if idx >= 0:
            forecasts.append(train[idx])
        else:
            # Fallback: use the mean if not enough history
            forecasts.append(float(np.mean(train)))
    return np.array(forecasts)


# ── Data preparation utilities ───────────────────────────────────────

def create_sequences(
    data: np.ndarray,
    lookback: int = LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """Create input/output sequences for LSTM training.

    Parameters
    ----------
    data : np.ndarray
        Scaled time series values, shape (n_samples, n_features).
    lookback : int
        Number of past time steps to use as input.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        X with shape (n_sequences, lookback, n_features),
        y with shape (n_sequences,).
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])  # predict the search_interest column
    return np.array(X), np.array(y)


def add_seasonal_features(dates: pd.Series) -> pd.DataFrame:
    """Create sine/cosine seasonal features from dates.

    Parameters
    ----------
    dates : pd.Series
        Series of datetime values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: month_sin, month_cos.
    """
    dates = pd.to_datetime(dates)
    months = dates.dt.month
    return pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12),
    })


def prepare_series(
    df: pd.DataFrame,
    keyword: str,
    country: str,
) -> Optional[pd.DataFrame]:
    """Extract and sort a single keyword-country time series.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset with date, keyword, country, search_interest columns.
    keyword : str
        Keyword to filter.
    country : str
        Country name to filter.

    Returns
    -------
    pd.DataFrame or None
        Sorted DataFrame with datetime index, or None if no data.
    """
    subset = df[(df["keyword"] == keyword) & (df["country"] == country)].copy()
    if subset.empty:
        logger.warning(f"No data for '{keyword}' in {country}")
        return None
    subset["date"] = pd.to_datetime(subset["date"])
    subset = subset.sort_values("date").reset_index(drop=True)
    # Drop rows with missing search_interest
    subset = subset.dropna(subset=["search_interest"])
    return subset


def chronological_split(
    series: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series chronologically into train/val/test.

    Parameters
    ----------
    series : pd.DataFrame
        Sorted time series.
    train_ratio : float
        Fraction for training.
    val_ratio : float
        Fraction for validation.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        Train, validation, test DataFrames.
    """
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return series.iloc[:train_end], series.iloc[train_end:val_end], series.iloc[val_end:]


# ── LSTM model builders ─────────────────────────────────────────────

def build_lstm_model(
    input_shape: tuple,
    units: int = 50,
) -> "tf.keras.Model":
    """Build a standard LSTM model.

    Parameters
    ----------
    input_shape : tuple
        Shape of input (lookback, n_features).
    units : int
        Number of LSTM units.

    Returns
    -------
    tf.keras.Model
        Compiled LSTM model.
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, input_shape=input_shape),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    input_shape: tuple,
    epochs: int = 100,
    batch_size: int = 16,
    patience: int = 10,
    verbose: int = 0,
) -> tuple:
    """Train an LSTM model with early stopping.

    Parameters
    ----------
    train_X, train_y : np.ndarray
        Training data.
    val_X, val_y : np.ndarray
        Validation data.
    input_shape : tuple
        Shape for model input.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Batch size.
    patience : int
        Early stopping patience.
    verbose : int
        Verbosity level.

    Returns
    -------
    tuple of (model, history)
        Trained model and training history.
    """
    import tensorflow as tf

    model = build_lstm_model(input_shape)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )

    return model, history


# ── Full pipeline per series ─────────────────────────────────────────

def run_single_series(
    df: pd.DataFrame,
    keyword: str,
    country: str,
    lookback: int = LOOKBACK,
    epochs: int = 100,
    seed: int = 42,
) -> Optional[dict]:
    """Run all three models on a single keyword-country series.

    Parameters
    ----------
    df : pd.DataFrame
        Full processed dataset.
    keyword : str
        Target keyword.
    country : str
        Target country.
    lookback : int
        LSTM lookback window.
    epochs : int
        Max training epochs.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict or None
        Results dictionary with predictions and metrics for all models.
    """
    import tensorflow as tf

    # Set seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    series = prepare_series(df, keyword, country)
    if series is None or len(series) < lookback + 24:
        logger.warning(
            f"Insufficient data for '{keyword}' in {country} "
            f"(need at least {lookback + 24} observations, have {len(series) if series is not None else 0})"
        )
        return None

    # Chronological split
    train_df, val_df, test_df = chronological_split(series)
    logger.info(
        f"[{keyword} / {country}] Split: train={len(train_df)}, "
        f"val={len(val_df)}, test={len(test_df)}"
    )

    if len(test_df) < 2 or len(val_df) < 2:
        logger.warning(f"Not enough data for validation/testing for '{keyword}' in {country}")
        return None

    # Store dates and actuals for test period
    test_dates = test_df["date"].values
    test_actual = test_df["search_interest"].values

    # ─── Model 1: Seasonal Naive ─────────────────────────────────
    train_values = train_df["search_interest"].values
    naive_preds = seasonal_naive_forecast(
        np.concatenate([train_values, val_df["search_interest"].values]),
        n_forecast=len(test_df),
    )

    # ─── Model 2: Standard LSTM ──────────────────────────────────
    # Fit scaler on training data only
    scaler_std = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler_std.fit_transform(
        train_df[["search_interest"]].values
    )
    val_scaled = scaler_std.transform(
        val_df[["search_interest"]].values
    )
    # For test, we need the full history up to test start
    full_before_test = pd.concat([train_df, val_df])
    full_scaled = scaler_std.transform(
        full_before_test[["search_interest"]].values
    )
    test_scaled = scaler_std.transform(
        test_df[["search_interest"]].values
    )

    # Create sequences for standard LSTM (1 feature: search_interest)
    all_train_val_scaled = np.vstack([train_scaled, val_scaled])

    trn_X, trn_y = create_sequences(train_scaled, lookback)
    # For validation sequences, we need context from end of training
    val_combined = np.vstack([train_scaled[-lookback:], val_scaled])
    val_X, val_y = create_sequences(val_combined, lookback)

    if len(trn_X) < 2 or len(val_X) < 1:
        logger.warning(f"Not enough sequences for LSTM training: '{keyword}' in {country}")
        return None

    # Train standard LSTM
    std_model, std_hist = train_lstm(
        trn_X, trn_y, val_X, val_y,
        input_shape=(lookback, 1),
        epochs=epochs,
        verbose=0,
    )

    # Generate test predictions using rolling forecast
    std_preds_scaled = _rolling_forecast(
        std_model, all_train_val_scaled, test_scaled, lookback, n_features=1
    )
    std_preds = scaler_std.inverse_transform(
        std_preds_scaled.reshape(-1, 1)
    ).flatten()

    # ─── Model 3: Seasonality-Aware LSTM ─────────────────────────
    # Add seasonal features
    train_seasonal = add_seasonal_features(train_df["date"])
    val_seasonal = add_seasonal_features(val_df["date"])
    test_seasonal = add_seasonal_features(test_df["date"])
    full_before_seasonal = add_seasonal_features(full_before_test["date"])

    # Scale seasonal features using training data
    scaler_sea = MinMaxScaler(feature_range=(0, 1))
    train_si_scaled = scaler_sea.fit_transform(
        train_df[["search_interest"]].values
    )

    # Build feature arrays: [search_interest_scaled, month_sin, month_cos]
    train_feat = np.hstack([
        train_si_scaled,
        train_seasonal[["month_sin", "month_cos"]].values,
    ])
    val_si_scaled = scaler_sea.transform(val_df[["search_interest"]].values)
    val_feat = np.hstack([
        val_si_scaled,
        val_seasonal[["month_sin", "month_cos"]].values,
    ])
    test_si_scaled_sea = scaler_sea.transform(test_df[["search_interest"]].values)
    test_feat = np.hstack([
        test_si_scaled_sea,
        test_seasonal[["month_sin", "month_cos"]].values,
    ])

    all_train_val_feat = np.vstack([train_feat, val_feat])

    sea_trn_X, sea_trn_y = create_sequences(train_feat, lookback)
    val_feat_combined = np.vstack([train_feat[-lookback:], val_feat])
    sea_val_X, sea_val_y = create_sequences(val_feat_combined, lookback)

    if len(sea_trn_X) < 2 or len(sea_val_X) < 1:
        logger.warning(f"Not enough seasonal sequences for '{keyword}' in {country}")
        return None

    # Train seasonality-aware LSTM (3 features)
    sea_model, sea_hist = train_lstm(
        sea_trn_X, sea_trn_y, sea_val_X, sea_val_y,
        input_shape=(lookback, 3),
        epochs=epochs,
        verbose=0,
    )

    # Generate test predictions
    sea_preds_scaled = _rolling_forecast(
        sea_model, all_train_val_feat, test_feat, lookback, n_features=3
    )
    sea_preds = scaler_sea.inverse_transform(
        sea_preds_scaled.reshape(-1, 1)
    ).flatten()

    # ─── Calculate metrics ───────────────────────────────────────
    from src.evaluation import calculate_mae, calculate_rmse

    results = {
        "keyword": keyword,
        "country": country,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "test_dates": test_dates,
        "test_actual": test_actual,
        "seasonal_naive": {
            "predictions": naive_preds,
            "mae": calculate_mae(test_actual, naive_preds),
            "rmse": calculate_rmse(test_actual, naive_preds),
        },
        "standard_lstm": {
            "predictions": std_preds,
            "mae": calculate_mae(test_actual, std_preds),
            "rmse": calculate_rmse(test_actual, std_preds),
        },
        "seasonal_lstm": {
            "predictions": sea_preds,
            "mae": calculate_mae(test_actual, sea_preds),
            "rmse": calculate_rmse(test_actual, sea_preds),
        },
        "train_dates": train_df["date"].values,
        "train_actual": train_df["search_interest"].values,
        "val_dates": val_df["date"].values,
        "val_actual": val_df["search_interest"].values,
    }

    logger.info(
        f"[{keyword} / {country}] Results - "
        f"Naive MAE={results['seasonal_naive']['mae']:.2f}, "
        f"Std LSTM MAE={results['standard_lstm']['mae']:.2f}, "
        f"Sea LSTM MAE={results['seasonal_lstm']['mae']:.2f}"
    )

    return results


def _rolling_forecast(
    model,
    history: np.ndarray,
    test_data: np.ndarray,
    lookback: int,
    n_features: int,
) -> np.ndarray:
    """Generate rolling one-step-ahead forecasts on the test set.

    Uses actual values from test set to update the lookback window
    after each prediction (walk-forward validation).

    Parameters
    ----------
    model : tf.keras.Model
        Trained LSTM model.
    history : np.ndarray
        All data before the test period (scaled).
    test_data : np.ndarray
        Test period data (scaled), used to update the window.
    lookback : int
        Number of lookback steps.
    n_features : int
        Number of features per time step.

    Returns
    -------
    np.ndarray
        Predicted values (scaled) for the test period.
    """
    predictions = []
    # Start with the last lookback steps from history
    window = history[-lookback:].copy()

    for i in range(len(test_data)):
        x_input = window.reshape(1, lookback, n_features)
        pred = model.predict(x_input, verbose=0)[0, 0]
        predictions.append(pred)

        # Update window with actual test value (walk-forward)
        if i < len(test_data):
            new_row = test_data[i].copy()
            window = np.vstack([window[1:], new_row.reshape(1, -1)])

    return np.array(predictions)
