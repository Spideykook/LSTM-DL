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


# ── Fair walk-forward baseline ───────────────────────────────────────

def seasonal_naive_walkforward(
    history: np.ndarray,
    test: np.ndarray,
    seasonal_period: int = SEASONAL_PERIOD,
) -> np.ndarray:
    """One-step-ahead seasonal naive: forecast for t is the actual value at t-12.

    Matches the LSTM protocol (walk-forward, actuals revealed after each step),
    so all three models face the same information set at every test month.
    """
    full = np.concatenate([history, test])
    start = len(history)
    return np.array([full[start + i - seasonal_period] for i in range(len(test))])


# ── Full pipeline per series ─────────────────────────────────────────

def run_single_series(
    df: pd.DataFrame,
    keyword: str,
    country: str,
    lookback: int = LOOKBACK,
    epochs: int = 100,
    seed: int = 42,
    n_seeds: int = 5,
) -> Optional[dict]:
    """Run all three models on one keyword-country series.

    Protocol: chronological 70/15/15 split; MinMax scaler fit on the training
    segment only; windows look strictly backward; test predictions are
    one-step-ahead walk-forward (actuals revealed after each step) for all
    three models. LSTMs are trained with `n_seeds` seeds; the reported
    prediction is the mean across seeds and per-seed MAE is also stored.
    The seasonality-aware model receives sin/cos of the *target* month
    (calendar is known in advance) in addition to the lagged series value.
    """
    import tensorflow as tf
    from src.evaluation import calculate_mae, calculate_rmse

    series = prepare_series(df, keyword, country)
    if series is None or len(series) < lookback + 24:
        logger.warning(f"Insufficient data for '{keyword}' in {country}")
        return None

    train_df, val_df, test_df = chronological_split(series)
    n_tr, n_va, n_te = len(train_df), len(val_df), len(test_df)
    if n_te < 2 or n_va < 2:
        return None
    y_all = series["search_interest"].values.astype(float)

    # Scaler fit on TRAIN ONLY
    scaler = MinMaxScaler().fit(y_all[:n_tr].reshape(-1, 1))
    scaled = scaler.transform(y_all.reshape(-1, 1))

    # Seasonal features of the month being predicted at the next step
    next_month = series["date"] + pd.offsets.MonthBegin(1)
    cal = add_seasonal_features(next_month).values
    feats = {"standard": scaled, "seasonal": np.hstack([scaled, cal])}

    # Baseline (walk-forward, same protocol as LSTMs)
    test_actual = y_all[n_tr + n_va:]
    naive = seasonal_naive_walkforward(y_all[:n_tr + n_va], test_actual)

    preds, seed_mae, seed_rmse = {}, {}, {}
    for name, arr in feats.items():
        X, y = create_sequences(arr, lookback)
        tgt = np.arange(lookback, len(arr))  # target index of each window
        m_tr, m_va, m_te = tgt < n_tr, (tgt >= n_tr) & (tgt < n_tr + n_va), tgt >= n_tr + n_va
        per_seed = []
        seed_mae[name], seed_rmse[name] = [], []
        for k in range(n_seeds):
            tf.keras.utils.set_random_seed(seed + k)
            model, _ = train_lstm(
                X[m_tr], y[m_tr], X[m_va], y[m_va],
                input_shape=(lookback, arr.shape[1]), epochs=epochs, verbose=0,
            )
            p = scaler.inverse_transform(
                model.predict(X[m_te], verbose=0).reshape(-1, 1)).flatten()
            per_seed.append(p)
            seed_mae[name].append(calculate_mae(test_actual, p))
            seed_rmse[name].append(calculate_rmse(test_actual, p))
            tf.keras.backend.clear_session()
        preds[name] = np.mean(per_seed, axis=0)

    def pack(p, key=None):
        d = {"predictions": p, "mae": calculate_mae(test_actual, p),
             "rmse": calculate_rmse(test_actual, p)}
        if key:
            d["seed_mae"], d["seed_rmse"] = seed_mae[key], seed_rmse[key]
        return d

    results = {
        "keyword": keyword, "country": country,
        "n_train": n_tr, "n_val": n_va, "n_test": n_te, "n_seeds": n_seeds,
        "test_dates": test_df["date"].values, "test_actual": test_actual,
        "seasonal_naive": pack(naive),
        "standard_lstm": pack(preds["standard"], "standard"),
        "seasonal_lstm": pack(preds["seasonal"], "seasonal"),
        "train_dates": train_df["date"].values, "train_actual": train_df["search_interest"].values,
        "val_dates": val_df["date"].values, "val_actual": val_df["search_interest"].values,
    }
    logger.info(f"[{keyword}/{country}] MAE naive={results['seasonal_naive']['mae']:.2f} "
                f"std={results['standard_lstm']['mae']:.2f} sea={results['seasonal_lstm']['mae']:.2f}")
    return results
