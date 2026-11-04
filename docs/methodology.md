# Methodology

**Data:** 8 Google Trends monthly series (4 keywords x India, France), 2016-01 to 2026-09; see `data_verification.md`.

**Split (per series, chronological 70/15/15, identical dates for every model):**
train 2016-01..2023-06 (90 months), validation 2023-07..2025-01 (19), test 2025-02..2026-09 (20).

**Leakage controls:** MinMax scaler fit on the training segment only; input windows (12 months) look strictly
backward; validation (early stopping) precedes test; no test information is used for training or model selection.
Walk-forward test protocol: at each test month all models forecast one step ahead using the *actual* history up to
the previous month (actuals are revealed after each step). No multi-step forecasts are evaluated.

**Models**
1. *Seasonal naive:* y_hat[t] = y[t-12] (one-step walk-forward, so it sees the same information as the LSTMs).
   The original multi-step version in the repo (kept as `seasonal_naive_forecast`) was replaced for evaluation because
   it forecast the whole test period from one fixed point while the LSTMs were one-step, an unfair comparison.
2. *Standard LSTM:* input = 12 lagged scaled values; LSTM(50) -> Dense(1); Adam, MSE, batch 16, up to 100 epochs,
   early stopping on validation loss (patience 10, best weights restored).
3. *Seasonality-aware LSTM:* identical architecture; each input step additionally carries sin(2*pi*m/12), cos(2*pi*m/12)
   of the **month being predicted** (the calendar is known in advance), giving 3 features per step.
   (The original repo code encoded each step's own month; the target-month encoding is a more direct seasonal signal.)

**Seeds:** each LSTM is trained with 5 seeds (42-46). Reported predictions are the mean of the 5 runs (ensemble);
per-seed MAE/RMSE mean and SD are in `results/tables/seed_variability.csv`. No hyperparameter tuning was done.

**Metrics:** MAE and RMSE on the 20 test months, per keyword-country. Sensitivity: same metrics excluding 2026-02.
Supplementary diagnostic (not a pre-specified model): last-value persistence y[t-1].

**Deliberately not done:** no significance tests (8 series, 20 test months, one split - too little for reliable inference);
no tuning; no pooled/global models; no rolling-origin evaluation.
