"""Supplementary outputs (run after src.run_experiments).

1. test_period_grid.png: all 8 series, test period only, actual vs 3 models;
   mae_by_series.png: grouped test-MAE bars (both re-plotted from saved tables).
2. supplementary_persistence.csv: last-value persistence (y[t-1]) on the same
   test months. NOT one of the three pre-specified models; a diagnostic to show
   whether LSTM gains reflect level adaptation rather than learned seasonality.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import COUNTRIES, FIGURES_DIR, KEYWORDS, PROCESSED_DATA_DIR, PROCESSED_DATASET_FILENAME, TABLES_DIR
from src.evaluation import ANOMALY_MONTH, calculate_mae, calculate_rmse

LABELS = {"seasonal_naive": ("Seasonal naive", "#2ca02c"), "standard_lstm": ("Standard LSTM", "#1f77b4"),
          "seasonal_lstm": ("Seasonality-aware LSTM", "#d62728")}


def _fmt_axis(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 6, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(alpha=0.3)


def main():
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    pred = pd.read_csv(TABLES_DIR / "predictions.csv", parse_dates=["date"])
    raw = pd.read_csv(PROCESSED_DATA_DIR / PROCESSED_DATASET_FILENAME, parse_dates=["date"])
    fig, axes = plt.subplots(4, 2, figsize=(8.6, 8.4), sharex=True)
    rows = []
    anomaly = pd.Timestamp(ANOMALY_MONTH)
    for i, kw in enumerate(KEYWORDS):
        for j, co in enumerate(COUNTRIES.values()):
            ax, g = axes[i, j], pred[(pred.keyword == kw) & (pred.country == co)]
            ax.axvline(anomaly, color="#999999", ls=":", lw=1.2)
            ax.plot(g.date, g.actual, "k-o", ms=3, lw=1.8, label="Actual")
            for k, (lab, col) in LABELS.items():
                ax.plot(g.date, g[k], "--", color=col, lw=1.6, label=lab)
            ax.set_title(f"{kw.capitalize()} - {co}", fontsize=10, fontweight="bold")
            _fmt_axis(ax)
            s = raw[(raw.keyword == kw) & (raw.country == co)].sort_values("date").set_index("date").search_interest
            persist = s.shift(1).reindex(g.date).values
            for tag, m in [("full", g.date == g.date), ("excl_2026-02", g.date != ANOMALY_MONTH)]:
                rows.append({"keyword": kw, "country": co, "test_scope": tag,
                             "persistence_MAE": calculate_mae(g.actual.values[m], persist[m]),
                             "persistence_RMSE": calculate_rmse(g.actual.values[m], persist[m])})
    for ax in axes[:, 0]:
        ax.set_ylabel("Search interest", fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.0, w_pad=1.2)
    fig.savefig(FIGURES_DIR / "test_period_grid.png", dpi=220)
    plt.close(fig)

    # grouped MAE bars from the existing results table
    comp = pd.read_csv(TABLES_DIR / "model_comparison.csv")
    comp["series"] = comp["keyword"].str.replace(" clothing", "").str.replace(" fashion", "").str.capitalize() + "\n" + comp["country"]
    order = comp.sort_values(["keyword", "country"])["series"].unique()
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    wdt = 0.26
    for n, (lab, col) in enumerate([("Seasonal Naive", "#2ca02c"), ("Standard LSTM", "#1f77b4"), ("Seasonality-Aware LSTM", "#d62728")]):
        d = comp[comp.model == lab].set_index("series").loc[order]
        ax.bar([x + (n - 1) * wdt for x in range(len(order))], d["MAE"], wdt, color=col, label=lab)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8.5)
    ax.set_ylabel("Test MAE (index points)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mae_by_series.png", dpi=220)
    plt.close(fig)

    out = pd.DataFrame(rows).round(4)
    out.to_csv(TABLES_DIR / "supplementary_persistence.csv", index=False)
    print(out.groupby("test_scope")[["persistence_MAE", "persistence_RMSE"]].mean().round(2))


if __name__ == "__main__":
    main()
