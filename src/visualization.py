"""Visualization module for fashion trend forecasting.

Provides reusable plotting functions for exploratory analysis of
Google Trends search-interest data.

All charts clearly label that values represent normalized Google Trends
interest scores (0-100), not absolute search volumes.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from src.config import (
    COLOR_PALETTE,
    COUNTRIES,
    FIGURE_DPI,
    FIGURE_SIZE,
    FIGURES_DIR,
    KEYWORD_STYLES,
    ensure_directories,
)

logger = logging.getLogger(__name__)


def plot_keyword_country(
    df: pd.DataFrame,
    keyword: str,
    country: str,
    save_dir: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot time series of search interest for a single keyword and country.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame with columns: date, keyword, country, search_interest.
    keyword : str
        The keyword to plot.
    country : str
        The country name to plot.
    save_dir : Path, optional
        Directory to save the figure. Defaults to FIGURES_DIR.
    show : bool
        Whether to display the plot (set False for batch processing).

    Returns
    -------
    Path or None
        Path to the saved figure, or None if data is unavailable.
    """
    if save_dir is None:
        save_dir = FIGURES_DIR
    ensure_directories()

    subset = df[(df["keyword"] == keyword) & (df["country"] == country)].copy()
    if subset.empty:
        logger.warning(f"No data for '{keyword}' in {country}. Skipping plot.")
        return None

    subset["date"] = pd.to_datetime(subset["date"])
    subset = subset.sort_values("date")

    color = COLOR_PALETTE.get(country, "#333333")
    style = KEYWORD_STYLES.get(keyword, {"linestyle": "-", "marker": "o"})

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(
        subset["date"],
        subset["search_interest"],
        color=color,
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=3,
        linewidth=1.2,
        label=f"{keyword} ({country})",
    )

    ax.set_title(f"Search Interest: '{keyword}' in {country}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Normalized Search Interest (0-100)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(-2, 105)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Add note about normalization
    ax.text(
        0.99, 0.01,
        "Values: Google Trends normalized interest (not absolute volume)",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        color="gray",
        style="italic",
    )

    plt.tight_layout()

    safe_kw = keyword.replace(" ", "_").lower()
    safe_co = country.replace(" ", "_").lower()
    filename = f"timeseries_{safe_kw}_{safe_co}.png"
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info(f"Saved figure: {filepath}")

    if show:
        plt.show()
    plt.close(fig)

    return filepath


def plot_country_comparison(
    df: pd.DataFrame,
    keyword: str,
    save_dir: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot search interest for a keyword comparing India and France.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame.
    keyword : str
        The keyword to compare across countries.
    save_dir : Path, optional
        Directory to save the figure.
    show : bool
        Whether to display the plot.

    Returns
    -------
    Path or None
        Path to the saved figure.
    """
    if save_dir is None:
        save_dir = FIGURES_DIR
    ensure_directories()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    has_data = False

    for country_code, country_name in COUNTRIES.items():
        subset = df[(df["keyword"] == keyword) & (df["country"] == country_name)].copy()
        if subset.empty:
            logger.warning(f"No data for '{keyword}' in {country_name}. Skipping.")
            continue

        has_data = True
        subset["date"] = pd.to_datetime(subset["date"])
        subset = subset.sort_values("date")
        color = COLOR_PALETTE.get(country_name, "#333333")

        ax.plot(
            subset["date"],
            subset["search_interest"],
            color=color,
            linewidth=1.5,
            label=country_name,
            alpha=0.85,
        )

    if not has_data:
        logger.warning(f"No data available for '{keyword}' comparison. Skipping.")
        plt.close(fig)
        return None

    ax.set_title(
        f"Search Interest Comparison: '{keyword}'\nIndia vs France",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Normalized Search Interest (0-100)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(-2, 105)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    ax.text(
        0.99, 0.01,
        (
            "CAUTION: Each country's series is independently normalized by Google.\n"
            "A higher score does NOT imply higher absolute search volume."
        ),
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        color="gray",
        style="italic",
    )

    plt.tight_layout()

    safe_kw = keyword.replace(" ", "_").lower()
    filename = f"comparison_{safe_kw}_india_vs_france.png"
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info(f"Saved comparison figure: {filepath}")

    if show:
        plt.show()
    plt.close(fig)

    return filepath


def plot_all_keywords_for_country(
    df: pd.DataFrame,
    country: str,
    save_dir: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot all keywords for a single country on one chart.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame.
    country : str
        Country name (e.g., 'India', 'France').
    save_dir : Path, optional
        Directory to save the figure.
    show : bool
        Whether to display the plot.

    Returns
    -------
    Path or None
        Path to the saved figure.
    """
    if save_dir is None:
        save_dir = FIGURES_DIR
    ensure_directories()

    country_data = df[df["country"] == country].copy()
    if country_data.empty:
        logger.warning(f"No data for {country}. Skipping.")
        return None

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    keywords_found = country_data["keyword"].unique()
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(keywords_found))

    for i, keyword in enumerate(sorted(keywords_found)):
        subset = country_data[country_data["keyword"] == keyword].copy()
        subset["date"] = pd.to_datetime(subset["date"])
        subset = subset.sort_values("date")

        style = KEYWORD_STYLES.get(keyword, {"linestyle": "-", "marker": "o"})

        ax.plot(
            subset["date"],
            subset["search_interest"],
            color=cmap(i),
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=2,
            linewidth=1.2,
            label=keyword,
            alpha=0.85,
        )

    ax.set_title(
        f"Fashion Search Interest Trends: {country}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Normalized Search Interest (0-100)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(-2, 105)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    ax.text(
        0.99, 0.01,
        "Values: Google Trends normalized interest (not absolute volume)",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        color="gray",
        style="italic",
    )

    plt.tight_layout()

    safe_co = country.replace(" ", "_").lower()
    filename = f"all_keywords_{safe_co}.png"
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info(f"Saved multi-keyword figure: {filepath}")

    if show:
        plt.show()
    plt.close(fig)

    return filepath


def generate_data_availability_summary(
    df: pd.DataFrame,
    save_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Generate a heatmap-style summary of data availability.

    Shows observation count per keyword-country combination.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame.
    save_dir : Path, optional
        Directory to save the figure.

    Returns
    -------
    Path or None
        Path to the saved figure.
    """
    if save_dir is None:
        save_dir = FIGURES_DIR
    ensure_directories()

    if df.empty:
        logger.warning("No data for availability summary.")
        return None

    pivot = df.groupby(["keyword", "country"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.8)))

    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_title("Data Availability: Observations per Keyword-Country", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Number of observations")
    plt.tight_layout()

    filepath = save_dir / "data_availability_summary.png"
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info(f"Saved availability summary: {filepath}")
    plt.close(fig)

    return filepath


def generate_all_plots(
    df: pd.DataFrame,
    save_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate all standard exploratory plots.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame.
    save_dir : Path, optional
        Directory to save all figures.

    Returns
    -------
    list of Path
        Paths to all generated figures.
    """
    if save_dir is None:
        save_dir = FIGURES_DIR

    ensure_directories()
    generated = []

    if df.empty:
        logger.warning("No data available. Cannot generate plots.")
        return generated

    keywords = df["keyword"].unique()
    countries = df["country"].unique()

    # 1. Individual keyword-country plots
    for keyword in keywords:
        for country in countries:
            path = plot_keyword_country(df, keyword, country, save_dir)
            if path:
                generated.append(path)

    # 2. Country comparison plots
    for keyword in keywords:
        path = plot_country_comparison(df, keyword, save_dir)
        if path:
            generated.append(path)

    # 3. All keywords per country
    for country in countries:
        path = plot_all_keywords_for_country(df, country, save_dir)
        if path:
            generated.append(path)

    # 4. Data availability summary
    path = generate_data_availability_summary(df, save_dir)
    if path:
        generated.append(path)

    logger.info(f"Generated {len(generated)} figures total.")
    return generated


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR, PROCESSED_DATASET_FILENAME

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    processed_path = PROCESSED_DATA_DIR / PROCESSED_DATASET_FILENAME
    if not processed_path.exists():
        print(f"Processed dataset not found at {processed_path}")
        print("Run preprocessing first: python -m src.preprocessing")
    else:
        df = pd.read_csv(processed_path)
        print(f"Loaded {len(df)} observations from {processed_path}")
        paths = generate_all_plots(df)
        print(f"\nGenerated {len(paths)} figures:")
        for p in paths:
            print(f"  {p}")
