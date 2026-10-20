"""Central configuration for the fashion trend forecasting project.

All configurable parameters are defined here to ensure consistency
across data collection, preprocessing, and analysis.
"""

from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"

# ── Google Trends configuration ──────────────────────────────────────
KEYWORDS = [
    "streetwear",
    "oversized clothing",
    "sustainable fashion",
    "traditional clothing",
]

# Countries to compare (ISO 3166-1 alpha-2 codes used by Google Trends)
COUNTRIES = {
    "IN": "India",
    "FR": "France",
}

# Data collection time range
START_DATE = "2016-01-01"
END_DATE = "2026-09-30"
TIMEFRAME = f"{START_DATE} {END_DATE}"

GRANULARITY = "monthly"

# pytrends request parameters
PYTRENDS_HOST_LANGUAGE = "en-US"
PYTRENDS_TIMEZONE = 330  # IST offset in minutes
PYTRENDS_RETRIES = 3
PYTRENDS_BACKOFF_FACTOR = 2.0
PYTRENDS_REQUEST_DELAY = 15  # seconds between requests

# ── Data schema ──────────────────────────────────────────────────────
PROCESSED_COLUMNS = ["date", "keyword", "country", "search_interest"]
DATE_FORMAT = "%Y-%m-%d"

# ── Metadata keys ────────────────────────────────────────────────────
METADATA_FIELDS = [
    "collection_date",
    "collection_method",
    "country_code",
    "country_name",
    "keyword",
    "date_range_start",
    "date_range_end",
    "normalization_note",
    "limitations",
]

# ── Visualization defaults ───────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_SIZE = (12, 5)
COLOR_PALETTE = {
    "India": "#FF6B35",
    "France": "#004E89",
}
KEYWORD_STYLES = {
    "streetwear": {"linestyle": "-", "marker": "o"},
    "oversized clothing": {"linestyle": "--", "marker": "s"},
    "sustainable fashion": {"linestyle": "-.", "marker": "^"},
    "traditional clothing": {"linestyle": ":", "marker": "D"},
}

# ── File naming conventions ──────────────────────────────────────────
def raw_csv_filename(keyword: str, country_code: str) -> str:
    """Generate a consistent filename for raw Google Trends CSV data."""
    safe_keyword = keyword.replace(" ", "_").lower()
    return f"gtrends_{safe_keyword}_{country_code.lower()}.csv"


def metadata_filename(keyword: str, country_code: str) -> str:
    """Generate a consistent filename for collection metadata JSON."""
    safe_keyword = keyword.replace(" ", "_").lower()
    return f"metadata_{safe_keyword}_{country_code.lower()}.json"


PROCESSED_DATASET_FILENAME = "fashion_trends_processed.csv"
DATA_QUALITY_REPORT_FILENAME = "data_quality_report.txt"


def ensure_directories():
    """Create all required project directories if they do not exist."""
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR, DOCS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
