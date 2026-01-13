"""CBP Currency Seizure Data loader.

Loads currency seizure statistics from U.S. Customs and Border Protection.
Used to test the cash substitution hypothesis - whether crypto is replacing
cash as the settlement layer for criminal enterprise.

Source: https://www.cbp.gov/newsroom/stats/currency-other-monetary-instrument-seizures
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from btc_analysis.config import get_config

logger = logging.getLogger(__name__)


def fetch_currency_seizures(
    output_path: Optional[Path] = None,
    start_year: int = 2017,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Load currency seizure data from CBP CSV files.

    CBP publishes monthly currency seizure statistics.
    Data must be manually downloaded from CBP website and placed in
    data/raw/cbp_currency/ directory.

    Args:
        output_path: Path to save the processed CSV output.
        start_year: Start year for the data range.
        end_year: End year for the data range.

    Returns:
        DataFrame with monthly currency seizure totals.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "cbp_currency_seizures.csv"

    # Check for CBP currency subdirectory with CSV files
    cbp_dir = config.paths.raw_dir / "cbp_currency"
    if cbp_dir.exists():
        cbp_files = list(cbp_dir.glob("*.csv"))
        if cbp_files:
            logger.info(f"Found {len(cbp_files)} CBP currency CSV files in {cbp_dir}")
            df = _load_cbp_currency_files(cbp_files)
            if df is not None and not df.empty:
                df = _filter_by_year(df, start_year, end_year)
                df.to_csv(output_path, index=False)
                logger.info(f"Saved processed CBP currency data to {output_path}")
                return df

    # Check for existing output file
    if output_path.exists():
        logger.info(f"Loading cached currency seizure data from {output_path}")
        return pd.read_csv(output_path, parse_dates=["date"])

    # If no data available, return empty DataFrame
    logger.warning(
        "Could not find CBP currency seizure data. "
        "Please download from: https://www.cbp.gov/newsroom/stats/currency-other-monetary-instrument-seizures "
        "and save CSV files to data/raw/cbp_currency/ directory."
    )

    return _create_empty_df()


def _load_cbp_currency_files(files: list) -> Optional[pd.DataFrame]:
    """
    Load and combine multiple CBP currency CSV files.

    Args:
        files: List of Path objects to CBP CSV files.

    Returns:
        Combined and processed DataFrame.
    """
    all_data = []

    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath.name}")
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Error loading {filepath}: {e}")

    if not all_data:
        return None

    # Combine all files
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined {len(combined)} total records from {len(all_data)} files")

    # Process the combined data
    return _process_cbp_currency_data(combined)


def _process_cbp_currency_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw CBP currency data into monthly national totals.

    Args:
        df: Raw DataFrame from CBP CSV files.

    Returns:
        Processed DataFrame with monthly national totals.
    """
    logger.info(f"Processing CBP currency data with columns: {df.columns.tolist()}")

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Expected columns: fiscal_year, month_(abbv), ..., count_of_seizure_events, currency_seizures_amount_(usd)
    fy_col = next((c for c in df.columns if "fiscal" in c or c == "fy"), None)
    month_col = next((c for c in df.columns if "month" in c), None)
    count_col = next((c for c in df.columns if "count" in c and "seizure" in c), None)
    amount_col = next((c for c in df.columns if "amount" in c or "usd" in c), None)

    if not all([fy_col, month_col, count_col, amount_col]):
        logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
        return _create_empty_df()

    # Convert amounts to numeric
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")

    # Map month abbreviations to numbers
    month_abbr_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    df["month_num"] = df[month_col].str.upper().map(month_abbr_map)

    # Clean FY column - extract just the year number
    df[fy_col] = df[fy_col].astype(str).str.extract(r"(\d{4})")[0]
    df[fy_col] = pd.to_numeric(df[fy_col], errors="coerce")

    # Drop rows with invalid dates
    df = df.dropna(subset=[fy_col, "month_num"])

    # Convert fiscal year + month to calendar year + month
    # Federal FY starts October 1: FY2024 Oct = Calendar Oct 2023
    def fy_to_calendar(row):
        fy = int(row[fy_col])
        month = int(row["month_num"])
        # Oct-Dec of FY are in prior calendar year
        if month >= 10:
            return fy - 1, month
        else:
            return fy, month

    cal_dates = df.apply(fy_to_calendar, axis=1, result_type="expand")
    df["year"] = cal_dates[0].astype(int)
    df["month"] = cal_dates[1].astype(int)

    # Aggregate to national monthly totals
    monthly = df.groupby(["year", "month"]).agg({
        count_col: "sum",
        amount_col: "sum",
    }).reset_index()

    monthly.columns = ["year", "month", "currency_seizure_count", "currency_seizure_usd"]

    # Create date column
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" +
        monthly["month"].astype(str).str.zfill(2) + "-01"
    )

    # Sort by date and remove duplicates
    monthly = monthly.sort_values("date").reset_index(drop=True)
    monthly = monthly.drop_duplicates(subset=["year", "month"], keep="last")

    logger.info(f"Processed CBP currency data: {len(monthly)} months")

    return monthly


def _filter_by_year(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Filter DataFrame by year range."""
    if "year" in df.columns:
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    return df


def _create_empty_df() -> pd.DataFrame:
    """Create empty DataFrame with expected schema."""
    return pd.DataFrame(columns=[
        "date", "year", "month",
        "currency_seizure_count", "currency_seizure_usd",
    ])
