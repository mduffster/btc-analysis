"""CDC Overdose Death Data fetcher.

Fetches monthly provisional drug overdose death counts from the CDC VSRR
(Vital Statistics Rapid Release) system.

Source: https://data.cdc.gov/NCHS/VSRR-Provisional-Drug-Overdose-Death-Counts/xkb8-kh2a
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from btc_analysis.config import get_config

logger = logging.getLogger(__name__)

# CDC Socrata API endpoint for VSRR Provisional Drug Overdose Death Counts
CDC_VSRR_DATASET_ID = "xkb8-kh2a"
CDC_API_BASE = "https://data.cdc.gov/resource"


def fetch_overdose_deaths(
    output_path: Optional[Path] = None,
    start_year: int = 2015,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Fetch monthly provisional drug overdose death counts from CDC VSRR.

    The data includes:
    - Total overdose deaths
    - Deaths by drug category (opioids, synthetic opioids, heroin, cocaine, etc.)
    - State-level breakdowns
    - Provisional vs. predicted counts

    Args:
        output_path: Path to save the CSV output. If None, uses default from config.
        start_year: Start year for the data range.
        end_year: End year for the data range.

    Returns:
        DataFrame with monthly overdose death counts.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "cdc_overdose_deaths.csv"

    # CDC Socrata API endpoint
    api_url = f"{CDC_API_BASE}/{CDC_VSRR_DATASET_ID}.json"

    # Build query - fetch national totals (state = 'US')
    params = {
        "$limit": 50000,
        "$order": "year,month",
        "$where": f"state = 'US' AND year >= '{start_year}' AND year <= '{end_year}'",
    }

    try:
        logger.info(f"Fetching CDC overdose death data from {api_url}...")
        response = requests.get(
            api_url,
            params=params,
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        logger.info(f"Retrieved {len(data)} records from CDC API")

        if not data:
            logger.warning("No data returned from CDC API")
            return _create_empty_df()

        df = pd.DataFrame(data)
        df = _process_overdose_data(df, start_year, end_year)

        if not df.empty:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved overdose death data to {output_path}")

        return df

    except requests.RequestException as e:
        logger.error(f"Failed to fetch CDC overdose data: {e}")

        # Check for local file
        if output_path.exists():
            logger.info(f"Loading cached data from {output_path}")
            return pd.read_csv(output_path)

        return _create_empty_df()


def _process_overdose_data(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Process raw CDC overdose data into a clean time series.

    Args:
        df: Raw DataFrame from CDC API.
        start_year: Filter start year.
        end_year: Filter end year.

    Returns:
        Processed DataFrame with monthly national totals.
    """
    logger.info(f"Available columns: {df.columns.tolist()}")
    logger.info(f"Processing {len(df)} records")

    if df.empty:
        logger.warning("No data to process")
        return _create_empty_df()

    # Convert year to numeric
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Convert month names to numbers if needed
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    if df["month"].dtype == object and df["month"].str.isdigit().sum() == 0:
        df["month"] = df["month"].map(month_map)
    else:
        df["month"] = pd.to_numeric(df["month"], errors="coerce")

    # Filter year range
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    # Get value column - keep as numeric
    value_col = "data_value" if "data_value" in df.columns else None
    if value_col is None:
        logger.error("Could not find death count column in CDC data")
        return _create_empty_df()

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Get indicator column
    indicator_col = "indicator" if "indicator" in df.columns else None

    if indicator_col:
        # Map indicators to standard column names
        indicator_map = {
            "Number of Drug Overdose Deaths": "total_overdose_deaths",
            "Cocaine (T40.5)": "cocaine_deaths",
            "Heroin (T40.1)": "heroin_deaths",
            "Synthetic opioids, excl. methadone (T40.4)": "synthetic_opioid_deaths",
            "Natural & semi-synthetic opioids (T40.2)": "natural_opioid_deaths",
            "Psychostimulants with abuse potential (T43.6)": "stimulant_deaths",
        }

        # Pivot the data - use dropna=False to keep all combinations
        pivot_df = df.pivot_table(
            index=["year", "month"],
            columns=indicator_col,
            values=value_col,
            aggfunc="first",
            dropna=False,
        ).reset_index()

        # Rename columns based on indicator map
        rename_dict = {}
        for orig_name in pivot_df.columns:
            if orig_name in indicator_map:
                rename_dict[orig_name] = indicator_map[orig_name]
            elif orig_name not in ["year", "month"]:
                # Create a clean column name for unmapped indicators
                rename_dict[orig_name] = _clean_indicator_name(orig_name)

        pivot_df = pivot_df.rename(columns=rename_dict)

        # Remove the 'indicator' name from columns index
        pivot_df.columns.name = None

        # Remove duplicate columns by keeping first occurrence
        pivot_df = pivot_df.loc[:, ~pivot_df.columns.duplicated()]

        df = pivot_df
    else:
        # No indicator column - just use raw values
        df = df[["year", "month", value_col]].copy()
        df = df.rename(columns={value_col: "total_overdose_deaths"})
        df = df.drop_duplicates(subset=["year", "month"], keep="first")

    # Create date column
    df["date"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" +
        df["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        f"Processed overdose data: {len(df)} months, "
        f"{df['year'].min()}-{df['year'].max()}"
    )

    return df


def _clean_indicator_name(indicator: str) -> str:
    """Clean indicator name for use as column name."""
    # Common indicator patterns in CDC data
    name = str(indicator).lower()

    if "all drug" in name or "number of drug" in name:
        return "total_overdose_deaths"
    elif "opioid" in name and "synthetic" not in name:
        return "opioid_deaths"
    elif "synthetic" in name or "fentanyl" in name:
        return "synthetic_opioid_deaths"
    elif "heroin" in name:
        return "heroin_deaths"
    elif "cocaine" in name:
        return "cocaine_deaths"
    elif "psychostimulant" in name or "methamphetamine" in name:
        return "stimulant_deaths"
    else:
        # Clean up generic names
        clean = name.replace(" ", "_").replace("-", "_")
        clean = "".join(c for c in clean if c.isalnum() or c == "_")
        return clean[:50]  # Truncate long names


def _create_empty_df() -> pd.DataFrame:
    """Create empty DataFrame with expected schema."""
    return pd.DataFrame(columns=[
        "date", "year", "month",
        "total_overdose_deaths",
        "opioid_deaths",
        "synthetic_opioid_deaths",
        "heroin_deaths",
        "cocaine_deaths",
    ])


def get_national_monthly_overdose(
    start_year: int = 2015,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Get a simple monthly time series of national overdose deaths.

    This is a convenience function that returns just the total deaths
    aggregated at the national level.

    Args:
        start_year: Start year.
        end_year: End year.

    Returns:
        DataFrame with date, year, month, total_overdose_deaths columns.
    """
    df = fetch_overdose_deaths(start_year=start_year, end_year=end_year)

    if df.empty:
        return df

    # Ensure we have the total column
    if "total_overdose_deaths" not in df.columns:
        # Try to find a suitable total column
        death_cols = [c for c in df.columns if "death" in c.lower()]
        if death_cols:
            df["total_overdose_deaths"] = df[death_cols[0]]

    # Return simplified view
    cols = ["date", "year", "month", "total_overdose_deaths"]
    available_cols = [c for c in cols if c in df.columns]

    return df[available_cols].copy()
