"""Time series data merging module.

Merges BTC price, drug overdose deaths, drug seizures, and market controls
into a unified monthly time series for analyzing the relationship between
Bitcoin price and criminal enterprise demand.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from btc_analysis.config import get_config
from btc_analysis.data.btc_price import fetch_btc_price, add_halving_controls
from btc_analysis.data.cdc_overdose import fetch_overdose_deaths
from btc_analysis.data.cbp_seizures import fetch_drug_seizures

logger = logging.getLogger(__name__)


def merge_timeseries(
    output_path: Optional[Path] = None,
    start_year: int = 2015,
    end_year: int = 2024,
    include_market_controls: bool = True,
) -> pd.DataFrame:
    """
    Merge time series datasets for BTC price analysis.

    Combines:
    - BTC price (daily/monthly)
    - CDC overdose deaths (monthly)
    - CBP drug seizures (monthly)
    - Market controls (S&P 500, DXY, interest rates)

    Args:
        output_path: Path to save the merged CSV.
        start_year: Start year for the data.
        end_year: End year for the data.
        include_market_controls: Whether to include market control variables.

    Returns:
        Merged DataFrame with monthly observations.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.processed_dir / "timeseries_panel.csv"

    logger.info(f"Building time series panel from {start_year} to {end_year}")

    # Step 1: Get BTC price data (base series)
    logger.info("Fetching BTC price data...")
    btc = fetch_btc_price(
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
        frequency="daily",
    )

    if btc.empty:
        logger.error("Failed to fetch BTC price data")
        return pd.DataFrame()

    # Aggregate to monthly
    btc_monthly = _aggregate_btc_monthly(btc)
    logger.info(f"BTC price data: {len(btc_monthly)} months")

    # Step 2: Get overdose death data
    logger.info("Fetching CDC overdose death data...")
    overdose = fetch_overdose_deaths(start_year=start_year, end_year=end_year)

    if not overdose.empty:
        overdose_monthly = _standardize_monthly(overdose, "overdose")
        logger.info(f"Overdose death data: {len(overdose_monthly)} months")
    else:
        logger.warning("No overdose data available")
        overdose_monthly = pd.DataFrame()

    # Step 3: Get drug seizure data
    logger.info("Fetching CBP drug seizure data...")
    seizures = fetch_drug_seizures(start_year=start_year, end_year=end_year)

    if not seizures.empty:
        seizures_monthly = _standardize_monthly(seizures, "seizures")
        logger.info(f"Seizure data: {len(seizures_monthly)} months")
    else:
        logger.warning("No seizure data available")
        seizures_monthly = pd.DataFrame()

    # Step 4: Merge all datasets on year-month
    merged = btc_monthly.copy()

    if not overdose_monthly.empty:
        merged = pd.merge(
            merged,
            overdose_monthly,
            on=["year", "month"],
            how="left",
        )

    if not seizures_monthly.empty:
        merged = pd.merge(
            merged,
            seizures_monthly,
            on=["year", "month"],
            how="left",
        )

    # Step 5: Add market controls if requested
    if include_market_controls:
        logger.info("Adding market control variables...")
        merged = _add_market_controls(merged)

    # Step 6: Add halving cycle controls
    merged = add_halving_controls(merged, date_col="date")

    # Step 7: Create drug market index
    merged = _create_drug_market_index(merged)

    # Step 8: Create lagged variables for time series analysis
    merged = _add_lagged_variables(merged)

    # Step 9: Create percentage change variables
    merged = _add_pct_changes(merged)

    # Save output
    merged.to_csv(output_path, index=False)
    logger.info(f"Saved merged time series to {output_path}")

    # Log summary
    _log_merge_summary(merged)

    return merged


def _aggregate_btc_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily BTC data to monthly frequency.

    Args:
        df: Daily BTC price DataFrame.

    Returns:
        Monthly aggregated DataFrame.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Group by year-month
    agg_dict = {
        "price": ["mean", "last", "first", "max", "min", "std"],
    }

    if "volume" in df.columns:
        agg_dict["volume"] = "sum"
    if "market_cap" in df.columns:
        agg_dict["market_cap"] = "last"

    monthly = df.groupby(["year", "month"]).agg(agg_dict)

    # Flatten column names
    monthly.columns = ["_".join(col).strip() for col in monthly.columns.values]
    monthly = monthly.reset_index()

    # Rename for clarity
    monthly = monthly.rename(columns={
        "price_mean": "btc_price_avg",
        "price_last": "btc_price_close",
        "price_first": "btc_price_open",
        "price_max": "btc_price_high",
        "price_min": "btc_price_low",
        "price_std": "btc_price_volatility",
        "volume_sum": "btc_volume",
        "market_cap_last": "btc_market_cap",
    })

    # Create date column (first of month)
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" +
        monthly["month"].astype(str).str.zfill(2) + "-01"
    )

    return monthly


def _standardize_monthly(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Standardize a dataset to monthly frequency with prefixed columns.

    Args:
        df: DataFrame with year/month columns.
        prefix: Prefix for non-key columns.

    Returns:
        Standardized monthly DataFrame.
    """
    df = df.copy()

    # Reset any multi-level column names
    if hasattr(df.columns, 'names') and df.columns.names[0] is not None:
        df.columns = [str(c) for c in df.columns]

    # Ensure year/month columns
    if "year" not in df.columns or "month" not in df.columns:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
        else:
            return pd.DataFrame()

    # Remove duplicate columns by keeping first occurrence
    df = df.loc[:, ~df.columns.duplicated()]

    # Group by year-month (in case of duplicates)
    key_cols = ["year", "month"]
    value_cols = [c for c in df.columns if c not in key_cols + ["date"]]

    # Only aggregate numeric columns
    numeric_cols = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c])]

    if numeric_cols:
        # Take first value for each year-month (data is already aggregated)
        monthly = df.groupby(key_cols)[numeric_cols].first().reset_index()
    else:
        monthly = df[key_cols].drop_duplicates()

    # Prefix columns (except year/month)
    rename_dict = {
        c: f"{prefix}_{c}" if c not in key_cols else c
        for c in monthly.columns
    }
    monthly = monthly.rename(columns=rename_dict)

    return monthly


def _add_market_controls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market control variables from external sources.

    Controls include:
    - S&P 500 index
    - US Dollar Index (DXY)
    - Federal funds rate
    - VIX volatility index

    Args:
        df: DataFrame with date column.

    Returns:
        DataFrame with market controls added.
    """
    try:
        import yfinance as yf

        # Get S&P 500 (^GSPC)
        logger.info("Fetching S&P 500 data...")
        sp500 = _fetch_market_index("^GSPC", "sp500")
        if not sp500.empty:
            df = pd.merge(df, sp500, on=["year", "month"], how="left")

        # Get DXY (Dollar Index)
        logger.info("Fetching DXY data...")
        dxy = _fetch_market_index("DX-Y.NYB", "dxy")
        if not dxy.empty:
            df = pd.merge(df, dxy, on=["year", "month"], how="left")

        # Get VIX
        logger.info("Fetching VIX data...")
        vix = _fetch_market_index("^VIX", "vix")
        if not vix.empty:
            df = pd.merge(df, vix, on=["year", "month"], how="left")

    except ImportError:
        logger.warning("yfinance not installed. Skipping market controls.")

    return df


def _fetch_market_index(ticker: str, name: str) -> pd.DataFrame:
    """
    Fetch a market index and aggregate to monthly.

    Args:
        ticker: Yahoo Finance ticker symbol.
        name: Name prefix for columns.

    Returns:
        Monthly DataFrame with index data.
    """
    try:
        import yfinance as yf

        data = yf.Ticker(ticker)
        hist = data.history(period="max")

        if hist.empty:
            return pd.DataFrame()

        hist = hist.reset_index()
        hist["year"] = hist["Date"].dt.year
        hist["month"] = hist["Date"].dt.month

        monthly = hist.groupby(["year", "month"]).agg({
            "Close": "last",
            "Volume": "sum" if ticker != "^VIX" else "mean",
        }).reset_index()

        monthly = monthly.rename(columns={
            "Close": f"{name}_close",
            "Volume": f"{name}_volume",
        })

        return monthly

    except Exception as e:
        logger.warning(f"Could not fetch {ticker}: {e}")
        return pd.DataFrame()


def _add_lagged_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged versions of key variables for time series analysis.

    Args:
        df: DataFrame sorted by date.

    Returns:
        DataFrame with lagged variables.
    """
    df = df.sort_values(["year", "month"]).reset_index(drop=True)

    # Lag BTC price (1, 3, 6 months)
    for lag in [1, 3, 6]:
        df[f"btc_price_lag{lag}"] = df["btc_price_close"].shift(lag)

    # Lag overdose deaths if available
    overdose_cols = [c for c in df.columns if "overdose" in c.lower() and "death" in c.lower()]
    for col in overdose_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag3"] = df[col].shift(3)

    # Lag seizures if available
    seizure_cols = [c for c in df.columns if "seizure" in c.lower() or "lbs" in c.lower()]
    for col in seizure_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag3"] = df[col].shift(3)

    return df


def _create_drug_market_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregate drug market indices from individual drug metrics.

    Creates:
    - total_overdose_deaths: Sum of all drug overdose deaths
    - total_seizures_value: Estimated street value of seizures
    - drug_market_index: Standardized composite index

    Args:
        df: DataFrame with overdose and seizure columns.

    Returns:
        DataFrame with drug market index columns added.
    """
    df = df.copy()

    # Estimated street value per lb (rough 2023 estimates, USD)
    # Used to weight seizures by economic value
    street_value_per_lb = {
        "cocaine_lbs": 15000,      # ~$15k/lb wholesale
        "fentanyl_lbs": 750000,    # Extremely high value per weight
        "heroin_lbs": 25000,       # ~$25k/lb
        "meth_lbs": 5000,          # ~$5k/lb
        "marijuana_lbs": 500,      # ~$500/lb (low value)
        "ecstasy_lbs": 10000,      # ~$10k/lb
        "ketamine_lbs": 8000,      # ~$8k/lb
        "other_drugs_lbs": 5000,   # Average estimate
    }

    # Calculate total seizure value
    seizure_cols = [c for c in df.columns if c.startswith("seizures_") and c.endswith("_lbs")]

    if seizure_cols:
        df["seizures_total_value"] = 0
        for col in seizure_cols:
            drug_name = col.replace("seizures_", "")
            value_per_lb = street_value_per_lb.get(drug_name, 5000)
            if col in df.columns:
                df["seizures_total_value"] += df[col].fillna(0) * value_per_lb

        # Log transform for better scaling (values are in millions)
        df["seizures_total_value_log"] = np.log1p(df["seizures_total_value"])

    # Calculate total overdose deaths (if not already present)
    overdose_cols = [c for c in df.columns if "overdose" in c and "death" in c.lower()
                     and "total" not in c and "lag" not in c]

    if "overdose_total_overdose_deaths" not in df.columns and overdose_cols:
        # Use the provided total or sum individual categories
        pass  # Already have total_overdose_deaths from CDC

    # Create standardized composite index
    # Combine demand (deaths) and supply (seizures) signals
    index_components = []

    # Standardize each component to z-scores
    if "overdose_total_overdose_deaths" in df.columns:
        deaths = df["overdose_total_overdose_deaths"]
        df["deaths_zscore"] = (deaths - deaths.mean()) / deaths.std()
        index_components.append("deaths_zscore")

    if "seizures_total_value" in df.columns:
        seizures = df["seizures_total_value"]
        df["seizures_zscore"] = (seizures - seizures.mean()) / seizures.std()
        index_components.append("seizures_zscore")

    # Composite index = average of z-scores
    if index_components:
        df["drug_market_index"] = df[index_components].mean(axis=1)

        # Also create a simple sum index (for robustness)
        if "overdose_total_overdose_deaths" in df.columns and "seizures_total_value" in df.columns:
            # Normalize to similar scales and sum
            deaths_norm = df["overdose_total_overdose_deaths"] / df["overdose_total_overdose_deaths"].max()
            seizures_norm = df["seizures_total_value"] / df["seizures_total_value"].max()
            df["drug_market_index_simple"] = deaths_norm + seizures_norm

    return df


def _add_pct_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percentage change variables.

    Args:
        df: DataFrame sorted by date.

    Returns:
        DataFrame with percentage change variables.
    """
    df = df.sort_values(["year", "month"]).reset_index(drop=True)

    # BTC price changes
    df["btc_pct_change_1m"] = df["btc_price_close"].pct_change(1) * 100
    df["btc_pct_change_3m"] = df["btc_price_close"].pct_change(3) * 100
    df["btc_pct_change_12m"] = df["btc_price_close"].pct_change(12) * 100

    # Log returns (for time series models)
    df["btc_log_return"] = np.log(df["btc_price_close"] / df["btc_price_close"].shift(1))

    return df


def _log_merge_summary(df: pd.DataFrame) -> None:
    """Log summary statistics of the merged dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("TIME SERIES MERGE SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Total observations: {len(df)}")
    logger.info(f"Date range: {df['year'].min()}-{df['month'].min():02d} to "
                f"{df['year'].max()}-{df['month'].max():02d}")

    # Count non-null values for key columns
    key_cols = ["btc_price_close", "btc_volume"]
    overdose_cols = [c for c in df.columns if "overdose" in c.lower()]
    seizure_cols = [c for c in df.columns if "seizure" in c.lower() or "_lbs" in c]

    logger.info("\nData availability:")
    for col in key_cols + overdose_cols[:2] + seizure_cols[:2]:
        if col in df.columns:
            n_valid = df[col].notna().sum()
            pct = n_valid / len(df) * 100
            logger.info(f"  {col}: {n_valid}/{len(df)} ({pct:.1f}%)")

    logger.info("=" * 60)


def get_analysis_ready_data(
    start_year: int = 2017,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Get a clean dataset ready for time series analysis.

    Applies additional cleaning and filtering for analysis:
    - Removes rows with missing BTC price
    - Forward-fills limited gaps in control variables
    - Creates analysis-ready variable set

    Args:
        start_year: Start year (default 2017 for data availability).
        end_year: End year.

    Returns:
        Analysis-ready DataFrame.
    """
    df = merge_timeseries(start_year=start_year, end_year=end_year)

    if df.empty:
        return df

    # Remove rows with missing BTC price
    df = df[df["btc_price_close"].notna()].copy()

    # Forward fill limited gaps (max 2 months) in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(method="ffill", limit=2)

    # Drop any remaining rows with too many missing values
    # Keep rows with at least 50% of values
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=int(threshold))

    return df
