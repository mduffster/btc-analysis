"""Bitcoin Price Data fetcher.

Fetches historical BTC/USD price data from multiple sources.
Primary: CoinGecko API (free tier)
Fallback: Yahoo Finance via yfinance
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from btc_analysis.config import get_config

logger = logging.getLogger(__name__)

# CoinGecko API endpoints
COINGECKO_API = "https://api.coingecko.com/api/v3"


def fetch_btc_price(
    output_path: Optional[Path] = None,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    frequency: str = "daily",
) -> pd.DataFrame:
    """
    Fetch historical Bitcoin price data.

    Args:
        output_path: Path to save the CSV output. If None, uses default from config.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format. Defaults to today.
        frequency: Data frequency - 'daily', 'monthly', or 'weekly'.

    Returns:
        DataFrame with BTC price data.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "btc_price.csv"

    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Try CoinGecko first
    df = _fetch_coingecko_price(start_date, end_date)

    if df is None or df.empty:
        # Try yfinance fallback
        logger.info("CoinGecko failed, trying yfinance...")
        df = _fetch_yfinance_price(start_date, end_date)

    if df is None or df.empty:
        # Check for cached data
        if output_path.exists():
            logger.info(f"Loading cached BTC price data from {output_path}")
            return pd.read_csv(output_path, parse_dates=["date"])
        return _create_empty_df()

    # Resample to requested frequency
    if frequency == "monthly":
        df = _resample_to_monthly(df)
    elif frequency == "weekly":
        df = _resample_to_weekly(df)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved BTC price data to {output_path}")

    return df


def _fetch_coingecko_price(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch BTC price from CoinGecko API.

    CoinGecko provides free historical data with some rate limits.

    Args:
        start_date: Start date string.
        end_date: End date string.

    Returns:
        DataFrame with daily price data or None.
    """
    # Convert dates to Unix timestamps
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp(end_date).timestamp())

    # CoinGecko market_chart/range endpoint
    url = f"{COINGECKO_API}/coins/bitcoin/market_chart/range"

    params = {
        "vs_currency": "usd",
        "from": start_ts,
        "to": end_ts,
    }

    try:
        logger.info("Fetching BTC price from CoinGecko...")
        response = requests.get(url, params=params, timeout=60)

        if response.status_code == 429:
            logger.warning("CoinGecko rate limit hit")
            return None

        response.raise_for_status()
        data = response.json()

        # Parse price data
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        market_caps = data.get("market_caps", [])

        if not prices:
            logger.warning("No price data returned from CoinGecko")
            return None

        # Create DataFrame
        df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
        df["date"] = df["date"].dt.normalize()  # Remove time component

        # Add volume and market cap if available
        if volumes and len(volumes) == len(prices):
            df["volume"] = [v[1] for v in volumes]
        if market_caps and len(market_caps) == len(prices):
            df["market_cap"] = [m[1] for m in market_caps]

        # Drop timestamp column
        df = df.drop(columns=["timestamp_ms"])

        # Remove duplicates (keep last price of the day)
        df = df.drop_duplicates(subset=["date"], keep="last")

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        # Add derived columns
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        logger.info(f"Fetched {len(df)} days of BTC price data from CoinGecko")
        return df

    except requests.RequestException as e:
        logger.error(f"CoinGecko API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing CoinGecko data: {e}")
        return None


def _fetch_yfinance_price(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch BTC price from Yahoo Finance using yfinance.

    Args:
        start_date: Start date string.
        end_date: End date string.

    Returns:
        DataFrame with daily price data or None.
    """
    try:
        import yfinance as yf

        logger.info("Fetching BTC price from Yahoo Finance...")

        # BTC-USD ticker
        btc = yf.Ticker("BTC-USD")
        hist = btc.history(start=start_date, end=end_date)

        if hist.empty:
            logger.warning("No data returned from Yahoo Finance")
            return None

        # Create standardized DataFrame
        df = pd.DataFrame({
            "date": hist.index.normalize(),
            "price": hist["Close"],
            "volume": hist["Volume"],
            "high": hist["High"],
            "low": hist["Low"],
            "open": hist["Open"],
        })

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        logger.info(f"Fetched {len(df)} days of BTC price data from Yahoo Finance")
        return df

    except ImportError:
        logger.warning("yfinance not installed. Run: pip install yfinance")
        return None
    except Exception as e:
        logger.error(f"Yahoo Finance error: {e}")
        return None


def _resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily data to monthly frequency.

    Args:
        df: Daily price DataFrame.

    Returns:
        Monthly DataFrame with average/end-of-month values.
    """
    df = df.set_index("date")

    monthly = df.resample("M").agg({
        "price": ["mean", "last", "first", "max", "min"],
        "volume": "sum" if "volume" in df.columns else "first",
    })

    # Flatten column names
    monthly.columns = ["_".join(col).strip() for col in monthly.columns.values]
    monthly = monthly.reset_index()

    # Rename columns
    monthly = monthly.rename(columns={
        "price_mean": "price_avg",
        "price_last": "price_close",
        "price_first": "price_open",
        "price_max": "price_high",
        "price_min": "price_low",
    })

    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month

    return monthly


def _resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily data to weekly frequency.

    Args:
        df: Daily price DataFrame.

    Returns:
        Weekly DataFrame.
    """
    df = df.set_index("date")

    weekly = df.resample("W").agg({
        "price": ["mean", "last", "first", "max", "min"],
        "volume": "sum" if "volume" in df.columns else "first",
    })

    weekly.columns = ["_".join(col).strip() for col in weekly.columns.values]
    weekly = weekly.reset_index()

    weekly = weekly.rename(columns={
        "price_mean": "price_avg",
        "price_last": "price_close",
        "price_first": "price_open",
        "price_max": "price_high",
        "price_min": "price_low",
    })

    weekly["year"] = weekly["date"].dt.year
    weekly["week"] = weekly["date"].dt.isocalendar().week

    return weekly


def _create_empty_df() -> pd.DataFrame:
    """Create empty DataFrame with expected schema."""
    return pd.DataFrame(columns=[
        "date", "year", "month", "price", "volume", "market_cap",
    ])


def get_monthly_btc_price(
    start_year: int = 2015,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Get monthly BTC price time series.

    Convenience function that returns monthly data.

    Args:
        start_year: Start year.
        end_year: End year.

    Returns:
        DataFrame with monthly BTC price data.
    """
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    return fetch_btc_price(
        start_date=start_date,
        end_date=end_date,
        frequency="monthly",
    )


def get_btc_halving_dates() -> pd.DataFrame:
    """
    Return Bitcoin halving event dates.

    Halvings are important events that affect BTC supply and price.

    Returns:
        DataFrame with halving dates and block heights.
    """
    halvings = [
        {"date": "2012-11-28", "block": 210000, "reward_btc": 25, "halving_num": 1},
        {"date": "2016-07-09", "block": 420000, "reward_btc": 12.5, "halving_num": 2},
        {"date": "2020-05-11", "block": 630000, "reward_btc": 6.25, "halving_num": 3},
        {"date": "2024-04-20", "block": 840000, "reward_btc": 3.125, "halving_num": 4},
    ]

    df = pd.DataFrame(halvings)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_halving_controls(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add halving cycle control variables to a DataFrame.

    Creates indicators for:
    - halving_cycle: Which halving era (1, 2, 3, 4)
    - months_since_halving: Months since last halving
    - months_to_halving: Months until next halving

    Args:
        df: DataFrame with a date column.
        date_col: Name of the date column.

    Returns:
        DataFrame with halving control variables added.
    """
    halvings = get_btc_halving_dates()
    halving_dates = halvings["date"].tolist()

    def get_halving_info(date):
        # Find which halving cycle we're in
        cycle = 0
        for i, h_date in enumerate(halving_dates):
            if date >= h_date:
                cycle = i + 1

        # Months since last halving
        if cycle > 0:
            last_halving = halving_dates[cycle - 1]
            months_since = (date.year - last_halving.year) * 12 + (date.month - last_halving.month)
        else:
            months_since = None

        # Months to next halving
        if cycle < len(halving_dates):
            next_halving = halving_dates[cycle]
            months_to = (next_halving.year - date.year) * 12 + (next_halving.month - date.month)
        else:
            months_to = None

        return pd.Series({
            "halving_cycle": cycle,
            "months_since_halving": months_since,
            "months_to_halving": months_to,
        })

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    halving_info = df[date_col].apply(get_halving_info)
    df = pd.concat([df, halving_info], axis=1)

    return df
