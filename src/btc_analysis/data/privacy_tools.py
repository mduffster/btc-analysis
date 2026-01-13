"""
Privacy tool usage data fetching and processing.
Sources: DefiLlama (Tornado Cash TVL), Crystal Blockchain reports
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

RAW_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "stablecoins"
PROCESSED_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed"


def fetch_tornado_cash_tvl() -> dict:
    """Fetch Tornado Cash TVL history from DefiLlama."""
    url = "https://api.llama.fi/protocol/tornado-cash"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / "tornado_cash_tvl.json", 'w') as f:
        json.dump(data, f)

    return data


def process_tornado_cash_data(json_path: Path) -> pd.DataFrame:
    """Process Tornado Cash TVL data into daily DataFrame."""
    with open(json_path) as f:
        data = json.load(f)

    records = []
    for entry in data.get('tvl', []):
        ts = entry.get('date')
        tvl = entry.get('totalLiquidityUSD', 0)
        if ts:
            dt = datetime.fromtimestamp(ts, tz=None)
            records.append({
                'date': dt,
                'tornado_tvl_usd': tvl
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values('date')
        df = df.drop_duplicates(subset=['date'], keep='last')

    return df


def aggregate_to_monthly(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Aggregate daily data to monthly."""
    df = df.copy()
    df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()

    # Take end-of-month value and also compute mean
    monthly = df.groupby('year_month').agg({
        value_col: ['last', 'mean', 'max']
    }).reset_index()

    monthly.columns = ['year_month', f'{value_col}_eom', f'{value_col}_avg', f'{value_col}_max']
    return monthly


def build_privacy_tools_dataset(output_path: Optional[Path] = None) -> pd.DataFrame:
    """Build privacy tools dataset with TVL and changes."""

    # Process Tornado Cash
    tornado_path = RAW_DIR / "tornado_cash_tvl.json"
    if tornado_path.exists():
        tornado_df = process_tornado_cash_data(tornado_path)
        tornado_monthly = aggregate_to_monthly(tornado_df, 'tornado_tvl_usd')
    else:
        print("Warning: Tornado Cash data not found. Fetching...")
        try:
            fetch_tornado_cash_tvl()
            tornado_df = process_tornado_cash_data(tornado_path)
            tornado_monthly = aggregate_to_monthly(tornado_df, 'tornado_tvl_usd')
        except Exception as e:
            print(f"Error fetching Tornado Cash data: {e}")
            tornado_monthly = pd.DataFrame()

    if tornado_monthly.empty:
        print("No privacy tool data available")
        return pd.DataFrame()

    combined = tornado_monthly.copy()

    # Calculate month-over-month changes
    for col in ['tornado_tvl_usd_eom', 'tornado_tvl_usd_avg']:
        if col in combined.columns:
            combined[f'{col}_pct_change'] = combined[col].pct_change() * 100

    # Add structural break indicators
    # OFAC sanctions: August 8, 2022
    combined['post_ofac_sanctions'] = (combined['year_month'] >= '2022-08-01').astype(int)

    # OFAC delisting: March 21, 2025
    combined['post_ofac_delisting'] = (combined['year_month'] >= '2025-03-01').astype(int)

    # Save
    if output_path is None:
        output_path = PROCESSED_DIR / "privacy_tools_monthly.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved privacy tools data to {output_path}")
    print(f"Period: {combined['year_month'].min()} to {combined['year_month'].max()}")
    print(f"Observations: {len(combined)}")

    return combined


# Key dates for structural breaks
PRIVACY_TOOL_EVENTS = {
    'tornado_launch': '2019-12-01',       # Tornado Cash launch
    'ofac_sanctions': '2022-08-01',       # OFAC sanctions on Tornado Cash
    'tornado_delisting': '2025-03-01',    # OFAC delisting
    'wasabi_launch': '2018-07-01',        # Wasabi Wallet 1.0 launch
    'samourai_shutdown': '2024-04-01',    # Samourai founders arrested
}


if __name__ == "__main__":
    # Fetch fresh data
    try:
        fetch_tornado_cash_tvl()
    except Exception as e:
        print(f"Warning: Could not fetch fresh data: {e}")

    # Build dataset
    df = build_privacy_tools_dataset()
    if not df.empty:
        print("\nSample data:")
        print(df.tail(10))

        print("\nKey statistics:")
        print(f"Pre-sanctions avg TVL: ${df[df['post_ofac_sanctions'] == 0]['tornado_tvl_usd_avg'].mean()/1e6:.1f}M")
        print(f"Post-sanctions avg TVL: ${df[df['post_ofac_sanctions'] == 1]['tornado_tvl_usd_avg'].mean()/1e6:.1f}M")
