"""
Stablecoin data fetching and processing.
Source: DefiLlama API
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

RAW_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "stablecoins"
PROCESSED_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed"


def fetch_stablecoin_history(stablecoin_id: int, name: str) -> dict:
    """Fetch historical data for a stablecoin from DefiLlama."""
    url = f"https://stablecoins.llama.fi/stablecoin/{stablecoin_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / f"{name.lower()}_defillama.json", 'w') as f:
        json.dump(data, f)

    return data


def fetch_chain_stablecoins(chain: str) -> list:
    """Fetch stablecoin data for a specific chain."""
    url = f"https://stablecoins.llama.fi/stablecoincharts/{chain}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / f"{chain.lower()}_stablecoins.json", 'w') as f:
        json.dump(data, f)

    return data


def process_stablecoin_data(json_path: Path) -> pd.DataFrame:
    """Process stablecoin JSON data into daily DataFrame."""
    with open(json_path) as f:
        data = json.load(f)

    records = []
    # Handle both formats: dict with 'tokens' key, or direct list
    if isinstance(data, dict):
        entries = data.get('tokens', [])
    else:
        entries = data

    for entry in entries:
        if isinstance(entry, dict):
            date_val = entry.get('date')
            if date_val:
                # Handle both int and string timestamps
                ts = int(date_val)
                dt = datetime.fromtimestamp(ts, tz=None)

                # Extract circulating supply (different formats)
                circulating = entry.get('totalCirculatingUSD', entry.get('totalCirculating', entry.get('circulating', {})))
                if isinstance(circulating, dict):
                    usd_val = circulating.get('peggedUSD', 0)
                else:
                    usd_val = circulating

                records.append({
                    'date': dt,
                    'circulating_usd': usd_val
                })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values('date')
        df = df.drop_duplicates(subset=['date'], keep='last')

    return df


def aggregate_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to monthly (end of month values)."""
    df = df.copy()
    df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()

    # Take end-of-month value
    monthly = df.groupby('year_month').agg({
        'circulating_usd': 'last'
    }).reset_index()

    return monthly


def build_stablecoin_dataset(output_path: Optional[Path] = None) -> pd.DataFrame:
    """Build combined stablecoin dataset with supply changes."""

    # Process USDT (global)
    usdt_path = RAW_DIR / "usdt_defillama.json"
    if usdt_path.exists():
        usdt_df = process_stablecoin_data(usdt_path)
        usdt_monthly = aggregate_to_monthly(usdt_df)
        usdt_monthly = usdt_monthly.rename(columns={'circulating_usd': 'usdt_supply'})
    else:
        print("Warning: USDT data not found")
        usdt_monthly = pd.DataFrame()

    # Process USDC (global)
    usdc_path = RAW_DIR / "usdc_defillama.json"
    if usdc_path.exists():
        usdc_df = process_stablecoin_data(usdc_path)
        usdc_monthly = aggregate_to_monthly(usdc_df)
        usdc_monthly = usdc_monthly.rename(columns={'circulating_usd': 'usdc_supply'})
    else:
        print("Warning: USDC data not found")
        usdc_monthly = pd.DataFrame()

    # Process Tron stablecoins (proxy for illicit usage)
    tron_path = RAW_DIR / "tron_stablecoins.json"
    if tron_path.exists():
        tron_df = process_stablecoin_data(tron_path)
        tron_monthly = aggregate_to_monthly(tron_df)
        tron_monthly = tron_monthly.rename(columns={'circulating_usd': 'tron_stablecoin_supply'})
    else:
        print("Warning: Tron data not found")
        tron_monthly = pd.DataFrame()

    # Merge all
    if not usdt_monthly.empty:
        combined = usdt_monthly.copy()
    else:
        combined = pd.DataFrame({'year_month': pd.date_range('2017-01-01', '2025-12-01', freq='MS')})

    if not usdc_monthly.empty:
        combined = combined.merge(usdc_monthly, on='year_month', how='outer')

    if not tron_monthly.empty:
        combined = combined.merge(tron_monthly, on='year_month', how='outer')

    combined = combined.sort_values('year_month')

    # Calculate month-over-month changes
    for col in ['usdt_supply', 'usdc_supply', 'tron_stablecoin_supply']:
        if col in combined.columns:
            combined[f'{col}_pct_change'] = combined[col].pct_change() * 100

    # Calculate total stablecoin supply
    supply_cols = [c for c in ['usdt_supply', 'usdc_supply'] if c in combined.columns]
    if supply_cols:
        combined['total_stablecoin_supply'] = combined[supply_cols].sum(axis=1)
        combined['total_supply_pct_change'] = combined['total_stablecoin_supply'].pct_change() * 100

    # Save
    if output_path is None:
        output_path = PROCESSED_DIR / "stablecoin_monthly.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved stablecoin data to {output_path}")
    print(f"Period: {combined['year_month'].min()} to {combined['year_month'].max()}")
    print(f"Observations: {len(combined)}")

    return combined


if __name__ == "__main__":
    # Fetch fresh data if needed
    try:
        fetch_stablecoin_history(1, "usdt")
        fetch_stablecoin_history(2, "usdc")
        fetch_chain_stablecoins("Tron")
    except Exception as e:
        print(f"Warning: Could not fetch fresh data: {e}")

    # Build dataset
    df = build_stablecoin_dataset()
    print("\nSample data:")
    print(df.tail(10))
