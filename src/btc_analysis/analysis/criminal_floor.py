"""Criminal Utility Floor Analysis.

Tests the hypothesis that criminal/gray utility provides a price floor for BTC.

Key insight: We don't need to measure criminal VOLUME, but rather whether
the ESTIMATED criminal utility is sufficient to explain price floors.

Data sources:
- Chainalysis annual illicit crypto estimates
- BTC price drawdowns and recovery levels
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from btc_analysis.config import get_config

logger = logging.getLogger(__name__)

# Chainalysis illicit volume estimates ($ billions)
# Sources: Chainalysis Crypto Crime Reports 2020-2025
# Note: These are KNOWN illicit only - gray market (capital flight, tax evasion) not included
CHAINALYSIS_ILLICIT_VOLUME = {
    2019: 10.0,   # ~2.1% of volume, outlier due to PlusToken
    2020: 7.8,    # 0.34-0.62% of volume
    2021: 18.0,   # Revised estimate
    2022: 31.0,   # Including sanctioned services (Garantex)
    2023: 46.1,   # Revised from initial $24.2B
    2024: 57.2,   # Revised from initial $40.9B
}

# Illicit share of total crypto volume
CHAINALYSIS_ILLICIT_SHARE = {
    2019: 0.021,  # 2.1%
    2020: 0.0062, # 0.62% (revised)
    2021: 0.0015, # 0.15%
    2022: 0.0024, # 0.24%
    2023: 0.0061, # 0.61%
    2024: 0.0014, # 0.14% (will be revised up)
}

# Gray market multiplier: how much larger is total gray/criminal vs known illicit?
# Conservative estimates suggest known illicit is 10-20% of actual gray flows
# Capital flight alone could be $100-200B/year
GRAY_MARKET_MULTIPLIER = {
    'conservative': 3,   # 3x known illicit
    'moderate': 5,       # 5x known illicit
    'aggressive': 10,    # 10x known illicit
}


def build_criminal_utility_timeseries() -> pd.DataFrame:
    """
    Build annual time series of estimated criminal/gray crypto utility.

    Returns DataFrame with:
    - illicit_volume: Known illicit (Chainalysis)
    - gray_volume_*: Estimated total gray market (with multipliers)
    - btc_share: Estimated BTC share of illicit (was ~100% pre-2020, now ~37%)
    """
    # BTC share of illicit volume over time (Chainalysis)
    # Pre-2020: BTC dominated, ~80-100%
    # Post-2021: Stablecoins took over, BTC now ~37%
    btc_share = {
        2019: 0.85,
        2020: 0.75,
        2021: 0.55,
        2022: 0.45,
        2023: 0.40,
        2024: 0.37,
    }

    rows = []
    for year in range(2019, 2025):
        row = {'year': year}
        row['illicit_volume_bn'] = CHAINALYSIS_ILLICIT_VOLUME.get(year, np.nan)
        row['illicit_share'] = CHAINALYSIS_ILLICIT_SHARE.get(year, np.nan)
        row['btc_share_of_illicit'] = btc_share.get(year, 0.4)

        # BTC-specific illicit volume
        row['btc_illicit_bn'] = row['illicit_volume_bn'] * row['btc_share_of_illicit']

        # Gray market estimates (including capital flight, tax evasion, etc.)
        for scenario, mult in GRAY_MARKET_MULTIPLIER.items():
            row[f'gray_volume_{scenario}_bn'] = row['illicit_volume_bn'] * mult
            row[f'btc_gray_{scenario}_bn'] = row['btc_illicit_bn'] * mult

        rows.append(row)

    return pd.DataFrame(rows)


def estimate_criminal_float(gray_volume_bn: float, holding_period_months: float = 3) -> float:
    """
    Estimate the "float" of criminal/gray funds sitting in BTC at any time.

    If $X/year flows through BTC for criminal purposes, and average holding
    period is N months, then float = X * (N/12).
    """
    return gray_volume_bn * (holding_period_months / 12)


def calculate_floor_value(float_bn: float, velocity: float = 2) -> float:
    """
    Estimate the market cap that criminal float could justify.

    If criminals hold $X in BTC at any time, and velocity is V (how many
    times per year the average dollar turns over), then:

    Justified market cap = float * velocity multiplier

    Higher velocity = criminals are active traders, each dollar supports
    more market cap (like money multiplier in banking).
    """
    return float_bn * velocity


def run_floor_analysis(
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run the criminal utility floor analysis.

    Tests:
    1. Is estimated criminal float sufficient to explain BTC price floors?
    2. Do drawdown floors correlate with criminal utility estimates?
    3. What % of market cap could criminal utility explain?
    """
    config = get_config()

    if output_dir is None:
        output_dir = config.paths.outputs_dir / "criminal_floor"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Build criminal utility estimates
    logger.info("Building criminal utility estimates...")
    criminal_df = build_criminal_utility_timeseries()
    results['criminal_estimates'] = criminal_df.to_dict('records')

    # Get BTC price data
    logger.info("Loading BTC price data...")
    try:
        import yfinance as yf
        btc = yf.download('BTC-USD', start='2019-01-01', end='2025-01-01', progress=False)
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = [c[0] for c in btc.columns]
    except Exception as e:
        logger.error(f"Failed to load BTC data: {e}")
        return results

    # Annual BTC metrics
    btc['year'] = btc.index.year
    annual_btc = btc.groupby('year').agg({
        'Close': ['mean', 'min', 'max', 'last'],
        'Volume': 'sum'
    })
    annual_btc.columns = ['avg_price', 'min_price', 'max_price', 'year_end_price', 'total_volume']
    annual_btc = annual_btc.reset_index()

    # Calculate drawdown from peak
    annual_btc['peak_to_date'] = annual_btc['max_price'].cummax()
    annual_btc['drawdown_from_peak'] = (annual_btc['min_price'] - annual_btc['peak_to_date']) / annual_btc['peak_to_date']

    # Merge with criminal estimates
    merged = annual_btc.merge(criminal_df, on='year')

    # Estimate market cap at various points
    # BTC supply roughly 19M coins
    btc_supply = {2019: 18.0, 2020: 18.5, 2021: 18.8, 2022: 19.2, 2023: 19.5, 2024: 19.7}
    merged['btc_supply_m'] = merged['year'].map(btc_supply)
    merged['market_cap_avg_bn'] = merged['avg_price'] * merged['btc_supply_m'] / 1000
    merged['market_cap_min_bn'] = merged['min_price'] * merged['btc_supply_m'] / 1000
    merged['market_cap_max_bn'] = merged['max_price'] * merged['btc_supply_m'] / 1000

    # 1. What % of market cap could criminal utility explain?
    logger.info("Calculating criminal utility as % of market cap...")

    floor_analysis = []
    for _, row in merged.iterrows():
        floor_row = {'year': row['year']}
        floor_row['market_cap_avg_bn'] = row['market_cap_avg_bn']
        floor_row['market_cap_min_bn'] = row['market_cap_min_bn']

        for scenario in ['conservative', 'moderate', 'aggressive']:
            gray_vol = row[f'btc_gray_{scenario}_bn']

            # Float with different holding periods
            for hold_months in [3, 6, 12]:
                float_bn = estimate_criminal_float(gray_vol, hold_months)

                # As % of average market cap
                pct_of_avg = (float_bn / row['market_cap_avg_bn']) * 100 if row['market_cap_avg_bn'] > 0 else 0

                # As % of minimum market cap (floor)
                pct_of_min = (float_bn / row['market_cap_min_bn']) * 100 if row['market_cap_min_bn'] > 0 else 0

                floor_row[f'float_{scenario}_{hold_months}mo_bn'] = float_bn
                floor_row[f'pct_avg_{scenario}_{hold_months}mo'] = pct_of_avg
                floor_row[f'pct_min_{scenario}_{hold_months}mo'] = pct_of_min

        floor_analysis.append(floor_row)

    floor_df = pd.DataFrame(floor_analysis)
    results['floor_analysis'] = floor_df.to_dict('records')

    # 2. Summary statistics
    logger.info("Calculating summary statistics...")

    # Focus on moderate scenario, 6-month holding period
    summary_col = 'pct_min_moderate_6mo'
    if summary_col in floor_df.columns:
        results['summary'] = {
            'scenario': 'moderate (5x known illicit), 6-month hold',
            'avg_pct_of_floor': float(floor_df[summary_col].mean()),
            'min_pct_of_floor': float(floor_df[summary_col].min()),
            'max_pct_of_floor': float(floor_df[summary_col].max()),
            'interpretation': None,
        }

        avg_pct = floor_df[summary_col].mean()
        if avg_pct >= 50:
            results['summary']['interpretation'] = (
                f"Criminal float explains ~{avg_pct:.0f}% of BTC's price floor. "
                "This is substantial - criminal utility could be the primary floor."
            )
        elif avg_pct >= 20:
            results['summary']['interpretation'] = (
                f"Criminal float explains ~{avg_pct:.0f}% of BTC's price floor. "
                "Meaningful but not dominant - other factors also matter."
            )
        else:
            results['summary']['interpretation'] = (
                f"Criminal float explains only ~{avg_pct:.0f}% of BTC's price floor. "
                "Either multiplier is too conservative or speculation dominates even the floor."
            )

    # 3. Bear market floor analysis
    logger.info("Analyzing bear market floors...")

    # 2018 bear market: peak ~$20K, floor ~$3K (85% drawdown)
    # 2022 bear market: peak ~$69K, floor ~$16K (77% drawdown)

    bear_markets = [
        {'period': '2018 bear', 'peak_bn': 350, 'floor_bn': 55, 'drawdown': 0.85},
        {'period': '2022 bear', 'peak_bn': 1300, 'floor_bn': 310, 'drawdown': 0.76},
    ]

    # What criminal float was present during these floors?
    results['bear_market_analysis'] = []
    for bear in bear_markets:
        year = 2018 if '2018' in bear['period'] else 2022
        if year in CHAINALYSIS_ILLICIT_VOLUME:
            illicit = CHAINALYSIS_ILLICIT_VOLUME[year]
            btc_share = 0.8 if year == 2018 else 0.45
            btc_illicit = illicit * btc_share

            analysis = {
                'period': bear['period'],
                'floor_market_cap_bn': bear['floor_bn'],
                'btc_illicit_annual_bn': btc_illicit,
                'illicit_pct_of_floor': (btc_illicit / bear['floor_bn']) * 100,
            }

            # With gray market multiplier
            for scenario, mult in GRAY_MARKET_MULTIPLIER.items():
                gray = btc_illicit * mult
                float_6mo = gray * 0.5  # 6 month holding period
                analysis[f'{scenario}_float_bn'] = float_6mo
                analysis[f'{scenario}_pct_of_floor'] = (float_6mo / bear['floor_bn']) * 100

            results['bear_market_analysis'].append(analysis)

    # Save results
    import json

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_dir / 'criminal_floor_results.json', 'w') as f:
        json.dump(convert_types(results), f, indent=2)

    # Generate report
    report = _generate_report(results)
    with open(output_dir / 'criminal_floor_analysis.txt', 'w') as f:
        f.write(report)

    logger.info(f"Results saved to {output_dir}")

    return results


def _generate_report(results: Dict[str, Any]) -> str:
    """Generate text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("CRIMINAL UTILITY FLOOR ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Hypothesis: Criminal/gray market utility provides a price floor for BTC")
    lines.append("")
    lines.append("Methodology:")
    lines.append("- Known illicit volume from Chainalysis (lower bound)")
    lines.append("- Gray market multiplier: 3-10x known illicit (capital flight, tax evasion)")
    lines.append("- Float = annual volume * (holding period / 12)")
    lines.append("- Compare float to market cap at yearly lows (floor)")
    lines.append("")

    if 'criminal_estimates' in results:
        lines.append("KNOWN ILLICIT VOLUME (Chainalysis)")
        lines.append("-" * 40)
        lines.append("Year | Illicit ($B) | BTC Share | BTC Illicit ($B)")
        lines.append("-" * 50)
        for row in results['criminal_estimates']:
            lines.append(f"{row['year']} |    {row['illicit_volume_bn']:5.1f}    |   {row['btc_share_of_illicit']:.0%}   |     {row['btc_illicit_bn']:5.1f}")
        lines.append("")

    if 'summary' in results:
        lines.append("FLOOR EXPLANATION SUMMARY")
        lines.append("-" * 40)
        s = results['summary']
        lines.append(f"Scenario: {s['scenario']}")
        lines.append(f"Criminal float as % of price floor:")
        lines.append(f"  Average: {s['avg_pct_of_floor']:.1f}%")
        lines.append(f"  Range: {s['min_pct_of_floor']:.1f}% - {s['max_pct_of_floor']:.1f}%")
        lines.append(f"\n{s['interpretation']}")
        lines.append("")

    if 'bear_market_analysis' in results:
        lines.append("BEAR MARKET FLOOR ANALYSIS")
        lines.append("-" * 40)
        for bear in results['bear_market_analysis']:
            lines.append(f"\n{bear['period'].upper()}")
            lines.append(f"  Floor market cap: ${bear['floor_market_cap_bn']:.0f}B")
            lines.append(f"  BTC illicit (annual): ${bear['btc_illicit_annual_bn']:.1f}B")
            lines.append(f"  Known illicit as % of floor: {bear['illicit_pct_of_floor']:.1f}%")
            lines.append(f"  With gray multipliers (6mo float):")
            for scenario in ['conservative', 'moderate', 'aggressive']:
                pct = bear[f'{scenario}_pct_of_floor']
                lines.append(f"    {scenario}: {pct:.1f}%")
        lines.append("")

    lines.append("=" * 80)
    lines.append("INTERPRETATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("If criminal float explains >50% of the price floor, the thesis is supported:")
    lines.append("Criminal utility provides the fundamental value floor.")
    lines.append("")
    lines.append("If <20%, either:")
    lines.append("1. Gray market multiplier is too conservative")
    lines.append("2. Holding period is longer than estimated")
    lines.append("3. Other factors (speculation, narratives) dominate even the floor")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_floor_analysis()
