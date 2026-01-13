"""
Time Trend Model for Cash-to-Crypto Substitution Analysis.

This module tests for secular trends in criminal cash usage and whether
crypto adoption metrics can explain the decline in physical cash seizures.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

PROCESSED_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "time_trend"


# Key structural break dates
STRUCTURAL_BREAKS = {
    'btc_launch': '2009-01-03',           # Bitcoin genesis block
    'silk_road_launch': '2011-02-01',     # Silk Road launched
    'silk_road_shutdown': '2013-10-02',   # Silk Road seized
    'alphabay_shutdown': '2017-07-04',    # AlphaBay shutdown
    'tether_explosion': '2017-12-01',     # USDT market cap >$1B
    'microstrategy': '2020-08-11',        # MicroStrategy BTC purchase
    'tornado_sanctions': '2022-08-08',    # OFAC sanctions Tornado Cash
    'etf_filing': '2023-06-15',           # BlackRock ETF filing
    'etf_approval': '2024-01-10',         # Spot ETF approved
    'tornado_delisting': '2025-03-21',    # OFAC delists Tornado Cash
}


def load_all_data() -> pd.DataFrame:
    """Load and merge all datasets for time trend analysis."""

    # DOJ cash seizures
    doj_path = PROCESSED_DIR.parent / "raw" / "doj_cash_seizures_monthly.csv"
    doj = pd.read_csv(doj_path)
    doj['year_month'] = pd.to_datetime(doj['year_month'])
    doj = doj.rename(columns={'seizure_value': 'doj_seizure_value', 'seizure_count': 'doj_seizure_count'})

    # Stablecoin data
    stable_path = PROCESSED_DIR / "stablecoin_monthly.csv"
    if stable_path.exists():
        stable = pd.read_csv(stable_path)
        stable['year_month'] = pd.to_datetime(stable['year_month'])
    else:
        stable = pd.DataFrame()

    # Privacy tools data
    privacy_path = PROCESSED_DIR / "privacy_tools_monthly.csv"
    if privacy_path.exists():
        privacy = pd.read_csv(privacy_path)
        privacy['year_month'] = pd.to_datetime(privacy['year_month'])
    else:
        privacy = pd.DataFrame()

    # BTC prices
    btc_path = PROCESSED_DIR.parent / "raw" / "btc_prices_cryptocompare.csv"
    if btc_path.exists():
        btc = pd.read_csv(btc_path)
        btc['date'] = pd.to_datetime(btc['time'], unit='s')
        btc['year_month'] = btc['date'].dt.to_period('M').dt.to_timestamp()
        btc_monthly = btc.groupby('year_month').agg({
            'close': 'last',
            'volumeto': 'sum'
        }).reset_index()
        btc_monthly = btc_monthly.rename(columns={'close': 'btc_price', 'volumeto': 'btc_volume'})
    else:
        btc_monthly = pd.DataFrame()

    # Merge everything
    combined = doj.copy()

    if not stable.empty:
        combined = combined.merge(stable, on='year_month', how='left')

    if not privacy.empty:
        combined = combined.merge(privacy, on='year_month', how='left')

    if not btc_monthly.empty:
        combined = combined.merge(btc_monthly, on='year_month', how='left')

    # Add time trend
    combined = combined.sort_values('year_month')
    combined['time_index'] = range(len(combined))
    combined['year'] = combined['year_month'].dt.year
    combined['month'] = combined['year_month'].dt.month

    # Add structural break dummies
    for name, date_str in STRUCTURAL_BREAKS.items():
        date = pd.to_datetime(date_str)
        combined[f'post_{name}'] = (combined['year_month'] >= date).astype(int)

    return combined


def analyze_secular_trend(df: pd.DataFrame) -> dict:
    """Analyze secular trend in DOJ cash seizures."""

    # Subset to valid data
    valid = df[df['doj_seizure_value'].notna()].copy()

    # Log transform for percentage interpretation
    valid['log_seizure'] = np.log(valid['doj_seizure_value'].clip(lower=1))

    # Simple time trend
    X = sm.add_constant(valid['time_index'])
    y = valid['log_seizure']
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

    # Annualized trend rate
    monthly_trend = model.params['time_index']
    annual_trend = (np.exp(monthly_trend * 12) - 1) * 100  # Annualized % change

    results = {
        'monthly_coef': monthly_trend,
        'annual_pct_change': annual_trend,
        'pvalue': model.pvalues['time_index'],
        'r_squared': model.rsquared,
        'n_obs': len(valid),
        'model': model
    }

    return results


def analyze_btc_era_effects(df: pd.DataFrame) -> dict:
    """Test for structural breaks at key BTC milestones."""

    valid = df[df['doj_seizure_value'].notna()].copy()
    valid['log_seizure'] = np.log(valid['doj_seizure_value'].clip(lower=1))

    results = {}

    # Test each break
    breaks_to_test = ['post_btc_launch', 'post_silk_road_launch', 'post_microstrategy', 'post_etf_approval']

    for break_var in breaks_to_test:
        if break_var in valid.columns:
            X = sm.add_constant(valid[['time_index', break_var]])
            y = valid['log_seizure']
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

            results[break_var] = {
                'coef': model.params[break_var],
                'pvalue': model.pvalues[break_var],
                'pct_change': (np.exp(model.params[break_var]) - 1) * 100,
                'model': model
            }

    return results


def analyze_crypto_controls(df: pd.DataFrame) -> dict:
    """Test if crypto metrics explain seizure decline."""

    # Subset to period with crypto data
    valid = df[(df['doj_seizure_value'].notna()) & (df['year_month'] >= '2017-12-01')].copy()
    valid['log_seizure'] = np.log(valid['doj_seizure_value'].clip(lower=1))

    results = {}

    # Model 1: Time trend only (baseline)
    X1 = sm.add_constant(valid['time_index'])
    y = valid['log_seizure']
    model1 = sm.OLS(y, X1).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    results['baseline'] = {
        'r_squared': model1.rsquared,
        'time_coef': model1.params['time_index'],
        'time_pvalue': model1.pvalues['time_index']
    }

    # Model 2: Add stablecoin supply
    if 'usdt_supply' in valid.columns:
        valid_stable = valid[valid['usdt_supply'].notna()].copy()
        valid_stable['log_stablecoin'] = np.log(valid_stable['usdt_supply'].clip(lower=1))

        X2 = sm.add_constant(valid_stable[['time_index', 'log_stablecoin']])
        y2 = valid_stable['log_seizure']
        model2 = sm.OLS(y2, X2).fit(cov_type='HAC', cov_kwds={'maxlags': 6})

        results['with_stablecoin'] = {
            'r_squared': model2.rsquared,
            'time_coef': model2.params['time_index'],
            'time_pvalue': model2.pvalues['time_index'],
            'stablecoin_coef': model2.params['log_stablecoin'],
            'stablecoin_pvalue': model2.pvalues['log_stablecoin']
        }

    # Model 3: Add Tornado Cash TVL
    if 'tornado_tvl_usd_avg' in valid.columns:
        valid_tornado = valid[valid['tornado_tvl_usd_avg'].notna()].copy()
        valid_tornado['log_tornado'] = np.log(valid_tornado['tornado_tvl_usd_avg'].clip(lower=1))

        controls = ['time_index', 'log_tornado']
        if 'log_stablecoin' not in valid_tornado.columns and 'usdt_supply' in valid_tornado.columns:
            valid_tornado['log_stablecoin'] = np.log(valid_tornado['usdt_supply'].clip(lower=1))
        if 'log_stablecoin' in valid_tornado.columns:
            controls.append('log_stablecoin')

        X3 = sm.add_constant(valid_tornado[controls])
        y3 = valid_tornado['log_seizure']
        model3 = sm.OLS(y3, X3).fit(cov_type='HAC', cov_kwds={'maxlags': 6})

        results['with_tornado'] = {
            'r_squared': model3.rsquared,
            'time_coef': model3.params['time_index'],
            'time_pvalue': model3.pvalues['time_index'],
            'tornado_coef': model3.params['log_tornado'],
            'tornado_pvalue': model3.pvalues['log_tornado']
        }

    return results


def run_full_trend_analysis(output_dir: Optional[Path] = None) -> dict:
    """Run complete time trend analysis."""

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_all_data()

    # Run analyses
    secular_results = analyze_secular_trend(df)
    break_results = analyze_btc_era_effects(df)
    control_results = analyze_crypto_controls(df)

    # Generate report
    lines = []
    lines.append("=" * 80)
    lines.append("TIME TREND ANALYSIS: CASH-TO-CRYPTO SUBSTITUTION")
    lines.append("=" * 80)
    lines.append("")

    lines.append("DATA SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total observations: {len(df)}")
    lines.append(f"Period: {df['year_month'].min()} to {df['year_month'].max()}")
    lines.append(f"Pre-BTC observations: {len(df[df['year_month'] < '2009-01-01'])}")
    lines.append(f"Post-BTC observations: {len(df[df['year_month'] >= '2009-01-01'])}")
    lines.append("")

    lines.append("SECULAR TREND IN CASH SEIZURES")
    lines.append("-" * 40)
    lines.append(f"Monthly trend coefficient: {secular_results['monthly_coef']:.6f}")
    lines.append(f"Annualized change: {secular_results['annual_pct_change']:.2f}%")
    lines.append(f"P-value: {secular_results['pvalue']:.4f}")
    lines.append(f"R-squared: {secular_results['r_squared']:.4f}")
    lines.append("")

    lines.append("STRUCTURAL BREAK ANALYSIS")
    lines.append("-" * 40)
    for name, res in break_results.items():
        stars = ""
        if res['pvalue'] < 0.01:
            stars = "***"
        elif res['pvalue'] < 0.05:
            stars = "**"
        elif res['pvalue'] < 0.10:
            stars = "*"
        lines.append(f"{name}: {res['pct_change']:+.1f}% (p={res['pvalue']:.4f}) {stars}")
    lines.append("")

    lines.append("CRYPTO CONTROLS ANALYSIS (2017-12 onwards)")
    lines.append("-" * 40)
    for model_name, res in control_results.items():
        lines.append(f"\n{model_name.upper()}:")
        lines.append(f"  R-squared: {res['r_squared']:.4f}")
        lines.append(f"  Time trend: {res['time_coef']:.6f} (p={res['time_pvalue']:.4f})")
        if 'stablecoin_coef' in res:
            lines.append(f"  Stablecoin: {res['stablecoin_coef']:.4f} (p={res['stablecoin_pvalue']:.4f})")
        if 'tornado_coef' in res:
            lines.append(f"  Tornado TVL: {res['tornado_coef']:.4f} (p={res['tornado_pvalue']:.4f})")
    lines.append("")

    lines.append("=" * 80)
    lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.10")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)

    with open(output_dir / "time_trend_analysis.txt", 'w') as f:
        f.write(report)

    return {
        'secular': secular_results,
        'breaks': break_results,
        'controls': control_results,
        'data': df
    }


if __name__ == "__main__":
    results = run_full_trend_analysis()
