"""Criminal/Gray Market Proxy Construction and Analysis.

Builds better proxies for criminal/gray crypto demand using:
1. Geographic BTC premiums (capital flight signal)
2. Sanctions-related premiums (Russia, Iran proxies)
3. High-inflation country premiums (Argentina, Nigeria, Turkey)
4. Combined "gray demand" index

Tests the floor effect hypothesis using these proxies.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from btc_analysis.config import get_config
from btc_analysis.data.btc_price import fetch_btc_price

logger = logging.getLogger(__name__)

# Country classifications for proxy construction
CAPITAL_FLIGHT_COUNTRIES = ['ARS', 'NGN', 'VES', 'EGP']  # Strict capital controls
SANCTIONS_COUNTRIES = ['RUB', 'UAH', 'BYN']  # Sanctions-related
INFLATION_HEDGE_COUNTRIES = ['ARS', 'TRY', 'NGN', 'BRL']  # High inflation
DEVELOPED_BASELINE = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF']  # Baseline


def load_premium_data() -> pd.DataFrame:
    """Load monthly BTC premium data from IMF crypto shadow rates."""
    config = get_config()
    xlsx_path = config.paths.raw_dir / "Crypto Parallel Exchange Rates_February 18 2025.xlsx"

    if not xlsx_path.exists():
        logger.error(f"Premium data not found at {xlsx_path}")
        return pd.DataFrame()

    df = pd.read_excel(xlsx_path, sheet_name='Data')
    df.columns = ['date', 'currency', 'btc_volume', 'volume_usd',
                  'official_fx', 'crypto_fx', 'premium_pct']

    df['date'] = pd.to_datetime(df['date'])

    return df


def construct_gray_demand_proxies(premium_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct monthly gray market demand proxies from BTC premiums.

    Proxies:
    1. capital_flight_index: Average premium in capital-controlled countries
    2. sanctions_index: Premium in sanctioned countries (esp. post-2022)
    3. inflation_hedge_index: Premium in high-inflation countries
    4. gray_demand_index: Combined index
    5. premium_dispersion: Std dev of premiums (arbitrage opportunity)
    """
    results = []

    for date in premium_df['date'].unique():
        month_data = premium_df[premium_df['date'] == date].copy()

        row = {'date': date}

        # Capital flight index
        cf_data = month_data[month_data['currency'].isin(CAPITAL_FLIGHT_COUNTRIES)]
        row['capital_flight_premium'] = cf_data['premium_pct'].mean() if len(cf_data) > 0 else np.nan
        row['capital_flight_n'] = cf_data['premium_pct'].notna().sum()

        # Sanctions index
        sanc_data = month_data[month_data['currency'].isin(SANCTIONS_COUNTRIES)]
        row['sanctions_premium'] = sanc_data['premium_pct'].mean() if len(sanc_data) > 0 else np.nan
        row['sanctions_n'] = sanc_data['premium_pct'].notna().sum()

        # Inflation hedge index
        inf_data = month_data[month_data['currency'].isin(INFLATION_HEDGE_COUNTRIES)]
        row['inflation_hedge_premium'] = inf_data['premium_pct'].mean() if len(inf_data) > 0 else np.nan

        # Developed baseline
        dev_data = month_data[month_data['currency'].isin(DEVELOPED_BASELINE)]
        row['developed_premium'] = dev_data['premium_pct'].mean() if len(dev_data) > 0 else np.nan

        # Excess premium (high-risk minus developed)
        if pd.notna(row['capital_flight_premium']) and pd.notna(row['developed_premium']):
            row['excess_premium'] = row['capital_flight_premium'] - row['developed_premium']
        else:
            row['excess_premium'] = np.nan

        # Premium dispersion (arbitrage signal)
        all_premiums = month_data['premium_pct'].dropna()
        row['premium_dispersion'] = all_premiums.std() if len(all_premiums) > 5 else np.nan
        row['premium_max'] = all_premiums.max() if len(all_premiums) > 0 else np.nan
        row['premium_min'] = all_premiums.min() if len(all_premiums) > 0 else np.nan

        # Total BTC volume in high-risk countries
        hr_volume = month_data[month_data['currency'].isin(
            CAPITAL_FLIGHT_COUNTRIES + SANCTIONS_COUNTRIES
        )]['volume_usd'].sum()
        row['high_risk_volume'] = hr_volume if hr_volume > 0 else np.nan

        results.append(row)

    proxy_df = pd.DataFrame(results)
    proxy_df = proxy_df.sort_values('date').reset_index(drop=True)

    # Create combined gray demand index (z-score average of components)
    def zscore(s):
        return (s - s.mean()) / s.std() if s.std() > 0 else 0

    components = ['capital_flight_premium', 'sanctions_premium', 'excess_premium']
    for comp in components:
        if comp in proxy_df.columns:
            proxy_df[f'{comp}_z'] = zscore(proxy_df[comp])

    # Gray demand index = average of z-scored components
    z_cols = [f'{c}_z' for c in components if f'{c}_z' in proxy_df.columns]
    if z_cols:
        proxy_df['gray_demand_index'] = proxy_df[z_cols].mean(axis=1)

    return proxy_df


def run_gray_demand_analysis(
    output_dir: Optional[Path] = None,
    start_year: int = 2019,
    end_year: int = 2024,
) -> Dict[str, Any]:
    """
    Run analysis using geographic premium proxies for gray market demand.

    Tests:
    1. Do gray market premiums predict BTC returns?
    2. Is the effect stronger in down markets (floor effect)?
    3. How does this compare to the CBP seizure proxy?
    """
    config = get_config()

    if output_dir is None:
        output_dir = config.paths.outputs_dir / "gray_demand"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Load data
    logger.info("Loading premium data...")
    premium_df = load_premium_data()
    if premium_df.empty:
        return {"error": "No premium data available"}

    # Construct proxies
    logger.info("Constructing gray demand proxies...")
    proxy_df = construct_gray_demand_proxies(premium_df)

    results["proxy_summary"] = {
        "n_months": len(proxy_df),
        "date_range": f"{proxy_df['date'].min().strftime('%Y-%m')} to {proxy_df['date'].max().strftime('%Y-%m')}",
        "capital_flight_mean": float(proxy_df['capital_flight_premium'].mean()),
        "capital_flight_max": float(proxy_df['capital_flight_premium'].max()),
        "sanctions_mean": float(proxy_df['sanctions_premium'].mean()),
        "excess_premium_mean": float(proxy_df['excess_premium'].mean()),
    }

    # Load BTC price
    logger.info("Loading BTC price...")
    btc = fetch_btc_price(
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
    )

    if btc.empty:
        return {"error": "No BTC price data"}

    # Aggregate to monthly
    btc = btc.copy()
    btc['date'] = pd.to_datetime(btc['date'])
    btc_monthly = btc.groupby(btc['date'].dt.to_period('M')).agg({
        'price': 'last',
        'volume': 'sum'
    }).reset_index()
    btc_monthly['date'] = btc_monthly['date'].dt.to_timestamp()
    btc_monthly['log_btc'] = np.log(btc_monthly['price'])
    btc_monthly['btc_return'] = btc_monthly['log_btc'].diff()
    btc_monthly['volume_z'] = (btc_monthly['volume'] - btc_monthly['volume'].mean()) / btc_monthly['volume'].std()

    # Load S&P for control
    try:
        import yfinance as yf
        sp500_data = yf.download("^GSPC", start=f"{start_year}-01-01",
                                  end=f"{end_year+1}-01-01", progress=False)
        if isinstance(sp500_data.columns, pd.MultiIndex):
            sp500_data.columns = [col[0] for col in sp500_data.columns]
        sp500 = sp500_data['Close'].resample('ME').last()
        sp500_df = pd.DataFrame({'date': sp500.index, 'sp500_price': sp500.values})
        sp500_df['date'] = sp500_df['date'].dt.to_period('M').dt.to_timestamp()
        sp500_df['log_sp500'] = np.log(sp500_df['sp500_price'])
        sp500_df['sp500_return'] = sp500_df['log_sp500'].diff()
        btc_monthly = btc_monthly.merge(sp500_df[['date', 'sp500_return']], on='date', how='left')
    except Exception as e:
        logger.warning(f"Could not load S&P: {e}")

    # Merge with proxies
    proxy_df['date'] = pd.to_datetime(proxy_df['date']).dt.to_period('M').dt.to_timestamp()
    df = btc_monthly.merge(proxy_df, on='date', how='inner')

    results["merged_data"] = {
        "n_obs": len(df),
        "date_range": f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}",
    }

    # Create lagged proxies
    for proxy in ['capital_flight_premium', 'gray_demand_index', 'excess_premium']:
        if proxy in df.columns:
            df[f'{proxy}_lag1'] = df[proxy].shift(1)
            df[f'{proxy}_z'] = (df[proxy] - df[proxy].mean()) / df[proxy].std()
            df[f'{proxy}_z_lag1'] = df[f'{proxy}_z'].shift(1)

    # Market regime
    if 'sp500_return' in df.columns:
        df['down_market'] = (df['sp500_return'] < 0).astype(int)

    df_clean = df.dropna(subset=['btc_return', 'sp500_return']).copy()

    if len(df_clean) < 20:
        return {"error": "Insufficient data after merge"}

    # 1. Basic regression: gray demand -> BTC returns
    results["basic_regression"] = _run_basic_regression(df_clean)

    # 2. Floor effect test: stronger in down markets?
    results["floor_effect"] = _run_floor_effect_test(df_clean)

    # 3. Compare proxies
    results["proxy_comparison"] = _compare_proxies(df_clean)

    # 4. Time series of key proxies
    results["proxy_timeseries"] = {
        "dates": df_clean['date'].dt.strftime('%Y-%m').tolist(),
        "capital_flight": df_clean['capital_flight_premium'].tolist(),
        "btc_return": df_clean['btc_return'].tolist(),
    }

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
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        return obj

    with open(output_dir / "gray_demand_results.json", "w") as f:
        json.dump(convert_types(results), f, indent=2)

    df_clean.to_csv(output_dir / "gray_demand_data.csv", index=False)

    # Generate report
    report = _generate_report(results)
    with open(output_dir / "gray_demand_analysis.txt", "w") as f:
        f.write(report)

    logger.info(f"Results saved to {output_dir}")

    return results


def _run_basic_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """Test if gray demand proxies predict BTC returns."""
    results = {}

    proxies_to_test = [
        ('capital_flight_premium_z_lag1', 'Capital Flight Premium'),
        ('excess_premium', 'Excess Premium (vs Developed)'),
        ('gray_demand_index', 'Gray Demand Index'),
    ]

    for proxy_col, proxy_name in proxies_to_test:
        if proxy_col not in df.columns:
            continue

        required = ['btc_return', 'sp500_return', proxy_col]
        df_reg = df.dropna(subset=required).copy()

        if len(df_reg) < 15:
            continue

        # Model: btc_return ~ sp500_return + proxy
        X = df_reg[['sp500_return', proxy_col]]
        X = sm.add_constant(X)
        y = df_reg['btc_return']

        try:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
            results[proxy_col] = {
                'name': proxy_name,
                'n_obs': int(model.nobs),
                'r_squared': float(model.rsquared),
                'coefficient': float(model.params.get(proxy_col, np.nan)),
                'std_err': float(model.bse.get(proxy_col, np.nan)),
                'p_value': float(model.pvalues.get(proxy_col, np.nan)),
            }
        except Exception as e:
            results[proxy_col] = {'error': str(e)}

    return results


def _run_floor_effect_test(df: pd.DataFrame) -> Dict[str, Any]:
    """Test if gray demand effect is stronger in down markets."""
    results = {}

    if 'down_market' not in df.columns or 'capital_flight_premium_z_lag1' not in df.columns:
        return {'error': 'Missing required columns'}

    proxy_col = 'capital_flight_premium_z_lag1'

    # Split sample
    down = df[df['down_market'] == 1].dropna(subset=['btc_return', 'sp500_return', proxy_col])
    up = df[df['down_market'] == 0].dropna(subset=['btc_return', 'sp500_return', proxy_col])

    results['sample_sizes'] = {
        'down_months': len(down),
        'up_months': len(up),
    }

    for regime_name, regime_df in [('down_market', down), ('up_market', up)]:
        if len(regime_df) < 10:
            results[regime_name] = {'insufficient_data': True}
            continue

        X = regime_df[['sp500_return', proxy_col]]
        X = sm.add_constant(X)
        y = regime_df['btc_return']

        try:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 2})
            results[regime_name] = {
                'n_obs': int(model.nobs),
                'r_squared': float(model.rsquared),
                'proxy_coef': float(model.params.get(proxy_col, np.nan)),
                'proxy_se': float(model.bse.get(proxy_col, np.nan)),
                'proxy_pvalue': float(model.pvalues.get(proxy_col, np.nan)),
            }
        except Exception as e:
            results[regime_name] = {'error': str(e)}

    # Compare coefficients
    if ('down_market' in results and 'up_market' in results and
        not results['down_market'].get('insufficient_data') and
        not results['up_market'].get('insufficient_data')):

        coef_down = results['down_market']['proxy_coef']
        coef_up = results['up_market']['proxy_coef']
        se_down = results['down_market']['proxy_se']
        se_up = results['up_market']['proxy_se']

        diff = coef_down - coef_up
        se_diff = np.sqrt(se_down**2 + se_up**2)
        z_stat = diff / se_diff if se_diff > 0 else 0
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        results['comparison'] = {
            'down_coef': float(coef_down),
            'up_coef': float(coef_up),
            'difference': float(diff),
            'z_statistic': float(z_stat),
            'p_value': float(p_diff),
            'stronger_in_down': coef_down > coef_up,
        }

    # Asymmetric model
    df_asym = df.dropna(subset=['btc_return', 'sp500_return', proxy_col, 'down_market']).copy()
    if len(df_asym) >= 20:
        df_asym['down_x_proxy'] = df_asym['down_market'] * df_asym[proxy_col]

        X = df_asym[['sp500_return', proxy_col, 'down_market', 'down_x_proxy']]
        X = sm.add_constant(X)
        y = df_asym['btc_return']

        try:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
            results['asymmetric_model'] = {
                'n_obs': int(model.nobs),
                'base_effect': float(model.params.get(proxy_col, np.nan)),
                'base_pvalue': float(model.pvalues.get(proxy_col, np.nan)),
                'down_shift': float(model.params.get('down_x_proxy', np.nan)),
                'down_shift_pvalue': float(model.pvalues.get('down_x_proxy', np.nan)),
                'total_effect_down': float(
                    model.params.get(proxy_col, 0) + model.params.get('down_x_proxy', 0)
                ),
            }
        except Exception as e:
            results['asymmetric_model'] = {'error': str(e)}

    return results


def _compare_proxies(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare predictive power of different proxies."""
    results = {}

    proxies = [
        'capital_flight_premium_z_lag1',
        'sanctions_premium',
        'excess_premium',
        'premium_dispersion',
    ]

    for proxy in proxies:
        if proxy not in df.columns:
            continue

        df_test = df.dropna(subset=['btc_return', 'sp500_return', proxy]).copy()
        if len(df_test) < 15:
            continue

        # Simple correlation
        corr = df_test['btc_return'].corr(df_test[proxy])

        # Regression R-squared
        X = sm.add_constant(df_test[['sp500_return', proxy]])
        y = df_test['btc_return']

        try:
            model = sm.OLS(y, X).fit()
            results[proxy] = {
                'correlation': float(corr),
                'incremental_r2': float(model.rsquared),
                'coefficient': float(model.params.get(proxy, np.nan)),
                'p_value': float(model.pvalues.get(proxy, np.nan)),
            }
        except:
            pass

    return results


def _generate_report(results: Dict[str, Any]) -> str:
    """Generate text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("GRAY MARKET DEMAND ANALYSIS")
    lines.append("Using Geographic BTC Premiums as Proxy")
    lines.append("=" * 80)
    lines.append("")

    if "proxy_summary" in results:
        ps = results["proxy_summary"]
        lines.append("PROXY SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Period: {ps.get('date_range', 'N/A')}")
        lines.append(f"  N months: {ps.get('n_months', 'N/A')}")
        lines.append(f"  Capital flight premium (mean): {ps.get('capital_flight_mean', 0):.2f}%")
        lines.append(f"  Capital flight premium (max): {ps.get('capital_flight_max', 0):.2f}%")
        lines.append(f"  Excess premium vs developed: {ps.get('excess_premium_mean', 0):.2f}%")
        lines.append("")

    if "basic_regression" in results:
        lines.append("BASIC REGRESSION: Gray Demand -> BTC Returns")
        lines.append("-" * 40)
        for proxy, data in results["basic_regression"].items():
            if 'error' in data:
                continue
            sig = '***' if data['p_value'] < 0.01 else '**' if data['p_value'] < 0.05 else '*' if data['p_value'] < 0.1 else ''
            lines.append(f"  {data.get('name', proxy)}:")
            lines.append(f"    coef={data['coefficient']:+.4f}, p={data['p_value']:.4f} {sig}")
        lines.append("")

    if "floor_effect" in results:
        fe = results["floor_effect"]
        lines.append("FLOOR EFFECT TEST")
        lines.append("-" * 40)
        if 'sample_sizes' in fe:
            lines.append(f"  Down months: {fe['sample_sizes']['down_months']}, Up months: {fe['sample_sizes']['up_months']}")

        if 'down_market' in fe and not fe['down_market'].get('insufficient_data'):
            dm = fe['down_market']
            sig = '***' if dm['proxy_pvalue'] < 0.01 else '**' if dm['proxy_pvalue'] < 0.05 else '*' if dm['proxy_pvalue'] < 0.1 else ''
            lines.append(f"  DOWN: coef={dm['proxy_coef']:+.4f} (p={dm['proxy_pvalue']:.4f}) {sig}")

        if 'up_market' in fe and not fe['up_market'].get('insufficient_data'):
            um = fe['up_market']
            sig = '***' if um['proxy_pvalue'] < 0.01 else '**' if um['proxy_pvalue'] < 0.05 else '*' if um['proxy_pvalue'] < 0.1 else ''
            lines.append(f"  UP:   coef={um['proxy_coef']:+.4f} (p={um['proxy_pvalue']:.4f}) {sig}")

        if 'comparison' in fe:
            comp = fe['comparison']
            lines.append(f"\n  Difference: {comp['difference']:+.4f} (p={comp['p_value']:.4f})")
            lines.append(f"  Effect stronger in DOWN markets: {comp['stronger_in_down']}")

        if 'asymmetric_model' in fe and 'base_effect' in fe['asymmetric_model']:
            am = fe['asymmetric_model']
            lines.append(f"\n  Asymmetric model:")
            lines.append(f"    Base effect (UP): {am['base_effect']:+.4f} (p={am['base_pvalue']:.4f})")
            lines.append(f"    Down shift: {am['down_shift']:+.4f} (p={am['down_shift_pvalue']:.4f})")
            lines.append(f"    Total effect in DOWN: {am['total_effect_down']:+.4f}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_gray_demand_analysis()
