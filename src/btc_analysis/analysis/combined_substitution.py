"""Combined Law Enforcement Cash Substitution Analysis.

Extends the original CBP analysis by combining:
- CBP: Border currency seizures (international trafficking)
- DOJ/CATS: Domestic cash seizures (DEA, FBI, ATF, USPS, Secret Service)

Runs the FULL analytical suite from cash_substitution.py:
- Ratio trends
- Returns regression with volume interaction
- Extended lags (1-6) with reversal test
- Reverse causality
- Volatility prediction
- Floor effect (up vs down markets)
- Structural breaks (MicroStrategy, ETF filing)
- Falsification tests
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.stattools import durbin_watson

from btc_analysis.config import get_config

logger = logging.getLogger(__name__)


def _zscore(series: pd.Series) -> pd.Series:
    """Calculate z-score, handling NaN values."""
    return (series - series.mean()) / series.std()


def load_combined_data() -> pd.DataFrame:
    """Load and merge all data sources."""
    config = get_config()

    # Load combined seizure data
    seizure_path = config.paths.processed_dir / 'combined_criminal_seizures.csv'
    seizures = pd.read_csv(seizure_path, parse_dates=['year_month'])

    # Load BTC price data
    import yfinance as yf

    btc = yf.download('BTC-USD', start='2019-01-01', progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = [c[0] for c in btc.columns]

    btc['return'] = btc['Close'].pct_change()
    btc_monthly = btc.resample('MS').agg({
        'Close': 'last',
        'return': lambda x: (1 + x).prod() - 1,
        'Volume': 'sum',
    }).reset_index()
    btc_monthly = btc_monthly.rename(columns={
        'Date': 'year_month',
        'return': 'btc_return',
        'Close': 'btc_price',
        'Volume': 'btc_volume'
    })

    # Load S&P 500
    sp500 = yf.download('^GSPC', start='2019-01-01', progress=False)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = [c[0] for c in sp500.columns]
    sp500['return'] = sp500['Close'].pct_change()
    sp500_monthly = sp500.resample('MS').agg({
        'Close': 'last',
        'return': lambda x: (1 + x).prod() - 1,
    }).reset_index()
    sp500_monthly = sp500_monthly.rename(columns={
        'Date': 'year_month',
        'return': 'sp500_return',
        'Close': 'sp500_price'
    })

    # Merge
    df = pd.merge(seizures, btc_monthly, on='year_month', how='inner')
    df = pd.merge(df, sp500_monthly, on='year_month', how='left')

    # Load stablecoin data (if available)
    stablecoin_path = config.paths.processed_dir / 'stablecoin_monthly.csv'
    if stablecoin_path.exists():
        stablecoin = pd.read_csv(stablecoin_path, parse_dates=['year_month'])
        df = pd.merge(df, stablecoin[['year_month', 'usdt_supply', 'tron_stablecoin_supply',
                                       'total_stablecoin_supply']], on='year_month', how='left')

    # Load privacy tools data (if available)
    privacy_path = config.paths.processed_dir / 'privacy_tools_monthly.csv'
    if privacy_path.exists():
        privacy = pd.read_csv(privacy_path, parse_dates=['year_month'])
        df = pd.merge(df, privacy[['year_month', 'tornado_tvl_usd_avg',
                                    'post_ofac_sanctions']], on='year_month', how='left')

    return df


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all metrics with proper transformations."""
    df = df.copy()

    # Z-score the seizure values
    df['cbp_z'] = _zscore(df['cbp_value'])
    df['doj_z'] = _zscore(df['doj_value'])
    df['total_z'] = _zscore(df['total_value'])

    # Substitution indices (% change, inverted)
    df['cbp_pct_change'] = df['cbp_value'].pct_change()
    df['doj_pct_change'] = df['doj_value'].pct_change()
    df['total_pct_change'] = df['total_value'].pct_change()

    # Clean infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    # Z-scored substitution (negative of z-scored seizures - high value = low seizures)
    df['subst_cbp_z'] = -_zscore(df['cbp_value'].fillna(df['cbp_value'].mean()))
    df['subst_doj_z'] = -_zscore(df['doj_value'].fillna(df['doj_value'].mean()))
    df['subst_total_z'] = -_zscore(df['total_value'].fillna(df['total_value'].mean()))

    # Volume z-score
    if 'btc_volume' in df.columns:
        df['volume_z'] = _zscore(df['btc_volume'])
        df['high_volume'] = (df['volume_z'] > 0).astype(int)

    # Create interaction terms
    for subst_col in ['subst_cbp_z', 'subst_doj_z', 'subst_total_z']:
        df[f'{subst_col}_x_vol'] = df[subst_col] * df['volume_z']

    # Create lagged variables (1-6 months)
    for subst_col in ['subst_cbp_z', 'subst_doj_z', 'subst_total_z']:
        for lag in range(1, 7):
            df[f'{subst_col}_lag{lag}'] = df[subst_col].shift(lag)
            df[f'{subst_col}_x_vol_lag{lag}'] = df[f'{subst_col}_lag{lag}'] * df['volume_z']

    # Regime indicators
    df['post_microstrategy'] = (df['year_month'] >= '2020-08-01').astype(int)
    df['post_etf_filing'] = (df['year_month'] >= '2023-06-01').astype(int)
    df['down_market'] = (df['sp500_return'] < 0).astype(int)
    df['up_market'] = 1 - df['down_market']

    # Z-score the new control variables
    if 'usdt_supply' in df.columns:
        df['log_usdt'] = np.log(df['usdt_supply'].clip(lower=1))
        df['usdt_z'] = _zscore(df['log_usdt'].fillna(df['log_usdt'].mean()))

    if 'tron_stablecoin_supply' in df.columns:
        df['log_tron'] = np.log(df['tron_stablecoin_supply'].clip(lower=1))
        df['tron_z'] = _zscore(df['log_tron'].fillna(df['log_tron'].mean()))

    if 'tornado_tvl_usd_avg' in df.columns:
        df['log_tornado'] = np.log(df['tornado_tvl_usd_avg'].clip(lower=1))
        df['tornado_z'] = _zscore(df['log_tornado'].fillna(df['log_tornado'].mean()))

    return df


def run_full_analysis(
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the complete analytical suite on combined data."""
    config = get_config()

    if output_dir is None:
        output_dir = config.paths.outputs_dir / "combined_substitution"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading combined data...")
    df = load_combined_data()

    logger.info("Calculating metrics...")
    df = calculate_metrics(df)

    results = {
        'data_summary': {
            'n_obs': len(df),
            'period': f"{df['year_month'].min()} to {df['year_month'].max()}",
            'avg_cbp': df['cbp_value'].mean(),
            'avg_doj': df['doj_value'].mean(),
            'avg_total': df['total_value'].mean(),
            'cbp_doj_corr': df['cbp_value'].corr(df['doj_value']),
        }
    }

    # Run for each data source
    for source, subst_col in [
        ('CBP (Border)', 'subst_cbp_z'),
        ('DOJ (Domestic)', 'subst_doj_z'),
        ('Combined (All LE)', 'subst_total_z'),
    ]:
        logger.info(f"Analyzing: {source}")
        source_results = {}

        # 1. Basic regression with interaction
        source_results['basic_regression'] = _run_basic_regression(df, subst_col)

        # 2. Extended lag analysis (1-6)
        source_results['extended_lags'] = _run_extended_lags(df, subst_col)

        # 3. Floor effect (up vs down markets)
        source_results['floor_effect'] = _run_floor_effect(df, subst_col)

        # 4. Structural break analysis
        source_results['structural_breaks'] = _run_structural_breaks(df, subst_col)

        # 5. Reverse causality
        source_results['reverse_causality'] = _run_reverse_causality(df, subst_col)

        # 6. Volatility prediction
        source_results['volatility'] = _run_volatility_test(df, subst_col)

        # 7. Controlled regression (with stablecoin/privacy tool controls)
        source_results['controlled'] = _run_controlled_regression(df, subst_col)

        results[source] = source_results

    # Generate report
    report = _generate_report(results)
    with open(output_dir / 'full_analysis.txt', 'w') as f:
        f.write(report)

    # Save data
    df.to_csv(output_dir / 'analysis_data.csv', index=False)

    logger.info(f"Results saved to {output_dir}")

    return results


def _run_basic_regression(df: pd.DataFrame, subst_col: str) -> Dict[str, Any]:
    """Run basic regression with volume interaction."""
    results = {}

    interact_col = f'{subst_col}_x_vol'
    required = ['btc_return', 'sp500_return', subst_col, 'volume_z', interact_col]
    df_clean = df[required].replace([np.inf, -np.inf], np.nan).dropna()

    if len(df_clean) < 20:
        return {'error': 'Insufficient data'}

    # Contemporaneous
    X = sm.add_constant(df_clean[['sp500_return', subst_col, 'volume_z', interact_col]])
    y = df_clean['btc_return']
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

    results['contemporaneous'] = {
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficients': {
            name: {'coef': float(model.params[name]), 'pvalue': float(model.pvalues[name])}
            for name in model.params.index
        }
    }

    # Lagged (lag 1)
    lag_col = f'{subst_col}_lag1'
    interact_lag = f'{subst_col}_x_vol_lag1'

    if lag_col in df.columns and interact_lag in df.columns:
        df_lag = df[['btc_return', 'sp500_return', lag_col, 'volume_z', interact_lag]].replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_lag) >= 15:
            X_lag = sm.add_constant(df_lag[['sp500_return', lag_col, 'volume_z', interact_lag]])
            y_lag = df_lag['btc_return']
            model_lag = sm.OLS(y_lag, X_lag).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

            results['lag_1'] = {
                'n_obs': int(model_lag.nobs),
                'r_squared': float(model_lag.rsquared),
                'interaction_coef': float(model_lag.params.get(interact_lag, np.nan)),
                'interaction_pvalue': float(model_lag.pvalues.get(interact_lag, np.nan)),
            }

    return results


def _run_extended_lags(df: pd.DataFrame, subst_col: str) -> Dict[str, Any]:
    """Test lags 1-6 for reversal pattern."""
    results = {'lag_coefficients': []}

    for lag in range(1, 7):
        lag_col = f'{subst_col}_lag{lag}'
        interact_col = f'{subst_col}_x_vol_lag{lag}'

        if lag_col not in df.columns or interact_col not in df.columns:
            continue

        required = ['btc_return', 'sp500_return', lag_col, 'volume_z', interact_col]
        df_clean = df[required].replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_clean) < 15:
            continue

        X = sm.add_constant(df_clean[['sp500_return', lag_col, 'volume_z', interact_col]])
        y = df_clean['btc_return']

        try:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
            results['lag_coefficients'].append({
                'lag': lag,
                'n_obs': int(model.nobs),
                'interaction_coef': float(model.params.get(interact_col, np.nan)),
                'interaction_pvalue': float(model.pvalues.get(interact_col, np.nan)),
            })
        except:
            pass

    # Check for reversal
    if len(results['lag_coefficients']) >= 4:
        early = [r for r in results['lag_coefficients'] if r['lag'] <= 2]
        late = [r for r in results['lag_coefficients'] if r['lag'] >= 4]

        if early and late:
            avg_early = np.mean([r['interaction_coef'] for r in early])
            avg_late = np.mean([r['interaction_coef'] for r in late])

            results['reversal_test'] = {
                'avg_early': float(avg_early),
                'avg_late': float(avg_late),
                'sign_reversal': (avg_early > 0 and avg_late < 0) or (avg_early < 0 and avg_late > 0),
            }

    return results


def _run_floor_effect(df: pd.DataFrame, subst_col: str) -> Dict[str, Any]:
    """Test if effect is stronger in down markets."""
    results = {}

    lag_col = f'{subst_col}_lag1'
    interact_col = f'{subst_col}_x_vol_lag1'

    required = ['btc_return', 'sp500_return', lag_col, 'volume_z', interact_col, 'down_market']
    df_clean = df[required].replace([np.inf, -np.inf], np.nan).dropna()

    if len(df_clean) < 20:
        return {'error': 'Insufficient data'}

    # Split sample
    down = df_clean[df_clean['down_market'] == 1]
    up = df_clean[df_clean['down_market'] == 0]

    results['sample_sizes'] = {'down': len(down), 'up': len(up)}

    for regime_name, regime_df in [('down_market', down), ('up_market', up)]:
        if len(regime_df) < 10:
            results[regime_name] = {'insufficient_data': True}
            continue

        X = sm.add_constant(regime_df[['sp500_return', lag_col, 'volume_z', interact_col]])
        y = regime_df['btc_return']

        try:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 2})
            results[regime_name] = {
                'n_obs': int(model.nobs),
                'interaction_coef': float(model.params.get(interact_col, np.nan)),
                'interaction_pvalue': float(model.pvalues.get(interact_col, np.nan)),
            }
        except Exception as e:
            results[regime_name] = {'error': str(e)}

    # Compare
    if 'down_market' in results and 'up_market' in results:
        if not results['down_market'].get('insufficient_data') and not results['up_market'].get('insufficient_data'):
            coef_down = results['down_market'].get('interaction_coef', 0)
            coef_up = results['up_market'].get('interaction_coef', 0)
            results['comparison'] = {
                'down_coef': coef_down,
                'up_coef': coef_up,
                'stronger_in_down': abs(coef_down) > abs(coef_up),
            }

    return results


def _run_structural_breaks(df: pd.DataFrame, subst_col: str) -> Dict[str, Any]:
    """Test for structural breaks at MicroStrategy and ETF filing."""
    results = {}

    lag_col = f'{subst_col}_lag1'
    interact_col = f'{subst_col}_x_vol_lag1'

    breaks = {
        'microstrategy': pd.Timestamp('2020-08-01'),
        'etf_filing': pd.Timestamp('2023-06-01'),
    }

    required = ['btc_return', 'sp500_return', lag_col, 'volume_z', interact_col, 'year_month']
    df_clean = df[required].replace([np.inf, -np.inf], np.nan).dropna()

    for break_name, break_date in breaks.items():
        pre = df_clean[df_clean['year_month'] < break_date]
        post = df_clean[df_clean['year_month'] >= break_date]

        break_results = {'break_date': str(break_date.date())}

        for period_name, period_df in [('pre', pre), ('post', post)]:
            if len(period_df) < 10:
                break_results[period_name] = {'insufficient_data': True, 'n': len(period_df)}
                continue

            X = sm.add_constant(period_df[['sp500_return', lag_col, 'volume_z', interact_col]])
            y = period_df['btc_return']

            try:
                model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 2})
                break_results[period_name] = {
                    'n_obs': int(model.nobs),
                    'interaction_coef': float(model.params.get(interact_col, np.nan)),
                    'interaction_pvalue': float(model.pvalues.get(interact_col, np.nan)),
                }
            except:
                break_results[period_name] = {'error': 'Model failed'}

        results[break_name] = break_results

    return results


def _run_reverse_causality(df: pd.DataFrame, subst_col: str) -> Dict[str, Any]:
    """Test if BTC returns predict future substitution."""
    results = {}

    # Create leads
    df = df.copy()
    for lead in [1, 2, 3]:
        df[f'{subst_col}_lead{lead}'] = df[subst_col].shift(-lead)

    for lead in [1, 2, 3]:
        lead_col = f'{subst_col}_lead{lead}'
        required = ['btc_return', 'sp500_return', lead_col]
        df_clean = df[required].replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_clean) < 15:
            continue

        X = sm.add_constant(df_clean[['btc_return', 'sp500_return']])
        y = df_clean[lead_col]

        try:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 2})
            results[f'lead_{lead}'] = {
                'btc_coef': float(model.params.get('btc_return', np.nan)),
                'btc_pvalue': float(model.pvalues.get('btc_return', np.nan)),
            }
        except:
            pass

    return results


def _run_volatility_test(df: pd.DataFrame, subst_col: str) -> Dict[str, Any]:
    """Test if substitution predicts volatility."""
    results = {}

    df = df.copy()
    df['btc_volatility'] = df['btc_return'].abs()

    interact_col = f'{subst_col}_x_vol_lag1'
    lag_col = f'{subst_col}_lag1'

    required = ['btc_volatility', lag_col, 'volume_z', interact_col]
    df_clean = df[required].replace([np.inf, -np.inf], np.nan).dropna()

    if len(df_clean) < 15:
        return {'error': 'Insufficient data'}

    X = sm.add_constant(df_clean[[lag_col, 'volume_z', interact_col]])
    y = df_clean['btc_volatility']

    try:
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 2})
        results = {
            'n_obs': int(model.nobs),
            'interaction_coef': float(model.params.get(interact_col, np.nan)),
            'interaction_pvalue': float(model.pvalues.get(interact_col, np.nan)),
        }
    except Exception as e:
        results = {'error': str(e)}

    return results


def _run_controlled_regression(df: pd.DataFrame, subst_col: str) -> Dict[str, Any]:
    """Run regression with stablecoin and privacy tool controls."""
    results = {}

    interact_col = f'{subst_col}_x_vol'

    # Check what controls are available
    controls_available = []
    if 'usdt_z' in df.columns and df['usdt_z'].notna().sum() > 20:
        controls_available.append('usdt_z')
    if 'tron_z' in df.columns and df['tron_z'].notna().sum() > 20:
        controls_available.append('tron_z')
    if 'tornado_z' in df.columns and df['tornado_z'].notna().sum() > 20:
        controls_available.append('tornado_z')

    if not controls_available:
        return {'error': 'No control variables available'}

    # Model 1: Baseline (no controls)
    required_base = ['btc_return', 'sp500_return', subst_col, 'volume_z', interact_col]
    df_base = df[required_base].replace([np.inf, -np.inf], np.nan).dropna()

    if len(df_base) >= 20:
        X_base = sm.add_constant(df_base[['sp500_return', subst_col, 'volume_z', interact_col]])
        y_base = df_base['btc_return']
        model_base = sm.OLS(y_base, X_base).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

        results['baseline'] = {
            'n_obs': int(model_base.nobs),
            'r_squared': float(model_base.rsquared),
            'interaction_coef': float(model_base.params.get(interact_col, np.nan)),
            'interaction_pvalue': float(model_base.pvalues.get(interact_col, np.nan)),
        }

    # Model 2: With stablecoin controls
    if 'usdt_z' in controls_available:
        required_stable = required_base + ['usdt_z']
        df_stable = df[required_stable].replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_stable) >= 20:
            X_stable = sm.add_constant(df_stable[['sp500_return', subst_col, 'volume_z', interact_col, 'usdt_z']])
            y_stable = df_stable['btc_return']
            model_stable = sm.OLS(y_stable, X_stable).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

            results['with_stablecoin'] = {
                'n_obs': int(model_stable.nobs),
                'r_squared': float(model_stable.rsquared),
                'interaction_coef': float(model_stable.params.get(interact_col, np.nan)),
                'interaction_pvalue': float(model_stable.pvalues.get(interact_col, np.nan)),
                'stablecoin_coef': float(model_stable.params.get('usdt_z', np.nan)),
                'stablecoin_pvalue': float(model_stable.pvalues.get('usdt_z', np.nan)),
            }

    # Model 3: With Tornado Cash control
    if 'tornado_z' in controls_available:
        required_tornado = required_base + ['tornado_z']
        if 'usdt_z' in controls_available:
            required_tornado.append('usdt_z')

        df_tornado = df[required_tornado].replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_tornado) >= 20:
            control_vars = ['sp500_return', subst_col, 'volume_z', interact_col, 'tornado_z']
            if 'usdt_z' in required_tornado:
                control_vars.append('usdt_z')

            X_tornado = sm.add_constant(df_tornado[control_vars])
            y_tornado = df_tornado['btc_return']
            model_tornado = sm.OLS(y_tornado, X_tornado).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

            results['with_tornado'] = {
                'n_obs': int(model_tornado.nobs),
                'r_squared': float(model_tornado.rsquared),
                'interaction_coef': float(model_tornado.params.get(interact_col, np.nan)),
                'interaction_pvalue': float(model_tornado.pvalues.get(interact_col, np.nan)),
                'tornado_coef': float(model_tornado.params.get('tornado_z', np.nan)),
                'tornado_pvalue': float(model_tornado.pvalues.get('tornado_z', np.nan)),
            }

    # Model 4: With Tron stablecoin (proxy for illicit)
    if 'tron_z' in controls_available:
        required_tron = required_base + ['tron_z']
        df_tron = df[required_tron].replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_tron) >= 20:
            X_tron = sm.add_constant(df_tron[['sp500_return', subst_col, 'volume_z', interact_col, 'tron_z']])
            y_tron = df_tron['btc_return']
            model_tron = sm.OLS(y_tron, X_tron).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

            results['with_tron'] = {
                'n_obs': int(model_tron.nobs),
                'r_squared': float(model_tron.rsquared),
                'interaction_coef': float(model_tron.params.get(interact_col, np.nan)),
                'interaction_pvalue': float(model_tron.pvalues.get(interact_col, np.nan)),
                'tron_coef': float(model_tron.params.get('tron_z', np.nan)),
                'tron_pvalue': float(model_tron.pvalues.get('tron_z', np.nan)),
            }

    return results


def _get_sig(pvalue: float) -> str:
    """Return significance stars for p-value."""
    if pvalue < 0.01:
        return "***"
    elif pvalue < 0.05:
        return "**"
    elif pvalue < 0.10:
        return "*"
    return ""


def _generate_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive report."""
    lines = []
    lines.append("=" * 80)
    lines.append("COMBINED LAW ENFORCEMENT CASH SUBSTITUTION ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # Data summary
    summary = results['data_summary']
    lines.append("DATA SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Observations: {summary['n_obs']}")
    lines.append(f"Period: {summary['period']}")
    lines.append(f"Avg CBP (border): ${summary['avg_cbp']/1e6:.1f}M/month")
    lines.append(f"Avg DOJ (domestic): ${summary['avg_doj']/1e6:.1f}M/month")
    lines.append(f"Avg Total: ${summary['avg_total']/1e6:.1f}M/month")
    lines.append(f"CBP-DOJ correlation: {summary['cbp_doj_corr']:.3f}")
    lines.append("")

    # Results for each source
    for source in ['CBP (Border)', 'DOJ (Domestic)', 'Combined (All LE)']:
        if source not in results:
            continue

        lines.append("=" * 80)
        lines.append(f"{source}")
        lines.append("=" * 80)
        lines.append("")

        src = results[source]

        # Basic regression
        if 'basic_regression' in src and 'contemporaneous' in src['basic_regression']:
            reg = src['basic_regression']['contemporaneous']
            lines.append("CONTEMPORANEOUS REGRESSION (with interaction)")
            lines.append("-" * 40)
            for var, vals in reg.get('coefficients', {}).items():
                sig = "***" if vals['pvalue'] < 0.01 else "**" if vals['pvalue'] < 0.05 else "*" if vals['pvalue'] < 0.1 else ""
                lines.append(f"  {var}: {vals['coef']:+.4f} (p={vals['pvalue']:.4f}) {sig}")
            lines.append("")

        # Lag 1
        if 'basic_regression' in src and 'lag_1' in src['basic_regression']:
            lag1 = src['basic_regression']['lag_1']
            sig = "***" if lag1['interaction_pvalue'] < 0.01 else "**" if lag1['interaction_pvalue'] < 0.05 else "*" if lag1['interaction_pvalue'] < 0.1 else ""
            lines.append(f"LAG 1 INTERACTION: {lag1['interaction_coef']:+.4f} (p={lag1['interaction_pvalue']:.4f}) {sig}")
            lines.append("")

        # Extended lags
        if 'extended_lags' in src and 'lag_coefficients' in src['extended_lags']:
            lines.append("EXTENDED LAG ANALYSIS")
            lines.append("-" * 40)
            for lag_data in src['extended_lags']['lag_coefficients']:
                sig = "***" if lag_data['interaction_pvalue'] < 0.01 else "**" if lag_data['interaction_pvalue'] < 0.05 else "*" if lag_data['interaction_pvalue'] < 0.1 else ""
                lines.append(f"  Lag {lag_data['lag']}: {lag_data['interaction_coef']:+.4f} (p={lag_data['interaction_pvalue']:.4f}) {sig}")

            if 'reversal_test' in src['extended_lags']:
                rev = src['extended_lags']['reversal_test']
                lines.append(f"\n  Reversal test: early avg={rev['avg_early']:+.4f}, late avg={rev['avg_late']:+.4f}")
                lines.append(f"  Sign reversal: {rev['sign_reversal']}")
            lines.append("")

        # Floor effect
        if 'floor_effect' in src:
            floor = src['floor_effect']
            lines.append("FLOOR EFFECT (Up vs Down Markets)")
            lines.append("-" * 40)
            if 'down_market' in floor and not floor['down_market'].get('insufficient_data'):
                dm = floor['down_market']
                sig = "***" if dm['interaction_pvalue'] < 0.01 else "**" if dm['interaction_pvalue'] < 0.05 else "*" if dm['interaction_pvalue'] < 0.1 else ""
                lines.append(f"  DOWN markets (n={dm['n_obs']}): {dm['interaction_coef']:+.4f} (p={dm['interaction_pvalue']:.4f}) {sig}")
            if 'up_market' in floor and not floor['up_market'].get('insufficient_data'):
                um = floor['up_market']
                sig = "***" if um['interaction_pvalue'] < 0.01 else "**" if um['interaction_pvalue'] < 0.05 else "*" if um['interaction_pvalue'] < 0.1 else ""
                lines.append(f"  UP markets (n={um['n_obs']}): {um['interaction_coef']:+.4f} (p={um['interaction_pvalue']:.4f}) {sig}")
            if 'comparison' in floor:
                lines.append(f"  Stronger in down markets: {floor['comparison']['stronger_in_down']}")
            lines.append("")

        # Structural breaks
        if 'structural_breaks' in src:
            lines.append("STRUCTURAL BREAKS")
            lines.append("-" * 40)
            for break_name, break_data in src['structural_breaks'].items():
                lines.append(f"\n  {break_name.upper()} ({break_data.get('break_date', 'N/A')}):")
                for period in ['pre', 'post']:
                    if period in break_data and not break_data[period].get('insufficient_data'):
                        pd_data = break_data[period]
                        sig = "***" if pd_data['interaction_pvalue'] < 0.01 else "**" if pd_data['interaction_pvalue'] < 0.05 else "*" if pd_data['interaction_pvalue'] < 0.1 else ""
                        lines.append(f"    {period}: {pd_data['interaction_coef']:+.4f} (p={pd_data['interaction_pvalue']:.4f}) {sig}")
            lines.append("")

        # Reverse causality
        if 'reverse_causality' in src:
            lines.append("REVERSE CAUSALITY (BTC → Future Subst)")
            lines.append("-" * 40)
            for lead_name, lead_data in src['reverse_causality'].items():
                sig = "***" if lead_data['btc_pvalue'] < 0.01 else "**" if lead_data['btc_pvalue'] < 0.05 else "*" if lead_data['btc_pvalue'] < 0.1 else ""
                lines.append(f"  {lead_name}: {lead_data['btc_coef']:+.4f} (p={lead_data['btc_pvalue']:.4f}) {sig}")
            lines.append("")

        # Volatility
        if 'volatility' in src and 'interaction_coef' in src['volatility']:
            vol = src['volatility']
            sig = "***" if vol['interaction_pvalue'] < 0.01 else "**" if vol['interaction_pvalue'] < 0.05 else "*" if vol['interaction_pvalue'] < 0.1 else ""
            lines.append(f"VOLATILITY PREDICTION: {vol['interaction_coef']:+.4f} (p={vol['interaction_pvalue']:.4f}) {sig}")
            lines.append("")

        # Controlled regression results
        controlled = src.get('controlled', {})
        if 'error' not in controlled and controlled:
            lines.append("CONTROLLED REGRESSION (with crypto controls)")
            lines.append("-" * 40)

            if 'baseline' in controlled:
                b = controlled['baseline']
                sig = _get_sig(b.get('interaction_pvalue', 1))
                lines.append(f"  Baseline: coef={b['interaction_coef']:+.4f} (p={b.get('interaction_pvalue', np.nan):.4f}) {sig} [R²={b['r_squared']:.4f}]")

            if 'with_stablecoin' in controlled:
                s = controlled['with_stablecoin']
                sig = _get_sig(s.get('interaction_pvalue', 1))
                sig_stable = _get_sig(s.get('stablecoin_pvalue', 1))
                lines.append(f"  +Stablecoin: coef={s['interaction_coef']:+.4f} (p={s.get('interaction_pvalue', np.nan):.4f}) {sig} [R²={s['r_squared']:.4f}]")
                lines.append(f"    Stablecoin effect: {s.get('stablecoin_coef', np.nan):+.4f} (p={s.get('stablecoin_pvalue', np.nan):.4f}) {sig_stable}")

            if 'with_tornado' in controlled:
                t = controlled['with_tornado']
                sig = _get_sig(t.get('interaction_pvalue', 1))
                sig_tornado = _get_sig(t.get('tornado_pvalue', 1))
                lines.append(f"  +Tornado: coef={t['interaction_coef']:+.4f} (p={t.get('interaction_pvalue', np.nan):.4f}) {sig} [R²={t['r_squared']:.4f}]")
                lines.append(f"    Tornado effect: {t.get('tornado_coef', np.nan):+.4f} (p={t.get('tornado_pvalue', np.nan):.4f}) {sig_tornado}")

            if 'with_tron' in controlled:
                tr = controlled['with_tron']
                sig = _get_sig(tr.get('interaction_pvalue', 1))
                sig_tron = _get_sig(tr.get('tron_pvalue', 1))
                lines.append(f"  +Tron: coef={tr['interaction_coef']:+.4f} (p={tr.get('interaction_pvalue', np.nan):.4f}) {sig} [R²={tr['r_squared']:.4f}]")
                lines.append(f"    Tron effect: {tr.get('tron_coef', np.nan):+.4f} (p={tr.get('tron_pvalue', np.nan):.4f}) {sig_tron}")

            lines.append("")

    lines.append("=" * 80)
    lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.10")
    lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_full_analysis()
