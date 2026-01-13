"""Criminal Infrastructure Event Study.

Tests the hypothesis that criminal utility provides BTC's fundamental value
by examining price reactions to criminal infrastructure disruptions.

Key insight: If criminal utility is the value floor, destroying criminal
infrastructure should cause negative abnormal returns.

Natural experiments:
- Silk Road shutdown (Oct 2013)
- AlphaBay/Hansa shutdown (July 2017)
- Hydra seizure (April 2022)
- Tornado Cash sanctions (Aug 2022)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from btc_analysis.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class CriminalEvent:
    """A criminal infrastructure disruption event."""
    name: str
    date: str  # YYYY-MM-DD
    description: str
    magnitude: str  # 'major', 'medium', 'minor'
    btc_specific: bool  # Was this BTC-focused or broader crypto?


# Major criminal infrastructure disruption events
CRIMINAL_EVENTS = [
    CriminalEvent(
        name="silk_road_shutdown",
        date="2013-10-02",
        description="FBI seized Silk Road, arrested Ross Ulbricht, seized 144K BTC",
        magnitude="major",
        btc_specific=True,
    ),
    CriminalEvent(
        name="silk_road_2_shutdown",
        date="2014-11-06",
        description="Operation Onymous shut down Silk Road 2.0 and other markets",
        magnitude="medium",
        btc_specific=True,
    ),
    CriminalEvent(
        name="alphabay_hansa_shutdown",
        date="2017-07-20",
        description="Operation Bayonet: AlphaBay and Hansa coordinated takedown",
        magnitude="major",
        btc_specific=True,
    ),
    CriminalEvent(
        name="hydra_seizure",
        date="2022-04-05",
        description="German police seized Hydra servers, largest darknet market ($5.2B)",
        magnitude="major",
        btc_specific=True,  # Hydra was primarily BTC
    ),
    CriminalEvent(
        name="tornado_cash_sanctions",
        date="2022-08-08",
        description="OFAC sanctioned Tornado Cash mixer, first smart contract sanction",
        magnitude="major",
        btc_specific=False,  # ETH-based mixer
    ),
    CriminalEvent(
        name="btc_e_shutdown",
        date="2017-07-25",
        description="BTC-e exchange seized, accused of $4B money laundering",
        magnitude="medium",
        btc_specific=True,
    ),
    CriminalEvent(
        name="chipmixer_seizure",
        date="2023-03-15",
        description="ChipMixer seized, processed $3B including ransomware",
        magnitude="medium",
        btc_specific=True,
    ),
    CriminalEvent(
        name="garantex_sanctions",
        date="2022-04-05",
        description="OFAC sanctioned Garantex exchange for ransomware ties",
        magnitude="medium",
        btc_specific=False,
    ),
]


def get_btc_data(start_date: str = "2010-01-01") -> pd.DataFrame:
    """
    Fetch BTC price data from multiple sources.

    Uses CryptoCompare for early data (2010-2014), yfinance for later.
    """
    from pathlib import Path
    import yfinance as yf

    dfs = []

    # Try CryptoCompare for early data
    crypto_compare_path = Path(__file__).parent.parent.parent.parent / 'data' / 'raw' / 'btc_prices_cryptocompare.csv'
    if crypto_compare_path.exists():
        cc_df = pd.read_csv(crypto_compare_path, parse_dates=['date'])
        cc_df = cc_df[cc_df['close'] > 0]  # Filter zero prices
        cc_df = cc_df.set_index('date')
        cc_df = cc_df.rename(columns={'close': 'Close', 'volumeto': 'Volume'})
        cc_df['return'] = cc_df['Close'].pct_change()
        cc_df['log_return'] = np.log(cc_df['Close'] / cc_df['Close'].shift(1))
        dfs.append(cc_df[['Close', 'Volume', 'return', 'log_return']])
        logger.info(f"Loaded CryptoCompare data: {cc_df.index.min()} to {cc_df.index.max()}")

    # Get yfinance data for later period
    try:
        yf_start = "2014-09-01" if dfs else start_date
        btc = yf.download('BTC-USD', start=yf_start, progress=False)
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = [c[0] for c in btc.columns]

        btc['return'] = btc['Close'].pct_change()
        btc['log_return'] = np.log(btc['Close'] / btc['Close'].shift(1))
        dfs.append(btc[['Close', 'Volume', 'return', 'log_return']])
        logger.info(f"Loaded yfinance data: {btc.index.min()} to {btc.index.max()}")
    except Exception as e:
        logger.warning(f"Could not load yfinance data: {e}")

    if not dfs:
        raise ValueError("No BTC data available")

    # Combine and dedupe (prefer later data source)
    if len(dfs) > 1:
        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
    else:
        combined = dfs[0]

    # Recalculate returns after combining
    combined['return'] = combined['Close'].pct_change()
    combined['log_return'] = np.log(combined['Close'] / combined['Close'].shift(1))

    return combined


def estimate_market_model(
    returns: pd.Series,
    estimation_window: int = 120,
) -> Dict[str, float]:
    """
    Estimate market model parameters for expected return calculation.

    For crypto, we use a simple constant mean model since there's no
    clear market portfolio. Alternative: use total crypto market cap.

    Returns mean and std of returns in estimation window.
    """
    clean_returns = returns.dropna()

    if len(clean_returns) < estimation_window:
        return {'mean': np.nan, 'std': np.nan, 'n': len(clean_returns)}

    return {
        'mean': clean_returns.mean(),
        'std': clean_returns.std(),
        'n': len(clean_returns),
    }


def calculate_abnormal_returns(
    btc: pd.DataFrame,
    event_date: str,
    estimation_window: int = 120,  # ~6 months of trading days
    event_window: tuple = (-5, 10),  # 5 days before to 10 days after
) -> Dict[str, Any]:
    """
    Calculate abnormal returns around an event using market model.

    Abnormal Return = Actual Return - Expected Return
    Expected Return = Mean return from estimation window
    """
    event_dt = pd.Timestamp(event_date)

    # Find event date in data (or closest trading day)
    if event_dt not in btc.index:
        # Find next trading day
        future_dates = btc.index[btc.index > event_dt]
        if len(future_dates) == 0:
            return {'error': f'Event date {event_date} after data range'}
        event_dt = future_dates[0]

    event_idx = btc.index.get_loc(event_dt)

    # Estimation window: ends 10 days before event
    est_end = event_idx - 10
    est_start = est_end - estimation_window

    if est_start < 0:
        return {'error': f'Insufficient data before event for estimation window'}

    # Get estimation window returns
    est_returns = btc['return'].iloc[est_start:est_end]
    model_params = estimate_market_model(est_returns, estimation_window)

    if np.isnan(model_params['mean']):
        return {'error': 'Could not estimate market model'}

    expected_return = model_params['mean']
    std_return = model_params['std']

    # Event window
    ev_start = event_idx + event_window[0]
    ev_end = event_idx + event_window[1] + 1  # +1 for inclusive

    if ev_start < 0 or ev_end > len(btc):
        return {'error': 'Event window exceeds data range'}

    event_data = btc.iloc[ev_start:ev_end].copy()
    event_data['day'] = range(event_window[0], event_window[1] + 1)
    event_data['expected_return'] = expected_return
    event_data['abnormal_return'] = event_data['return'] - expected_return
    event_data['standardized_ar'] = event_data['abnormal_return'] / std_return

    # Cumulative abnormal returns
    event_data['car'] = event_data['abnormal_return'].cumsum()

    # Key metrics
    results = {
        'event_date': event_date,
        'actual_event_date': str(event_dt.date()),
        'estimation_window': estimation_window,
        'expected_daily_return': expected_return,
        'return_std': std_return,
        'event_window': event_window,

        # Day 0 (event day) abnormal return
        'ar_day0': float(event_data.loc[event_data['day'] == 0, 'abnormal_return'].iloc[0])
                   if 0 in event_data['day'].values else np.nan,

        # Cumulative abnormal return over different windows
        'car_0_1': float(event_data.loc[event_data['day'].between(0, 1), 'abnormal_return'].sum()),
        'car_0_3': float(event_data.loc[event_data['day'].between(0, 3), 'abnormal_return'].sum()),
        'car_0_5': float(event_data.loc[event_data['day'].between(0, 5), 'abnormal_return'].sum()),
        'car_0_10': float(event_data.loc[event_data['day'].between(0, 10), 'abnormal_return'].sum()),
        'car_m5_5': float(event_data.loc[event_data['day'].between(-5, 5), 'abnormal_return'].sum()),

        # Price data
        'price_day_m1': float(btc['Close'].iloc[event_idx - 1]) if event_idx > 0 else np.nan,
        'price_day_0': float(btc['Close'].iloc[event_idx]),
        'price_day_5': float(btc['Close'].iloc[event_idx + 5]) if event_idx + 5 < len(btc) else np.nan,
        'price_day_10': float(btc['Close'].iloc[event_idx + 10]) if event_idx + 10 < len(btc) else np.nan,

        # Daily data for plotting
        'daily_data': event_data[['day', 'Close', 'return', 'abnormal_return', 'car']].to_dict('records'),
    }

    # Statistical significance tests
    # Test if CAR is significantly different from 0
    n_days = event_window[1] - event_window[0] + 1
    car_std = std_return * np.sqrt(n_days)

    for window_name, car_value in [('car_0_5', results['car_0_5']),
                                    ('car_0_10', results['car_0_10']),
                                    ('car_m5_5', results['car_m5_5'])]:
        n = int(window_name.split('_')[-1]) - int(window_name.split('_')[-2].replace('m', '-')) + 1
        window_std = std_return * np.sqrt(n)
        t_stat = car_value / window_std if window_std > 0 else np.nan
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=estimation_window - 1)) if not np.isnan(t_stat) else np.nan

        results[f'{window_name}_tstat'] = t_stat
        results[f'{window_name}_pvalue'] = p_value

    return results


def run_event_study(
    events: Optional[List[CriminalEvent]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run event study for all criminal infrastructure disruptions.

    Tests: Do criminal infrastructure shutdowns cause negative abnormal returns?
    """
    config = get_config()

    if output_dir is None:
        output_dir = config.paths.outputs_dir / "criminal_events"
    output_dir.mkdir(parents=True, exist_ok=True)

    if events is None:
        events = CRIMINAL_EVENTS

    logger.info(f"Fetching BTC price data...")
    btc = get_btc_data(start_date="2013-01-01")

    results = {
        'events': [],
        'summary': {},
    }

    # Run event study for each event
    for event in events:
        logger.info(f"Analyzing event: {event.name} ({event.date})")

        event_results = calculate_abnormal_returns(btc, event.date)

        if 'error' in event_results:
            logger.warning(f"Skipping {event.name}: {event_results['error']}")
            continue

        event_results['name'] = event.name
        event_results['description'] = event.description
        event_results['magnitude'] = event.magnitude
        event_results['btc_specific'] = event.btc_specific

        results['events'].append(event_results)

    # Summary statistics
    if results['events']:
        # Average CAR across all events
        cars_0_5 = [e['car_0_5'] for e in results['events'] if not np.isnan(e.get('car_0_5', np.nan))]
        cars_0_10 = [e['car_0_10'] for e in results['events'] if not np.isnan(e.get('car_0_10', np.nan))]

        results['summary'] = {
            'n_events': len(results['events']),
            'avg_car_0_5': np.mean(cars_0_5) if cars_0_5 else np.nan,
            'avg_car_0_10': np.mean(cars_0_10) if cars_0_10 else np.nan,
            'pct_negative_car': sum(1 for c in cars_0_5 if c < 0) / len(cars_0_5) if cars_0_5 else np.nan,
        }

        # Test if average CAR is significantly negative
        if len(cars_0_5) > 1:
            t_stat, p_value = stats.ttest_1samp(cars_0_5, 0)
            results['summary']['avg_car_0_5_tstat'] = t_stat
            results['summary']['avg_car_0_5_pvalue'] = p_value
            # One-tailed test (we predict negative)
            results['summary']['avg_car_0_5_pvalue_onetail'] = p_value / 2 if t_stat < 0 else 1 - p_value / 2

        # Separate analysis for major events only
        major_cars = [e['car_0_5'] for e in results['events']
                      if e['magnitude'] == 'major' and not np.isnan(e.get('car_0_5', np.nan))]
        if major_cars:
            results['summary']['avg_car_0_5_major'] = np.mean(major_cars)
            results['summary']['n_major_events'] = len(major_cars)

        # BTC-specific events only
        btc_cars = [e['car_0_5'] for e in results['events']
                    if e['btc_specific'] and not np.isnan(e.get('car_0_5', np.nan))]
        if btc_cars:
            results['summary']['avg_car_0_5_btc_specific'] = np.mean(btc_cars)
            results['summary']['n_btc_specific_events'] = len(btc_cars)

    # Determine interpretation
    avg_car = results['summary'].get('avg_car_0_5', np.nan)
    pct_neg = results['summary'].get('pct_negative_car', np.nan)

    if not np.isnan(avg_car):
        if avg_car < -0.05 and pct_neg > 0.6:
            results['summary']['interpretation'] = (
                f"Strong support for criminal utility hypothesis: "
                f"Average CAR of {avg_car:.1%} with {pct_neg:.0%} negative reactions. "
                "Criminal infrastructure shutdowns hurt BTC price."
            )
        elif avg_car < -0.02:
            results['summary']['interpretation'] = (
                f"Moderate support: Average CAR of {avg_car:.1%}. "
                "Some evidence criminal infrastructure matters to BTC value."
            )
        elif avg_car > 0.02:
            results['summary']['interpretation'] = (
                f"AGAINST hypothesis: Average CAR of {avg_car:+.1%}. "
                "BTC actually rises after criminal infrastructure shutdowns. "
                "Suggests 'legitimization' effect or criminal utility is not fundamental."
            )
        else:
            results['summary']['interpretation'] = (
                f"Inconclusive: Average CAR of {avg_car:+.1%}. "
                "No clear relationship between criminal infrastructure and BTC price."
            )

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
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    # Remove daily_data for JSON (too verbose)
    results_clean = convert_types(results)
    for event in results_clean['events']:
        del event['daily_data']

    with open(output_dir / 'event_study_results.json', 'w') as f:
        json.dump(results_clean, f, indent=2)

    # Generate report
    report = _generate_report(results)
    with open(output_dir / 'criminal_events_analysis.txt', 'w') as f:
        f.write(report)

    logger.info(f"Results saved to {output_dir}")

    return results


def _generate_report(results: Dict[str, Any]) -> str:
    """Generate text report of event study."""
    lines = []
    lines.append("=" * 80)
    lines.append("CRIMINAL INFRASTRUCTURE EVENT STUDY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Hypothesis: Criminal utility provides BTC's fundamental value floor")
    lines.append("Test: Do criminal infrastructure shutdowns cause negative abnormal returns?")
    lines.append("")
    lines.append("Methodology:")
    lines.append("- Event study with 120-day estimation window")
    lines.append("- Abnormal Return = Actual Return - Expected Return (mean model)")
    lines.append("- CAR = Cumulative Abnormal Return over event window")
    lines.append("")

    lines.append("-" * 80)
    lines.append("EVENT RESULTS")
    lines.append("-" * 80)
    lines.append("")

    header = f"{'Event':<30} {'Date':<12} {'CAR(0,5)':<12} {'p-value':<10} {'Mag':<8}"
    lines.append(header)
    lines.append("-" * 75)

    for event in results.get('events', []):
        car = event.get('car_0_5', np.nan)
        pval = event.get('car_0_5_pvalue', np.nan)
        mag = event.get('magnitude', '')

        car_str = f"{car:+.2%}" if not np.isnan(car) else "N/A"
        pval_str = f"{pval:.3f}" if not np.isnan(pval) else "N/A"

        # Significance stars
        if not np.isnan(pval):
            if pval < 0.01:
                pval_str += " ***"
            elif pval < 0.05:
                pval_str += " **"
            elif pval < 0.10:
                pval_str += " *"

        lines.append(f"{event['name']:<30} {event.get('actual_event_date', ''):<12} {car_str:<12} {pval_str:<10} {mag:<8}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 80)

    summary = results.get('summary', {})
    lines.append(f"Number of events: {summary.get('n_events', 'N/A')}")

    avg_car = summary.get('avg_car_0_5', np.nan)
    if not np.isnan(avg_car):
        lines.append(f"Average CAR(0,5): {avg_car:+.2%}")

    pct_neg = summary.get('pct_negative_car', np.nan)
    if not np.isnan(pct_neg):
        lines.append(f"Percent negative: {pct_neg:.0%}")

    t_stat = summary.get('avg_car_0_5_tstat', np.nan)
    p_val = summary.get('avg_car_0_5_pvalue', np.nan)
    p_val_1t = summary.get('avg_car_0_5_pvalue_onetail', np.nan)
    if not np.isnan(t_stat):
        lines.append(f"t-statistic: {t_stat:.2f}")
        lines.append(f"p-value (two-tailed): {p_val:.3f}")
        lines.append(f"p-value (one-tailed, H1: negative): {p_val_1t:.3f}")

    avg_major = summary.get('avg_car_0_5_major', np.nan)
    if not np.isnan(avg_major):
        lines.append(f"\nMajor events only ({summary.get('n_major_events', 0)}): {avg_major:+.2%}")

    avg_btc = summary.get('avg_car_0_5_btc_specific', np.nan)
    if not np.isnan(avg_btc):
        lines.append(f"BTC-specific events ({summary.get('n_btc_specific_events', 0)}): {avg_btc:+.2%}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("INTERPRETATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append(summary.get('interpretation', 'No interpretation available'))
    lines.append("")

    lines.append("=" * 80)
    lines.append("WHAT THIS MEANS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("If criminal utility provides BTC's fundamental value:")
    lines.append("- Shutting down criminal infrastructure should hurt BTC")
    lines.append("- We expect NEGATIVE abnormal returns after shutdowns")
    lines.append("")
    lines.append("Alternative interpretations of POSITIVE returns:")
    lines.append("- 'Legitimization' effect: Less crime = more mainstream adoption")
    lines.append("- Criminal utility is marginal, not fundamental")
    lines.append("- Criminals quickly substitute to other infrastructure")
    lines.append("- Market already priced in shutdown risk")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_event_study()
