"""Stablecoin Analysis - Criminal Demand Hypothesis.

Tests whether criminal enterprise activity predicts stablecoin usage,
using the same methodology as the BTC cash substitution analysis.

Key differences from BTC analysis:
- Stablecoins are pegged to $1, so price isn't meaningful
- Dependent variables: volume changes, peg deviation
- Hypothesis: criminal activity → stablecoin volume (settlement use)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.stattools import durbin_watson

from btc_analysis.config import get_config
from btc_analysis.data.cbp_seizures import fetch_drug_seizures
from btc_analysis.data.cbp_currency import fetch_currency_seizures

logger = logging.getLogger(__name__)

# Drug street value estimates (USD per lb) - same as BTC analysis
STREET_VALUE_PER_LB = {
    "cocaine_lbs": 15000,
    "fentanyl_lbs": 750000,
    "heroin_lbs": 25000,
    "meth_lbs": 5000,
    "marijuana_lbs": 500,
    "ecstasy_lbs": 10000,
    "ketamine_lbs": 8000,
    "other_drugs_lbs": 5000,
}

STABLECOINS = {
    "USDT-USD": "Tether",
    "USDC-USD": "USD Coin",
}


def run_stablecoin_analysis(
    output_dir: Optional[Path] = None,
    start_year: int = 2019,
    end_year: int = 2024,
) -> Dict[str, Any]:
    """
    Run stablecoin analysis testing criminal demand hypothesis.

    Tests:
    1. Volume changes as dependent variable
    2. Peg deviation as dependent variable
    3. Same lag structure as BTC analysis
    4. Comparison across stablecoins

    Returns:
        Dictionary containing all analysis results.
    """
    config = get_config()

    if output_dir is None:
        output_dir = config.paths.outputs_dir / "stablecoin"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Load criminal activity data
    logger.info("Loading criminal activity data...")
    crime_data = _load_crime_data(start_year, end_year)
    if crime_data.empty:
        logger.error("No criminal activity data available")
        return {"error": "No crime data"}

    # Analyze each stablecoin
    for ticker, name in STABLECOINS.items():
        logger.info(f"Analyzing {name} ({ticker})...")

        stable_results = _analyze_stablecoin(
            ticker, name, crime_data, start_year, end_year
        )
        results[ticker] = stable_results

    # Cross-stablecoin comparison
    results["comparison"] = _compare_stablecoins(results)

    # Save results
    _save_results(results, output_dir)

    # Generate report
    report = _generate_report(results)
    with open(output_dir / "stablecoin_analysis.txt", "w") as f:
        f.write(report)

    logger.info(f"Results saved to {output_dir}")

    return results


def _load_crime_data(start_year: int, end_year: int) -> pd.DataFrame:
    """Load and prepare criminal activity data."""
    # Load currency seizures
    currency = fetch_currency_seizures(start_year=start_year, end_year=end_year)
    if currency.empty:
        return pd.DataFrame()

    # Load drug seizures
    drugs = fetch_drug_seizures(start_year=start_year, end_year=end_year)
    if drugs.empty:
        return pd.DataFrame()

    # Merge
    df = pd.merge(currency, drugs, on=["date", "year", "month"], how="inner")

    # Calculate drug value
    drug_cols = [c for c in df.columns if c.endswith("_lbs")]
    df["drug_value_usd"] = 0
    for col in drug_cols:
        value_per_lb = STREET_VALUE_PER_LB.get(col, 5000)
        df["drug_value_usd"] += df[col].fillna(0) * value_per_lb

    # Cash/drug ratio
    df["cash_drug_ratio"] = df["currency_seizure_usd"] / df["drug_value_usd"]
    df["cash_drug_ratio"] = df["cash_drug_ratio"].replace([np.inf, -np.inf], np.nan)

    # Z-scores
    def zscore(s):
        return (s - s.mean()) / s.std()

    df["cash_z"] = zscore(df["currency_seizure_usd"])
    df["drug_z"] = zscore(df["drug_value_usd"])
    df["substitution_index"] = df["drug_z"] - df["cash_z"]
    df["substitution_z"] = zscore(df["substitution_index"])
    df["substitution_diff"] = df["substitution_index"].diff()

    # Lagged variables
    for lag in [1, 2, 3]:
        df[f"substitution_z_lag{lag}"] = df["substitution_z"].shift(lag)
        df[f"substitution_diff_lag{lag}"] = df["substitution_diff"].shift(lag)

    return df


def _fetch_stablecoin_data(ticker: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch stablecoin price and volume data."""
    import yfinance as yf

    try:
        data = yf.download(
            ticker,
            start=f"{start_year}-01-01",
            end=f"{end_year + 1}-01-01",
            progress=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        if data.empty:
            return pd.DataFrame()

        data = data.reset_index()
        data.columns = [c.lower() for c in data.columns]

        # Aggregate to monthly
        data["date"] = pd.to_datetime(data["date"])
        data["year"] = data["date"].dt.year
        data["month"] = data["date"].dt.month

        monthly = data.groupby(["year", "month"]).agg({
            "close": "last",
            "volume": "sum",
            "high": "max",
            "low": "min",
        }).reset_index()

        monthly["date"] = pd.to_datetime(
            monthly["year"].astype(str) + "-" +
            monthly["month"].astype(str).str.zfill(2) + "-01"
        )

        # Calculate metrics
        monthly["peg_deviation"] = monthly["close"] - 1.0  # Distance from $1
        monthly["peg_deviation_abs"] = monthly["peg_deviation"].abs()
        monthly["price_range"] = monthly["high"] - monthly["low"]

        # Log volume and changes
        monthly["log_volume"] = np.log(monthly["volume"])
        monthly = monthly.sort_values("date").reset_index(drop=True)
        monthly["volume_change"] = monthly["log_volume"].diff()

        # Z-score volume
        def zscore(s):
            return (s - s.mean()) / s.std()
        monthly["volume_z"] = zscore(monthly["volume"])

        return monthly

    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def _analyze_stablecoin(
    ticker: str,
    name: str,
    crime_data: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> Dict[str, Any]:
    """Run full analysis for a single stablecoin."""
    results = {"name": name, "ticker": ticker}

    # Fetch stablecoin data
    stable_data = _fetch_stablecoin_data(ticker, start_year, end_year)
    if stable_data.empty:
        return {"error": f"No data for {ticker}"}

    # Merge with crime data
    df = pd.merge(
        crime_data,
        stable_data,
        on=["date", "year", "month"],
        how="inner"
    )

    if len(df) < 20:
        return {"error": f"Insufficient observations ({len(df)})"}

    results["data_summary"] = {
        "n_observations": len(df),
        "date_range": f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}",
        "avg_volume": float(df["volume"].mean()),
        "avg_peg_deviation": float(df["peg_deviation"].mean()),
    }

    # Load S&P 500 for control
    try:
        import yfinance as yf
        sp500_data = yf.download("^GSPC", start=f"{start_year}-01-01",
                                  end=f"{end_year+1}-01-01", progress=False)
        if isinstance(sp500_data.columns, pd.MultiIndex):
            sp500_data.columns = [col[0] for col in sp500_data.columns]
        sp500 = sp500_data["Close"].resample("ME").last()
        sp500_df = pd.DataFrame({"date": sp500.index, "sp500_price": sp500.values})
        sp500_df["date"] = sp500_df["date"].dt.to_period("M").dt.to_timestamp()
        sp500_df["log_sp500"] = np.log(sp500_df["sp500_price"])
        sp500_df["sp500_return"] = sp500_df["log_sp500"].diff()
        df = pd.merge(df, sp500_df[["date", "sp500_return"]], on="date", how="left")
    except Exception as e:
        logger.warning(f"Could not load S&P 500: {e}")

    # Stationarity tests
    results["stationarity"] = _run_stationarity_tests(df)

    # Volume analysis (primary)
    results["volume_analysis"] = _run_volume_analysis(df)

    # Peg deviation analysis (secondary)
    results["peg_analysis"] = _run_peg_analysis(df)

    return results


def _run_stationarity_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """Run ADF tests on key variables."""
    results = {}

    test_vars = [
        ("log_volume", "Log Volume (levels)"),
        ("volume_change", "Volume Change (diff)"),
        ("peg_deviation", "Peg Deviation"),
        ("substitution_index", "Substitution Index"),
    ]

    for var, label in test_vars:
        if var in df.columns:
            series = df[var].dropna()
            if len(series) > 10:
                try:
                    adf_stat, p_value, *_ = adfuller(series, autolag="AIC")
                    results[var] = {
                        "label": label,
                        "p_value": float(p_value),
                        "stationary": p_value < 0.05,
                    }
                except Exception as e:
                    logger.warning(f"ADF failed for {var}: {e}")

    return results


def _run_volume_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze relationship between criminal activity and stablecoin volume.

    Primary hypothesis: criminal activity → stablecoin volume (settlement use)
    """
    results = {}

    # Check required columns
    if "volume_change" not in df.columns:
        return {"error": "Volume data not available"}

    df = df.copy()

    # Create interaction
    if "substitution_z" in df.columns and "volume_z" not in df.columns:
        # For stablecoins, we might use BTC volume as the "liquidity" proxy
        # But for now, skip the interaction since volume IS the DV
        pass

    # =========================================================================
    # 1. Contemporaneous model
    # =========================================================================
    required = ["volume_change", "substitution_diff"]
    if "sp500_return" in df.columns:
        required.append("sp500_return")

    df_clean = df.dropna(subset=required).copy()

    if len(df_clean) >= 20:
        controls = ["substitution_diff"]
        if "sp500_return" in df_clean.columns:
            controls.append("sp500_return")

        X = sm.add_constant(df_clean[controls])
        y = df_clean["volume_change"]

        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

        results["contemporaneous"] = {
            "n_obs": int(model.nobs),
            "r_squared": float(model.rsquared),
            "coefficients": {
                name: {
                    "coef": float(model.params[name]),
                    "p_value": float(model.pvalues[name]),
                }
                for name in model.params.index
            },
        }

        # Direct effect significance
        if "substitution_diff" in model.pvalues:
            results["contemporaneous"]["substitution_significant"] = (
                model.pvalues["substitution_diff"] < 0.1
            )

    # =========================================================================
    # 2. Lagged models (primary causal spec)
    # =========================================================================
    results["lagged"] = {}

    for lag in [1, 2, 3]:
        lag_var = f"substitution_diff_lag{lag}"
        if lag_var not in df.columns:
            continue

        required_lag = ["volume_change", lag_var]
        if "sp500_return" in df.columns:
            required_lag.append("sp500_return")

        df_lag = df.dropna(subset=required_lag).copy()

        if len(df_lag) >= 20:
            controls = [lag_var]
            if "sp500_return" in df_lag.columns:
                controls.append("sp500_return")

            X = sm.add_constant(df_lag[controls])
            y = df_lag["volume_change"]

            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

            results["lagged"][f"lag_{lag}"] = {
                "n_obs": int(model.nobs),
                "r_squared": float(model.rsquared),
                "substitution_coef": float(model.params.get(lag_var, np.nan)),
                "substitution_pvalue": float(model.pvalues.get(lag_var, np.nan)),
            }

    # =========================================================================
    # 3. Granger causality
    # =========================================================================
    try:
        df_granger = df[["volume_change", "substitution_diff"]].dropna()
        if len(df_granger) > 15:
            granger_results = grangercausalitytests(
                df_granger[["volume_change", "substitution_diff"]],
                maxlag=3,
                verbose=False
            )
            results["granger_causality"] = {
                lag: {
                    "f_stat": float(granger_results[lag][0]["ssr_ftest"][0]),
                    "p_value": float(granger_results[lag][0]["ssr_ftest"][1]),
                }
                for lag in [1, 2, 3]
            }
    except Exception as e:
        logger.warning(f"Granger test failed: {e}")

    # =========================================================================
    # 4. Simple correlation
    # =========================================================================
    for var in ["substitution_z", "substitution_z_lag1", "substitution_diff"]:
        if var in df.columns:
            sub = df[["volume_change", var]].dropna()
            if len(sub) > 10:
                corr, p = stats.pearsonr(sub["volume_change"], sub[var])
                results[f"corr_{var}"] = {
                    "correlation": float(corr),
                    "p_value": float(p),
                }

    return results


def _run_peg_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze relationship between criminal activity and peg deviation.

    Secondary hypothesis: high demand → premium to peg
    """
    results = {}

    if "peg_deviation" not in df.columns:
        return {"error": "Peg deviation not available"}

    df = df.copy()

    # Simple correlation
    for var in ["substitution_z", "substitution_z_lag1"]:
        if var in df.columns:
            sub = df[["peg_deviation", var]].dropna()
            if len(sub) > 10:
                corr, p = stats.pearsonr(sub["peg_deviation"], sub[var])
                results[f"corr_{var}"] = {
                    "correlation": float(corr),
                    "p_value": float(p),
                }

    # Regression
    required = ["peg_deviation", "substitution_z"]
    if "sp500_return" in df.columns:
        required.append("sp500_return")

    df_clean = df.dropna(subset=required).copy()

    if len(df_clean) >= 20:
        controls = ["substitution_z"]
        if "sp500_return" in df_clean.columns:
            controls.append("sp500_return")

        X = sm.add_constant(df_clean[controls])
        y = df_clean["peg_deviation"]

        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

        results["regression"] = {
            "n_obs": int(model.nobs),
            "r_squared": float(model.rsquared),
            "substitution_coef": float(model.params.get("substitution_z", np.nan)),
            "substitution_pvalue": float(model.pvalues.get("substitution_z", np.nan)),
        }

    return results


def _compare_stablecoins(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare results across stablecoins."""
    comparison = {}

    for ticker in STABLECOINS:
        if ticker not in results or "error" in results[ticker]:
            continue

        stable_res = results[ticker]
        vol_res = stable_res.get("volume_analysis", {})

        # Get lag-1 result (primary spec)
        lag1 = vol_res.get("lagged", {}).get("lag_1", {})

        comparison[ticker] = {
            "name": STABLECOINS[ticker],
            "n_obs": stable_res.get("data_summary", {}).get("n_observations"),
            "lag1_coef": lag1.get("substitution_coef"),
            "lag1_pvalue": lag1.get("substitution_pvalue"),
        }

    return comparison


def _save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save results to JSON."""
    import json

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_dir / "results.json", "w") as f:
        json.dump(convert_types(results), f, indent=2)


def _generate_report(results: Dict[str, Any]) -> str:
    """Generate text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("STABLECOIN ANALYSIS - Criminal Demand Hypothesis")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Hypothesis: Criminal activity predicts stablecoin volume")
    lines.append("(stablecoins used for settlement, so volume = usage)")
    lines.append("")

    for ticker, name in STABLECOINS.items():
        if ticker not in results:
            continue

        stable_res = results[ticker]

        lines.append("-" * 80)
        lines.append(f"{name} ({ticker})")
        lines.append("-" * 80)

        if "error" in stable_res:
            lines.append(f"  Error: {stable_res['error']}")
            continue

        # Data summary
        if "data_summary" in stable_res:
            ds = stable_res["data_summary"]
            lines.append(f"  N: {ds.get('n_observations')}")
            lines.append(f"  Period: {ds.get('date_range')}")
            lines.append(f"  Avg daily volume: ${ds.get('avg_volume', 0):,.0f}")
            lines.append(f"  Avg peg deviation: {ds.get('avg_peg_deviation', 0):.4f}")

        # Stationarity
        if "stationarity" in stable_res:
            lines.append("\n  Stationarity:")
            for var, test in stable_res["stationarity"].items():
                status = "STATIONARY" if test.get("stationary") else "NON-STATIONARY"
                lines.append(f"    {test.get('label', var)}: [{status}]")

        # Volume analysis
        vol_res = stable_res.get("volume_analysis", {})
        if vol_res and "error" not in vol_res:
            lines.append("\n  VOLUME ANALYSIS (DV = volume change):")

            # Contemporaneous
            if "contemporaneous" in vol_res:
                contemp = vol_res["contemporaneous"]
                coef_data = contemp.get("coefficients", {}).get("substitution_diff", {})
                coef = coef_data.get("coef", np.nan)
                p = coef_data.get("p_value", np.nan)
                sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                lines.append(f"    Contemporaneous: coef={coef:+.4f} (p={p:.4f}) {sig}")

            # Lagged
            if "lagged" in vol_res:
                for lag_key in ["lag_1", "lag_2", "lag_3"]:
                    if lag_key in vol_res["lagged"]:
                        lag_data = vol_res["lagged"][lag_key]
                        coef = lag_data.get("substitution_coef", np.nan)
                        p = lag_data.get("substitution_pvalue", np.nan)
                        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                        lines.append(f"    {lag_key}: coef={coef:+.4f} (p={p:.4f}) {sig}")

            # Granger
            if "granger_causality" in vol_res:
                lines.append("\n    Granger Causality (Subst -> Volume):")
                for lag, gc in vol_res["granger_causality"].items():
                    sig = "***" if gc["p_value"] < 0.01 else "**" if gc["p_value"] < 0.05 else "*" if gc["p_value"] < 0.1 else ""
                    lines.append(f"      Lag {lag}: F={gc['f_stat']:.2f}, p={gc['p_value']:.4f} {sig}")

        # Peg analysis
        peg_res = stable_res.get("peg_analysis", {})
        if peg_res and "error" not in peg_res:
            lines.append("\n  PEG DEVIATION ANALYSIS:")
            if "regression" in peg_res:
                reg = peg_res["regression"]
                coef = reg.get("substitution_coef", np.nan)
                p = reg.get("substitution_pvalue", np.nan)
                sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                lines.append(f"    Substitution -> Peg: coef={coef:+.6f} (p={p:.4f}) {sig}")

        lines.append("")

    # Comparison
    if "comparison" in results and results["comparison"]:
        lines.append("=" * 80)
        lines.append("COMPARISON ACROSS STABLECOINS")
        lines.append("=" * 80)
        lines.append(f"{'Stablecoin':<15} {'N':>6} {'Lag-1 Coef':>12} {'p-value':>10}")
        lines.append("-" * 45)

        for ticker, comp in results["comparison"].items():
            coef = comp.get("lag1_coef")
            p = comp.get("lag1_pvalue")
            if coef is not None and p is not None:
                sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                lines.append(f"{comp['name']:<15} {comp.get('n_obs', 'N/A'):>6} {coef:>+12.4f} {p:>10.4f} {sig}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.1")
    lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_stablecoin_analysis()
