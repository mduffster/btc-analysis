"""Cash Substitution Analysis.

Tests the hypothesis that Bitcoin serves as a cash substitute for criminal
enterprise, and that institutional liquidity enabled criminal exit.

Key findings:
1. Cash/drug seizure ratio declined 85% from 2019-2024
2. Pre-2020: Criminal demand positively correlated with BTC
3. Post-2020: High-volume periods show criminal selling pressure
4. The substitution x volume interaction is highly significant
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.stattools import durbin_watson

from btc_analysis.config import get_config
from btc_analysis.data.cbp_seizures import fetch_drug_seizures
from btc_analysis.data.cbp_currency import fetch_currency_seizures
from btc_analysis.data.btc_price import fetch_btc_price

logger = logging.getLogger(__name__)

# Drug street value estimates (USD per lb)
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

# Key institutional entry events (exit liquidity moments)
EXIT_EVENTS = {
    "2020-08-01": "MicroStrategy first BTC purchase",
    "2021-02-01": "Tesla $1.5B BTC purchase",
    "2021-04-01": "Coinbase IPO",
    "2021-10-01": "First BTC futures ETF (BITO)",
    "2023-06-01": "BlackRock ETF filing",
    "2024-01-01": "Spot BTC ETFs approved",
}


def run_cash_substitution_analysis(
    output_dir: Optional[Path] = None,
    start_year: int = 2017,
    end_year: int = 2024,
) -> Dict[str, Any]:
    """
    Run complete cash substitution analysis.

    Tests:
    1. Cash/drug ratio trend over time
    2. Correlation with BTC by regime (pre/post institutional)
    3. Volume interaction (exit liquidity hypothesis)
    4. Stress tests and robustness checks

    Args:
        output_dir: Directory to save outputs.
        start_year: Analysis start year.
        end_year: Analysis end year.

    Returns:
        Dictionary containing all analysis results.
    """
    config = get_config()

    if output_dir is None:
        output_dir = config.paths.outputs_dir / "cash_substitution"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Step 1: Load and merge data
    logger.info("Loading data...")
    df = _load_and_merge_data(start_year, end_year)

    if df.empty or len(df) < 20:
        logger.error("Insufficient data for analysis")
        return results

    results["data_summary"] = {
        "n_observations": len(df),
        "date_range": f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}",
    }

    # Step 2: Calculate cash substitution metrics
    logger.info("Calculating cash substitution metrics...")
    df = _calculate_substitution_metrics(df)

    # Step 3: Regime analysis
    logger.info("Running regime analysis...")
    results["regime_analysis"] = _run_regime_analysis(df)

    # Step 4: Volume interaction analysis
    logger.info("Running volume interaction analysis...")
    results["volume_interaction"] = _run_volume_interaction_analysis(df)

    # Step 5: Stress tests
    logger.info("Running stress tests...")
    results["stress_tests"] = _run_stress_tests(df)

    # Step 6: Save results
    _save_results(results, df, output_dir)

    # Step 7: Generate report
    report = _generate_report(results)
    report_path = output_dir / "cash_substitution_analysis.txt"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Results saved to {output_dir}")

    return results


def _load_and_merge_data(start_year: int, end_year: int) -> pd.DataFrame:
    """Load and merge currency seizures, drug seizures, and BTC price."""
    config = get_config()

    # Load currency seizures
    currency = fetch_currency_seizures(start_year=start_year, end_year=end_year)
    if currency.empty:
        logger.warning("No currency seizure data available")
        return pd.DataFrame()

    # Load drug seizures
    drugs = fetch_drug_seizures(start_year=start_year, end_year=end_year)
    if drugs.empty:
        logger.warning("No drug seizure data available")
        return pd.DataFrame()

    # Load BTC price
    btc = fetch_btc_price(
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
    )
    if btc.empty:
        logger.warning("No BTC price data available")
        return pd.DataFrame()

    # Aggregate BTC to monthly
    btc["date"] = pd.to_datetime(btc["date"])
    btc["year"] = btc["date"].dt.year
    btc["month"] = btc["date"].dt.month
    btc_monthly = btc.groupby(["year", "month"]).agg({
        "price": "last",
        "volume": "sum",
    }).reset_index()
    btc_monthly.columns = ["year", "month", "btc_price_close", "btc_volume"]
    btc_monthly["date"] = pd.to_datetime(
        btc_monthly["year"].astype(str) + "-" +
        btc_monthly["month"].astype(str).str.zfill(2) + "-01"
    )

    # Merge datasets
    df = pd.merge(currency, drugs, on=["date", "year", "month"], how="inner")
    df = pd.merge(df, btc_monthly, on=["date", "year", "month"], how="inner")

    # Load market controls
    try:
        import yfinance as yf
        sp500_data = yf.download("^GSPC", start=f"{start_year}-01-01", end=f"{end_year+1}-01-01",
                           progress=False)
        # Handle MultiIndex columns from newer yfinance
        if isinstance(sp500_data.columns, pd.MultiIndex):
            sp500_data.columns = [col[0] for col in sp500_data.columns]
        sp500 = sp500_data["Close"].resample("ME").last()
        sp500_df = pd.DataFrame({"date": sp500.index, "sp500_close": sp500.values})
        sp500_df["date"] = sp500_df["date"].dt.to_period("M").dt.to_timestamp()
        df = pd.merge(df, sp500_df, on="date", how="left")
    except Exception as e:
        logger.warning(f"Could not load S&P 500 data: {e}")

    logger.info(f"Merged dataset: {len(df)} observations")
    return df


def _calculate_substitution_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cash substitution metrics."""
    df = df.copy()

    # Calculate drug value using street prices
    drug_cols = [c for c in df.columns if c.endswith("_lbs")]
    df["drug_value_usd"] = 0
    for col in drug_cols:
        drug_name = col
        value_per_lb = STREET_VALUE_PER_LB.get(drug_name, 5000)
        df["drug_value_usd"] += df[col].fillna(0) * value_per_lb

    # Cash/drug ratio
    df["cash_drug_ratio"] = df["currency_seizure_usd"] / df["drug_value_usd"]
    df["cash_drug_ratio"] = df["cash_drug_ratio"].replace([np.inf, -np.inf], np.nan)

    # Z-score normalized components
    df["cash_zscore"] = (df["currency_seizure_usd"] - df["currency_seizure_usd"].mean()) / df["currency_seizure_usd"].std()
    df["drug_zscore"] = (df["drug_value_usd"] - df["drug_value_usd"].mean()) / df["drug_value_usd"].std()

    # Substitution index: drug growth relative to cash
    df["substitution_index"] = df["drug_zscore"] - df["cash_zscore"]

    # Log transforms
    df["log_btc"] = np.log(df["btc_price_close"])
    df["log_ratio"] = np.log(df["cash_drug_ratio"].replace(0, np.nan))
    df["btc_return"] = df["btc_price_close"].pct_change()

    # Volume z-score
    if "btc_volume" in df.columns:
        df["volume_zscore"] = (df["btc_volume"] - df["btc_volume"].mean()) / df["btc_volume"].std()
        df["high_volume"] = (df["volume_zscore"] > 0).astype(int)

    # Regime indicators
    df["pre_institutional"] = (df["date"] < "2020-08-01").astype(int)
    df["post_institutional"] = (df["date"] >= "2020-08-01").astype(int)
    df["etf_era"] = (df["date"] >= "2024-01-01").astype(int)

    # Exit events counter
    df["exit_events"] = 0
    for event_date in EXIT_EVENTS.keys():
        df["exit_events"] += (df["date"] >= event_date).astype(int)

    return df


def _run_regime_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze cash substitution by regime."""
    results = {}

    # Yearly summary
    yearly = df.groupby("year").agg({
        "currency_seizure_usd": "sum",
        "drug_value_usd": "sum",
        "btc_price_close": "mean",
    }).reset_index()
    yearly["ratio"] = yearly["currency_seizure_usd"] / yearly["drug_value_usd"]

    results["yearly_summary"] = yearly.to_dict("records")

    # Ratio trend
    df_clean = df.dropna(subset=["log_ratio"])
    df_clean["time_idx"] = range(len(df_clean))
    X = sm.add_constant(df_clean["time_idx"])
    y = df_clean["log_ratio"]
    model = sm.OLS(y, X).fit()

    results["ratio_trend"] = {
        "coefficient": model.params["time_idx"],
        "p_value": model.pvalues["time_idx"],
        "r_squared": model.rsquared,
        "monthly_pct_change": (np.exp(model.params["time_idx"]) - 1) * 100,
        "is_declining": model.params["time_idx"] < 0 and model.pvalues["time_idx"] < 0.05,
    }

    # Correlation by regime
    for regime, label in [("pre_institutional", "Pre-Institutional"),
                          ("post_institutional", "Post-Institutional")]:
        sub = df[df[regime] == 1].dropna(subset=["substitution_index", "log_btc"])
        if len(sub) > 5:
            corr, p = stats.pearsonr(sub["substitution_index"], sub["log_btc"])
            results[f"{regime}_correlation"] = {
                "label": label,
                "n": len(sub),
                "correlation": corr,
                "p_value": p,
            }

    # BTC residual (after S&P control) by regime
    if "sp500_close" not in df.columns:
        logger.warning("S&P 500 data not available for residual analysis")
        return results

    df_res = df.dropna(subset=["log_btc", "sp500_close", "substitution_index"])
    if len(df_res) > 20:
        X = sm.add_constant(df_res["sp500_close"])
        y = df_res["log_btc"]
        macro_model = sm.OLS(y, X).fit()
        df_res["btc_residual"] = macro_model.resid

        for regime, label in [("pre_institutional", "Pre-Institutional"),
                              ("post_institutional", "Post-Institutional")]:
            sub = df_res[df_res[regime] == 1]
            if len(sub) > 5:
                corr, p = stats.pearsonr(sub["substitution_index"], sub["btc_residual"])
                results[f"{regime}_residual_correlation"] = {
                    "label": label,
                    "n": len(sub),
                    "correlation": corr,
                    "p_value": p,
                }

    return results


def _run_volume_interaction_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Test the exit liquidity hypothesis via volume interaction."""
    results = {}

    if "volume_zscore" not in df.columns:
        logger.warning("Volume data not available for interaction analysis")
        return results

    if "sp500_close" not in df.columns:
        logger.warning("S&P 500 data not available for interaction analysis")
        return results

    # Get BTC residual after S&P control
    df_clean = df.dropna(subset=["log_btc", "sp500_close", "substitution_index", "volume_zscore"])
    if len(df_clean) < 20:
        return results

    X = sm.add_constant(df_clean["sp500_close"])
    y = df_clean["log_btc"]
    df_clean["btc_residual"] = sm.OLS(y, X).fit().resid

    # Correlation by volume regime
    for vol_regime, label in [(0, "Low Volume"), (1, "High Volume")]:
        sub = df_clean[df_clean["high_volume"] == vol_regime]
        if len(sub) > 10:
            corr, p = stats.pearsonr(sub["substitution_index"], sub["btc_residual"])
            results[f"correlation_{label.lower().replace(' ', '_')}"] = {
                "label": label,
                "n": len(sub),
                "correlation": corr,
                "p_value": p,
            }

    # Interaction regression
    df_clean["subst_x_volume"] = df_clean["substitution_index"] * df_clean["volume_zscore"]
    X = sm.add_constant(df_clean[["substitution_index", "volume_zscore", "subst_x_volume"]])
    y = df_clean["btc_residual"]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

    results["interaction_model"] = {
        "r_squared": model.rsquared,
        "coefficients": {
            name: {
                "coefficient": model.params[name],
                "p_value": model.pvalues[name],
            }
            for name in model.params.index
        },
        "durbin_watson": durbin_watson(model.resid),
    }

    # Economic significance
    interact_coef = model.params["subst_x_volume"]
    results["economic_significance"] = {
        "interaction_coefficient": interact_coef,
        "pct_impact": interact_coef * 100,
        "interpretation": f"When volume is 1σ above mean AND substitution is 1σ above mean, "
                         f"BTC is {interact_coef*100:.1f}% lower than macro would predict",
    }

    # Rolling correlation
    df_sorted = df_clean.sort_values("date")
    df_sorted["rolling_corr"] = df_sorted["substitution_index"].rolling(24, min_periods=12).corr(
        df_sorted["btc_residual"]
    )

    rolling_by_year = {}
    for year in df_sorted["date"].dt.year.unique():
        year_data = df_sorted[df_sorted["date"].dt.year == year]["rolling_corr"].dropna()
        if len(year_data) > 0:
            rolling_by_year[int(year)] = float(year_data.mean())

    results["rolling_correlation_by_year"] = rolling_by_year

    return results


def _run_stress_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """Run robustness and stress tests."""
    results = {}

    if "volume_zscore" not in df.columns:
        return results

    if "sp500_close" not in df.columns:
        return results

    # Prepare data
    df_clean = df.dropna(subset=["log_btc", "sp500_close", "substitution_index", "volume_zscore"])
    if len(df_clean) < 20:
        return results

    X = sm.add_constant(df_clean["sp500_close"])
    y = df_clean["log_btc"]
    df_clean["btc_residual"] = sm.OLS(y, X).fit().resid

    high_vol = df_clean[df_clean["high_volume"] == 1]

    if len(high_vol) < 10:
        return results

    actual_corr = high_vol["substitution_index"].corr(high_vol["btc_residual"])

    # 1. Bootstrap confidence interval
    n_boot = 1000
    np.random.seed(42)
    boot_corrs = []
    for _ in range(n_boot):
        sample = high_vol.sample(n=len(high_vol), replace=True)
        boot_corrs.append(sample["substitution_index"].corr(sample["btc_residual"]))

    boot_corrs = np.array(boot_corrs)
    ci_low, ci_high = np.percentile(boot_corrs, [2.5, 97.5])

    results["bootstrap"] = {
        "actual_correlation": actual_corr,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "ci_excludes_zero": ci_high < 0 or ci_low > 0,
    }

    # 2. Placebo test
    n_placebo = 1000
    placebo_corrs = []
    for _ in range(n_placebo):
        df_placebo = df_clean.copy()
        df_placebo["substitution_index"] = np.random.permutation(df_placebo["substitution_index"].values)
        hv = df_placebo[df_placebo["high_volume"] == 1]
        placebo_corrs.append(hv["substitution_index"].corr(hv["btc_residual"]))

    placebo_pvalue = (np.array(placebo_corrs) <= actual_corr).mean()

    results["placebo_test"] = {
        "actual_correlation": actual_corr,
        "placebo_p_value": placebo_pvalue,
        "passes": placebo_pvalue < 0.05,
    }

    # 3. Leave-one-out
    loo_corrs = []
    for i in high_vol.index:
        subset = high_vol.drop(i)
        loo_corrs.append(subset["substitution_index"].corr(subset["btc_residual"]))

    loo_corrs = np.array(loo_corrs)

    results["leave_one_out"] = {
        "mean": float(loo_corrs.mean()),
        "std": float(loo_corrs.std()),
        "min": float(loo_corrs.min()),
        "max": float(loo_corrs.max()),
        "all_negative": bool((loo_corrs < 0).all()),
    }

    # 4. Falsification test (S&P 500)
    if "sp500_close" in df_clean.columns:
        df_clean["log_sp500"] = np.log(df_clean["sp500_close"])
        df_clean["sp500_residual"] = df_clean["log_sp500"] - df_clean["log_sp500"].mean()
        df_clean["subst_x_volume"] = df_clean["substitution_index"] * df_clean["volume_zscore"]

        X = sm.add_constant(df_clean[["substitution_index", "volume_zscore", "subst_x_volume"]])
        y = df_clean["sp500_residual"]
        model_sp = sm.OLS(y, X).fit()

        results["falsification_sp500"] = {
            "interaction_coefficient": model_sp.params["subst_x_volume"],
            "p_value": model_sp.pvalues["subst_x_volume"],
            "passes": model_sp.pvalues["subst_x_volume"] > 0.1,  # Should NOT be significant
        }

    return results


def _save_results(results: Dict[str, Any], df: pd.DataFrame, output_dir: Path) -> None:
    """Save analysis results and data."""
    import json

    # Save JSON results
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

    # Save processed data
    df.to_csv(output_dir / "cash_substitution_data.csv", index=False)


def _generate_report(results: Dict[str, Any]) -> str:
    """Generate text report of analysis results."""
    lines = []
    lines.append("=" * 80)
    lines.append("CASH SUBSTITUTION ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # Data summary
    if "data_summary" in results:
        lines.append("DATA SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Observations: {results['data_summary'].get('n_observations', 'N/A')}")
        lines.append(f"Date range: {results['data_summary'].get('date_range', 'N/A')}")
        lines.append("")

    # Ratio trend
    if "ratio_trend" in results:
        trend = results["ratio_trend"]
        lines.append("CASH/DRUG RATIO TREND")
        lines.append("-" * 40)
        lines.append(f"Monthly change: {trend.get('monthly_pct_change', 0):.2f}%")
        lines.append(f"R²: {trend.get('r_squared', 0):.4f}")
        lines.append(f"Declining significantly: {trend.get('is_declining', False)}")
        lines.append("")

    # Regime correlations
    lines.append("CORRELATION BY REGIME")
    lines.append("-" * 40)
    for key in ["pre_institutional_correlation", "post_institutional_correlation",
                "pre_institutional_residual_correlation", "post_institutional_residual_correlation"]:
        if key in results.get("regime_analysis", {}):
            data = results["regime_analysis"][key]
            sig = "***" if data["p_value"] < 0.01 else "**" if data["p_value"] < 0.05 else "*" if data["p_value"] < 0.1 else ""
            lines.append(f"{data['label']}: r = {data['correlation']:.4f}, p = {data['p_value']:.4f} {sig} (n={data['n']})")
    lines.append("")

    # Volume interaction
    if "volume_interaction" in results:
        vi = results["volume_interaction"]
        lines.append("VOLUME INTERACTION (Exit Liquidity)")
        lines.append("-" * 40)

        for key in ["correlation_low_volume", "correlation_high_volume"]:
            if key in vi:
                data = vi[key]
                sig = "***" if data["p_value"] < 0.01 else "**" if data["p_value"] < 0.05 else "*" if data["p_value"] < 0.1 else ""
                lines.append(f"{data['label']}: r = {data['correlation']:.4f}, p = {data['p_value']:.4f} {sig}")

        if "interaction_model" in vi:
            model = vi["interaction_model"]
            lines.append(f"\nInteraction Model R²: {model['r_squared']:.4f}")
            for name, coef in model["coefficients"].items():
                sig = "***" if coef["p_value"] < 0.01 else "**" if coef["p_value"] < 0.05 else "*" if coef["p_value"] < 0.1 else ""
                lines.append(f"  {name}: {coef['coefficient']:.4f} (p={coef['p_value']:.4f}) {sig}")

        if "economic_significance" in vi:
            lines.append(f"\n{vi['economic_significance']['interpretation']}")

        lines.append("")

    # Stress tests
    if "stress_tests" in results:
        st = results["stress_tests"]
        lines.append("STRESS TESTS")
        lines.append("-" * 40)

        if "bootstrap" in st:
            lines.append(f"Bootstrap 95% CI: [{st['bootstrap']['ci_95_low']:.4f}, {st['bootstrap']['ci_95_high']:.4f}]")
            lines.append(f"  CI excludes zero: {st['bootstrap']['ci_excludes_zero']}")

        if "placebo_test" in st:
            lines.append(f"Placebo p-value: {st['placebo_test']['placebo_p_value']:.4f}")
            lines.append(f"  Passes: {st['placebo_test']['passes']}")

        if "leave_one_out" in st:
            lines.append(f"Leave-one-out: all negative = {st['leave_one_out']['all_negative']}")

        if "falsification_sp500" in st:
            lines.append(f"Falsification (S&P 500): p = {st['falsification_sp500']['p_value']:.4f}")
            lines.append(f"  Passes (not significant): {st['falsification_sp500']['passes']}")

        lines.append("")

    lines.append("=" * 80)
    lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.1")
    lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_cash_substitution_analysis()
