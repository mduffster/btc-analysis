"""Cash Substitution Analysis.

Tests the hypothesis that Bitcoin serves as a cash substitute for criminal
enterprise, and that institutional liquidity enabled criminal exit.

Methodological notes:
- Primary spec uses log returns (stationary), levels models are descriptive only
- Single regression with controls (no two-step residualization)
- Variables centered/standardized before forming interactions
- Lagged specifications (1-3 months) for causal inference
- Symmetric falsification test using same functional form
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
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
    2. Returns-based regression (PRIMARY - stationary)
    3. Lagged specifications for causal inference
    4. Volume interaction with centered variables
    5. Stress tests and falsification

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

    # Step 2: Calculate metrics with proper transformations
    logger.info("Calculating metrics...")
    df = _calculate_metrics(df)

    # Step 3: Stationarity tests
    logger.info("Running stationarity tests...")
    results["stationarity"] = _run_stationarity_tests(df)

    # Step 4: Descriptive regime analysis (levels - for context only)
    logger.info("Running descriptive regime analysis...")
    results["regime_analysis"] = _run_regime_analysis(df)

    # Step 5: PRIMARY SPEC - Returns-based single regression
    logger.info("Running primary returns-based analysis...")
    results["primary_returns_model"] = _run_returns_regression(df)

    # Step 6: Lagged specifications for causal inference
    logger.info("Running lagged specifications...")
    results["lagged_models"] = _run_lagged_regressions(df)

    # Step 7: Volume interaction with centered variables
    logger.info("Running volume interaction analysis...")
    results["volume_interaction"] = _run_volume_interaction_analysis(df)

    # Step 8: Stress tests with symmetric falsification
    logger.info("Running stress tests...")
    results["stress_tests"] = _run_stress_tests(df)

    # Step 9: Save results
    _save_results(results, df, output_dir)

    # Step 10: Generate report
    report = _generate_report(results)
    report_path = output_dir / "cash_substitution_analysis.txt"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Results saved to {output_dir}")

    return results


def _load_and_merge_data(start_year: int, end_year: int) -> pd.DataFrame:
    """Load and merge currency seizures, drug seizures, and BTC price."""
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
    btc = btc.copy()
    btc["date"] = pd.to_datetime(btc["date"])
    btc["year"] = btc["date"].dt.year
    btc["month"] = btc["date"].dt.month
    btc_monthly = btc.groupby(["year", "month"]).agg({
        "price": "last",
        "volume": "sum",
    }).reset_index()
    btc_monthly.columns = ["year", "month", "btc_price", "btc_volume"]
    btc_monthly["date"] = pd.to_datetime(
        btc_monthly["year"].astype(str) + "-" +
        btc_monthly["month"].astype(str).str.zfill(2) + "-01"
    )

    # Merge datasets
    df = pd.merge(currency, drugs, on=["date", "year", "month"], how="inner")
    df = pd.merge(df, btc_monthly, on=["date", "year", "month"], how="inner")

    # Load S&P 500 for macro control
    try:
        import yfinance as yf
        sp500_data = yf.download(
            "^GSPC",
            start=f"{start_year}-01-01",
            end=f"{end_year+1}-01-01",
            progress=False
        )
        # Handle MultiIndex columns from newer yfinance
        if isinstance(sp500_data.columns, pd.MultiIndex):
            sp500_data.columns = [col[0] for col in sp500_data.columns]
        sp500 = sp500_data["Close"].resample("ME").last()
        sp500_df = pd.DataFrame({"date": sp500.index, "sp500_price": sp500.values})
        sp500_df["date"] = sp500_df["date"].dt.to_period("M").dt.to_timestamp()
        df = pd.merge(df, sp500_df, on="date", how="left")
    except Exception as e:
        logger.warning(f"Could not load S&P 500 data: {e}")

    logger.info(f"Merged dataset: {len(df)} observations")
    return df


def _calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all metrics with proper transformations."""
    df = df.copy()

    # Calculate drug value using street prices
    drug_cols = [c for c in df.columns if c.endswith("_lbs")]
    df["drug_value_usd"] = 0
    for col in drug_cols:
        value_per_lb = STREET_VALUE_PER_LB.get(col, 5000)
        df["drug_value_usd"] += df[col].fillna(0) * value_per_lb

    # Cash/drug ratio
    df["cash_drug_ratio"] = df["currency_seizure_usd"] / df["drug_value_usd"]
    df["cash_drug_ratio"] = df["cash_drug_ratio"].replace([np.inf, -np.inf], np.nan)

    # Log transforms (for levels analysis - descriptive only)
    df["log_btc"] = np.log(df["btc_price"])
    df["log_ratio"] = np.log(df["cash_drug_ratio"].replace(0, np.nan))
    if "sp500_price" in df.columns:
        df["log_sp500"] = np.log(df["sp500_price"])

    # RETURNS (PRIMARY - stationary)
    df = df.sort_values("date").reset_index(drop=True)
    df["btc_return"] = df["log_btc"].diff()
    if "log_sp500" in df.columns:
        df["sp500_return"] = df["log_sp500"].diff()

    # Substitution index components - Z-scored for interpretability
    df["cash_z"] = _zscore(df["currency_seizure_usd"])
    df["drug_z"] = _zscore(df["drug_value_usd"])
    df["substitution_index"] = df["drug_z"] - df["cash_z"]

    # Z-score the substitution index itself for interaction terms
    df["substitution_z"] = _zscore(df["substitution_index"])

    # First difference of substitution (for returns regression)
    df["substitution_diff"] = df["substitution_index"].diff()

    # Volume - Z-scored
    if "btc_volume" in df.columns:
        df["volume_z"] = _zscore(df["btc_volume"])
        df["high_volume"] = (df["volume_z"] > 0).astype(int)

    # Create lagged variables (1-3 months)
    for lag in [1, 2, 3]:
        df[f"substitution_z_lag{lag}"] = df["substitution_z"].shift(lag)
        df[f"substitution_diff_lag{lag}"] = df["substitution_diff"].shift(lag)

    # Regime indicators
    df["pre_institutional"] = (df["date"] < "2020-08-01").astype(int)
    df["post_institutional"] = (df["date"] >= "2020-08-01").astype(int)
    df["etf_era"] = (df["date"] >= "2024-01-01").astype(int)

    # Exit events counter
    df["exit_events"] = 0
    for event_date in EXIT_EVENTS.keys():
        df["exit_events"] += (df["date"] >= event_date).astype(int)

    return df


def _zscore(series: pd.Series) -> pd.Series:
    """Calculate z-score, handling NaN values."""
    return (series - series.mean()) / series.std()


def _run_stationarity_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """Run ADF tests on key variables."""
    results = {}

    test_vars = [
        ("log_btc", "Log BTC (levels)"),
        ("btc_return", "BTC Return (diff)"),
        ("substitution_index", "Substitution Index (levels)"),
        ("substitution_diff", "Substitution Index (diff)"),
    ]

    for var, label in test_vars:
        if var in df.columns:
            series = df[var].dropna()
            if len(series) > 10:
                try:
                    adf_stat, p_value, _, _, crit_values, _ = adfuller(series, autolag="AIC")
                    results[var] = {
                        "label": label,
                        "adf_statistic": adf_stat,
                        "p_value": p_value,
                        "stationary": p_value < 0.05,
                        "critical_1pct": crit_values["1%"],
                        "critical_5pct": crit_values["5%"],
                    }
                except Exception as e:
                    logger.warning(f"ADF test failed for {var}: {e}")

    return results


def _run_regime_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Descriptive regime analysis (levels - for context only)."""
    results = {}
    results["note"] = "Levels analysis is descriptive only due to nonstationarity"

    # Yearly summary
    yearly = df.groupby("year").agg({
        "currency_seizure_usd": "sum",
        "drug_value_usd": "sum",
        "btc_price": "mean",
    }).reset_index()
    yearly["ratio"] = yearly["currency_seizure_usd"] / yearly["drug_value_usd"]
    results["yearly_summary"] = yearly.to_dict("records")

    # Ratio trend (descriptive)
    df_clean = df.dropna(subset=["log_ratio"]).copy()
    if len(df_clean) > 10:
        df_clean["time_idx"] = range(len(df_clean))
        X = sm.add_constant(df_clean["time_idx"])
        y = df_clean["log_ratio"]
        model = sm.OLS(y, X).fit()

        results["ratio_trend"] = {
            "coefficient": float(model.params["time_idx"]),
            "p_value": float(model.pvalues["time_idx"]),
            "monthly_pct_change": float((np.exp(model.params["time_idx"]) - 1) * 100),
            "is_declining": model.params["time_idx"] < 0 and model.pvalues["time_idx"] < 0.05,
        }

    return results


def _run_returns_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """
    PRIMARY SPECIFICATION: Single regression on returns.

    Model: btc_return ~ sp500_return + substitution_diff + volume_z + substitution_z * volume_z

    Uses returns (stationary), single regression (no generated regressors),
    and centered variables for interactions.
    """
    results = {}

    required = ["btc_return", "sp500_return", "substitution_diff", "volume_z", "substitution_z"]
    if not all(col in df.columns for col in required):
        logger.warning("Missing required columns for returns regression")
        return results

    df_clean = df.dropna(subset=required).copy()
    if len(df_clean) < 20:
        return results

    # Create interaction with CENTERED variables
    df_clean["subst_x_vol"] = df_clean["substitution_z"] * df_clean["volume_z"]

    # Single regression with all controls
    X = df_clean[["sp500_return", "substitution_diff", "volume_z", "subst_x_vol"]]
    X = sm.add_constant(X)
    y = df_clean["btc_return"]

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    results["model_summary"] = {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "n_obs": int(model.nobs),
        "durbin_watson": float(durbin_watson(model.resid)),
    }

    results["coefficients"] = {}
    for name in model.params.index:
        results["coefficients"][name] = {
            "coef": float(model.params[name]),
            "std_err": float(model.bse[name]),
            "t_stat": float(model.tvalues[name]),
            "p_value": float(model.pvalues[name]),
        }

    # Economic interpretation
    interact_coef = model.params.get("subst_x_vol", 0)
    results["interpretation"] = {
        "interaction_effect": float(interact_coef),
        "description": (
            f"When both substitution and volume are 1σ above mean, "
            f"BTC monthly return is {interact_coef*100:.2f}pp {'lower' if interact_coef < 0 else 'higher'}"
        ),
    }

    return results


def _run_lagged_regressions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run lagged specifications for causal inference.

    Contemporaneous effects are weak for causality claims.
    Lagged substitution (1-3 months) is the primary causal spec.
    """
    results = {}

    if "btc_return" not in df.columns or "sp500_return" not in df.columns:
        return results

    # Test each lag
    for lag in [1, 2, 3]:
        lag_var = f"substitution_diff_lag{lag}"
        if lag_var not in df.columns:
            continue

        df_clean = df.dropna(subset=["btc_return", "sp500_return", lag_var, "volume_z"]).copy()
        if len(df_clean) < 20:
            continue

        # Create lagged interaction
        subst_z_lag = f"substitution_z_lag{lag}"
        if subst_z_lag in df_clean.columns:
            df_clean["subst_x_vol_lag"] = df_clean[subst_z_lag] * df_clean["volume_z"]
        else:
            df_clean["subst_x_vol_lag"] = 0

        X = df_clean[["sp500_return", lag_var, "volume_z", "subst_x_vol_lag"]]
        X = sm.add_constant(X)
        y = df_clean["btc_return"]

        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

        results[f"lag_{lag}"] = {
            "n_obs": int(model.nobs),
            "r_squared": float(model.rsquared),
            "substitution_coef": float(model.params.get(lag_var, np.nan)),
            "substitution_pvalue": float(model.pvalues.get(lag_var, np.nan)),
            "interaction_coef": float(model.params.get("subst_x_vol_lag", np.nan)),
            "interaction_pvalue": float(model.pvalues.get("subst_x_vol_lag", np.nan)),
        }

    # Granger causality test
    try:
        df_granger = df[["btc_return", "substitution_diff"]].dropna()
        if len(df_granger) > 15:
            granger_results = grangercausalitytests(
                df_granger[["btc_return", "substitution_diff"]],
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

    return results


def _run_volume_interaction_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Volume interaction analysis with properly centered variables.

    Key methodological points:
    - Variables are z-scored BEFORE forming interaction
    - Single regression with all controls (no two-step residualization)
    - Uses returns (stationary) as primary specification
    """
    results = {}

    if "volume_z" not in df.columns:
        logger.warning("Volume data not available")
        return results

    required = ["btc_return", "sp500_return", "substitution_z", "volume_z"]
    if not all(col in df.columns for col in required):
        return results

    df_clean = df.dropna(subset=required).copy()
    if len(df_clean) < 20:
        return results

    # Split sample correlations (descriptive)
    for vol_regime, label in [(0, "Low Volume"), (1, "High Volume")]:
        sub = df_clean[df_clean["high_volume"] == vol_regime].copy()
        if len(sub) > 10:
            corr, p = stats.pearsonr(sub["substitution_z"], sub["btc_return"])
            results[f"correlation_{label.lower().replace(' ', '_')}"] = {
                "label": label,
                "n": len(sub),
                "correlation": float(corr),
                "p_value": float(p),
            }

    # Create interaction with CENTERED (z-scored) variables
    df_clean["subst_x_vol"] = df_clean["substitution_z"] * df_clean["volume_z"]

    # Single regression - PRIMARY SPEC
    X = df_clean[["sp500_return", "substitution_z", "volume_z", "subst_x_vol"]]
    X = sm.add_constant(X)
    y = df_clean["btc_return"]

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    results["interaction_model"] = {
        "r_squared": float(model.rsquared),
        "durbin_watson": float(durbin_watson(model.resid)),
        "coefficients": {
            name: {
                "coefficient": float(model.params[name]),
                "std_err": float(model.bse[name]),
                "p_value": float(model.pvalues[name]),
            }
            for name in model.params.index
        },
    }

    # Economic significance
    interact_coef = model.params.get("subst_x_vol", 0)
    results["economic_significance"] = {
        "interaction_coefficient": float(interact_coef),
        "pct_impact_pp": float(interact_coef * 100),
        "interpretation": (
            f"When volume is 1σ above mean AND substitution is 1σ above mean, "
            f"BTC monthly return is {interact_coef*100:.2f}pp "
            f"{'lower' if interact_coef < 0 else 'higher'}"
        ),
    }

    return results


def _run_stress_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run robustness and stress tests.

    Key fix: Falsification test now uses symmetric functional form
    (same model on S&P returns vs BTC returns).
    """
    results = {}

    required = ["btc_return", "sp500_return", "substitution_z", "volume_z"]
    if not all(col in df.columns for col in required):
        return results

    df_clean = df.dropna(subset=required).copy()
    if len(df_clean) < 20:
        return results

    # Create interaction
    df_clean["subst_x_vol"] = df_clean["substitution_z"] * df_clean["volume_z"]

    high_vol = df_clean[df_clean["high_volume"] == 1].copy()
    if len(high_vol) < 10:
        return results

    actual_corr = float(high_vol["substitution_z"].corr(high_vol["btc_return"]))

    # 1. Bootstrap confidence interval
    n_boot = 1000
    np.random.seed(42)
    boot_corrs = []
    for _ in range(n_boot):
        sample = high_vol.sample(n=len(high_vol), replace=True)
        boot_corrs.append(sample["substitution_z"].corr(sample["btc_return"]))

    boot_corrs = np.array(boot_corrs)
    ci_low, ci_high = np.percentile(boot_corrs, [2.5, 97.5])

    results["bootstrap"] = {
        "actual_correlation": actual_corr,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "ci_excludes_zero": bool(ci_high < 0 or ci_low > 0),
    }

    # 2. Placebo test
    n_placebo = 1000
    placebo_corrs = []
    for _ in range(n_placebo):
        df_placebo = df_clean.copy()
        df_placebo["substitution_z"] = np.random.permutation(df_placebo["substitution_z"].values)
        hv = df_placebo[df_placebo["high_volume"] == 1]
        placebo_corrs.append(hv["substitution_z"].corr(hv["btc_return"]))

    placebo_pvalue = float((np.array(placebo_corrs) <= actual_corr).mean())

    results["placebo_test"] = {
        "actual_correlation": actual_corr,
        "placebo_p_value": placebo_pvalue,
        "passes": placebo_pvalue < 0.05,
    }

    # 3. Leave-one-out
    loo_corrs = []
    for i in high_vol.index:
        subset = high_vol.drop(i)
        loo_corrs.append(subset["substitution_z"].corr(subset["btc_return"]))

    loo_corrs = np.array(loo_corrs)

    results["leave_one_out"] = {
        "mean": float(loo_corrs.mean()),
        "std": float(loo_corrs.std()),
        "min": float(loo_corrs.min()),
        "max": float(loo_corrs.max()),
        "all_negative": bool((loo_corrs < 0).all()),
    }

    # 4. SYMMETRIC FALSIFICATION TEST
    # Run the SAME model on S&P returns as dependent variable
    # If the effect is BTC-specific, it should NOT appear for S&P
    X_btc = sm.add_constant(df_clean[["substitution_z", "volume_z", "subst_x_vol"]])
    y_btc = df_clean["btc_return"]
    model_btc = sm.OLS(y_btc, X_btc).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    X_sp = sm.add_constant(df_clean[["substitution_z", "volume_z", "subst_x_vol"]])
    y_sp = df_clean["sp500_return"]
    model_sp = sm.OLS(y_sp, X_sp).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    results["falsification_symmetric"] = {
        "btc_interaction_coef": float(model_btc.params.get("subst_x_vol", np.nan)),
        "btc_interaction_pvalue": float(model_btc.pvalues.get("subst_x_vol", np.nan)),
        "sp500_interaction_coef": float(model_sp.params.get("subst_x_vol", np.nan)),
        "sp500_interaction_pvalue": float(model_sp.pvalues.get("subst_x_vol", np.nan)),
        "passes": model_sp.pvalues.get("subst_x_vol", 0) > 0.1,
        "note": "Same model on BTC vs S&P returns - effect should be BTC-specific",
    }

    return results


def _save_results(results: Dict[str, Any], df: pd.DataFrame, output_dir: Path) -> None:
    """Save analysis results and data."""
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

    df.to_csv(output_dir / "cash_substitution_data.csv", index=False)


def _generate_report(results: Dict[str, Any]) -> str:
    """Generate text report of analysis results."""
    lines = []
    lines.append("=" * 80)
    lines.append("CASH SUBSTITUTION ANALYSIS (METHODOLOGICALLY CORRECTED)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Methodology notes:")
    lines.append("- Primary spec uses log returns (stationary)")
    lines.append("- Single regression with controls (no two-step residualization)")
    lines.append("- Variables z-scored before forming interactions")
    lines.append("- Lagged specs (1-3mo) for causal inference")
    lines.append("")

    # Data summary
    if "data_summary" in results:
        lines.append("DATA SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Observations: {results['data_summary'].get('n_observations', 'N/A')}")
        lines.append(f"Date range: {results['data_summary'].get('date_range', 'N/A')}")
        lines.append("")

    # Stationarity tests
    if "stationarity" in results:
        lines.append("STATIONARITY TESTS (ADF)")
        lines.append("-" * 40)
        for var, test in results["stationarity"].items():
            status = "STATIONARY" if test.get("stationary") else "NON-STATIONARY"
            lines.append(f"  {test.get('label', var)}: p={test.get('p_value', 1):.4f} [{status}]")
        lines.append("")

    # Primary returns model
    if "primary_returns_model" in results:
        prm = results["primary_returns_model"]
        lines.append("PRIMARY SPECIFICATION: Returns Regression")
        lines.append("-" * 40)
        if "model_summary" in prm:
            lines.append(f"R²: {prm['model_summary'].get('r_squared', 0):.4f}")
            lines.append(f"N: {prm['model_summary'].get('n_obs', 0)}")
            lines.append(f"DW: {prm['model_summary'].get('durbin_watson', 0):.2f}")
        if "coefficients" in prm:
            lines.append("\nCoefficients:")
            for name, coef in prm["coefficients"].items():
                sig = "***" if coef["p_value"] < 0.01 else "**" if coef["p_value"] < 0.05 else "*" if coef["p_value"] < 0.1 else ""
                lines.append(f"  {name}: {coef['coef']:.4f} (se={coef['std_err']:.4f}, p={coef['p_value']:.4f}) {sig}")
        if "interpretation" in prm:
            lines.append(f"\n{prm['interpretation'].get('description', '')}")
        lines.append("")

    # Lagged models
    if "lagged_models" in results:
        lm = results["lagged_models"]
        lines.append("LAGGED SPECIFICATIONS (Causal)")
        lines.append("-" * 40)
        for lag_key in ["lag_1", "lag_2", "lag_3"]:
            if lag_key in lm:
                lag_data = lm[lag_key]
                sig = "***" if lag_data.get("interaction_pvalue", 1) < 0.01 else "**" if lag_data.get("interaction_pvalue", 1) < 0.05 else "*" if lag_data.get("interaction_pvalue", 1) < 0.1 else ""
                lines.append(f"  {lag_key}: interaction={lag_data.get('interaction_coef', 0):.4f} "
                           f"(p={lag_data.get('interaction_pvalue', 1):.4f}) {sig}")
        if "granger_causality" in lm:
            lines.append("\n  Granger Causality (Subst -> BTC):")
            for lag, gc in lm["granger_causality"].items():
                sig = "***" if gc["p_value"] < 0.01 else "**" if gc["p_value"] < 0.05 else "*" if gc["p_value"] < 0.1 else ""
                lines.append(f"    Lag {lag}: F={gc['f_stat']:.2f}, p={gc['p_value']:.4f} {sig}")
        lines.append("")

    # Volume interaction
    if "volume_interaction" in results:
        vi = results["volume_interaction"]
        lines.append("VOLUME INTERACTION (Centered Variables)")
        lines.append("-" * 40)

        for key in ["correlation_low_volume", "correlation_high_volume"]:
            if key in vi:
                data = vi[key]
                sig = "***" if data["p_value"] < 0.01 else "**" if data["p_value"] < 0.05 else "*" if data["p_value"] < 0.1 else ""
                lines.append(f"  {data['label']}: r={data['correlation']:.4f} (p={data['p_value']:.4f}) {sig}")

        if "interaction_model" in vi:
            model = vi["interaction_model"]
            lines.append(f"\n  Model R²: {model['r_squared']:.4f}")
            interact = model["coefficients"].get("subst_x_vol", {})
            sig = "***" if interact.get("p_value", 1) < 0.01 else "**" if interact.get("p_value", 1) < 0.05 else "*" if interact.get("p_value", 1) < 0.1 else ""
            lines.append(f"  Interaction: {interact.get('coefficient', 0):.4f} (p={interact.get('p_value', 1):.4f}) {sig}")

        if "economic_significance" in vi:
            lines.append(f"\n  {vi['economic_significance']['interpretation']}")
        lines.append("")

    # Stress tests
    if "stress_tests" in results:
        st = results["stress_tests"]
        lines.append("STRESS TESTS")
        lines.append("-" * 40)

        if "bootstrap" in st:
            lines.append(f"  Bootstrap 95% CI: [{st['bootstrap']['ci_95_low']:.4f}, {st['bootstrap']['ci_95_high']:.4f}]")
            lines.append(f"    Excludes zero: {st['bootstrap']['ci_excludes_zero']}")

        if "placebo_test" in st:
            lines.append(f"  Placebo p-value: {st['placebo_test']['placebo_p_value']:.4f}")
            lines.append(f"    Passes: {st['placebo_test']['passes']}")

        if "leave_one_out" in st:
            lines.append(f"  Leave-one-out: all negative = {st['leave_one_out']['all_negative']}")

        if "falsification_symmetric" in st:
            fs = st["falsification_symmetric"]
            lines.append(f"\n  SYMMETRIC FALSIFICATION (same model, different DV):")
            lines.append(f"    BTC interaction: {fs['btc_interaction_coef']:.4f} (p={fs['btc_interaction_pvalue']:.4f})")
            lines.append(f"    S&P interaction: {fs['sp500_interaction_coef']:.4f} (p={fs['sp500_interaction_pvalue']:.4f})")
            lines.append(f"    Passes (S&P not significant): {fs['passes']}")

        lines.append("")

    # Descriptive regime analysis
    if "regime_analysis" in results:
        ra = results["regime_analysis"]
        lines.append("DESCRIPTIVE: Cash/Drug Ratio Trend (levels - context only)")
        lines.append("-" * 40)
        if "ratio_trend" in ra:
            trend = ra["ratio_trend"]
            lines.append(f"  Monthly change: {trend.get('monthly_pct_change', 0):.2f}%")
            lines.append(f"  Declining: {trend.get('is_declining', False)}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.1")
    lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_cash_substitution_analysis()
