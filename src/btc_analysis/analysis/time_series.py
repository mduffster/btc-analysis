"""Time series analysis for BTC price and criminal enterprise correlation.

Tests the hypothesis that Bitcoin price is driven by demand from criminal
enterprises needing alternative payment/settlement infrastructure.

Proxies for criminal enterprise demand:
- Drug overdose deaths (demand-side indicator)
- Drug seizures (supply/trafficking indicator)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

from btc_analysis.config import get_config
from btc_analysis.processing.merge_timeseries import merge_timeseries, get_analysis_ready_data

logger = logging.getLogger(__name__)


def run_time_series_analysis(
    output_dir: Optional[Path] = None,
    start_year: int = 2017,
    end_year: int = 2024,
) -> Dict[str, Any]:
    """
    Run complete time series analysis of BTC price vs criminal enterprise proxies.

    Analysis includes:
    1. Stationarity tests (ADF)
    2. Granger causality tests
    3. OLS regression with controls
    4. VAR model (if appropriate)

    Args:
        output_dir: Directory to save outputs.
        start_year: Analysis start year.
        end_year: Analysis end year.

    Returns:
        Dictionary containing all analysis results.
    """
    config = get_config()

    if output_dir is None:
        output_dir = config.paths.outputs_dir / "timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading time series data...")
    df = get_analysis_ready_data(start_year=start_year, end_year=end_year)

    if df.empty:
        logger.error("No data available for analysis")
        return {}

    logger.info(f"Analysis dataset: {len(df)} observations")

    results = {
        "data_summary": _summarize_data(df),
        "stationarity_tests": {},
        "granger_tests": {},
        "regression_models": {},
    }

    # Step 1: Stationarity tests
    logger.info("\nRunning stationarity tests...")
    results["stationarity_tests"] = _run_stationarity_tests(df)

    # Step 2: Granger causality tests
    logger.info("\nRunning Granger causality tests...")
    results["granger_tests"] = _run_granger_tests(df)

    # Step 3: OLS regressions
    logger.info("\nRunning regression analysis...")
    results["regression_models"] = _run_regressions(df)

    # Step 4: Drug market index models (cleaner specification)
    logger.info("\nRunning drug market index models...")
    results["index_models"] = _run_index_regressions(df)

    # Step 4: Save results
    _save_results(results, output_dir)

    # Step 5: Generate report
    report = _generate_report(results)
    report_path = output_dir / "time_series_analysis.txt"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"\nResults saved to {output_dir}")

    return results


def _summarize_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for the dataset."""
    summary = {
        "n_observations": len(df),
        "date_range": f"{df['year'].min()}-{df['month'].min():02d} to "
                      f"{df['year'].max()}-{df['month'].max():02d}",
        "variables": {},
    }

    # Key variables to summarize
    key_vars = [
        "btc_price_close",
        "btc_log_return",
        "btc_price_volatility",
    ]

    # Add overdose columns if present
    overdose_cols = [c for c in df.columns if "overdose" in c.lower() and "lag" not in c]
    key_vars.extend(overdose_cols[:3])

    # Add seizure columns if present
    seizure_cols = [c for c in df.columns if any(drug in c.lower() for drug in
                                                   ["cocaine", "fentanyl", "heroin", "meth", "seizure"])
                    and "lag" not in c]
    key_vars.extend(seizure_cols[:3])

    for var in key_vars:
        if var in df.columns:
            col = df[var].dropna()
            summary["variables"][var] = {
                "n": len(col),
                "mean": col.mean(),
                "std": col.std(),
                "min": col.min(),
                "max": col.max(),
            }

    return summary


def _run_stationarity_tests(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Run Augmented Dickey-Fuller tests for stationarity.

    Tests whether time series need differencing for valid regression.
    """
    tests = {}

    # Variables to test
    test_vars = ["btc_price_close", "btc_log_return"]

    # Add overdose/seizure variables
    for col in df.columns:
        if any(term in col.lower() for term in ["overdose", "cocaine", "fentanyl", "seizure"]):
            if "lag" not in col and "pct" not in col:
                test_vars.append(col)

    for var in test_vars:
        if var not in df.columns:
            continue

        series = df[var].dropna()
        if len(series) < 20:
            continue

        try:
            adf_result = adfuller(series, autolag="AIC")
            tests[var] = {
                "adf_statistic": adf_result[0],
                "p_value": adf_result[1],
                "lags_used": adf_result[2],
                "n_obs": adf_result[3],
                "critical_values": adf_result[4],
                "is_stationary": adf_result[1] < 0.05,
            }
        except Exception as e:
            logger.warning(f"ADF test failed for {var}: {e}")

    return tests


def _run_granger_tests(df: pd.DataFrame, max_lags: int = 6) -> Dict[str, Dict]:
    """
    Run Granger causality tests.

    Tests whether criminal enterprise proxies Granger-cause BTC price changes.
    """
    results = {}

    # Use log returns (stationary series)
    btc_var = "btc_log_return"
    if btc_var not in df.columns:
        logger.warning("BTC log returns not available for Granger test")
        return results

    # Test variables (potential causes of BTC price)
    test_vars = []
    for col in df.columns:
        if any(term in col.lower() for term in ["overdose", "cocaine", "fentanyl", "heroin", "seizure"]):
            if "lag" not in col:
                test_vars.append(col)

    for var in test_vars:
        if var not in df.columns:
            continue

        # Create clean dataset for this pair
        test_df = df[[btc_var, var]].dropna()

        if len(test_df) < max_lags * 3:
            logger.warning(f"Insufficient data for Granger test: {var}")
            continue

        try:
            # Test if var Granger-causes btc_var
            granger_result = grangercausalitytests(
                test_df[[btc_var, var]],
                maxlag=max_lags,
                verbose=False,
            )

            # Extract p-values for each lag
            lag_results = {}
            for lag in range(1, max_lags + 1):
                if lag in granger_result:
                    f_test = granger_result[lag][0]["ssr_ftest"]
                    lag_results[lag] = {
                        "f_statistic": f_test[0],
                        "p_value": f_test[1],
                    }

            # Find best lag (lowest p-value)
            best_lag = min(lag_results.keys(), key=lambda k: lag_results[k]["p_value"])

            results[f"{var} -> {btc_var}"] = {
                "lag_results": lag_results,
                "best_lag": best_lag,
                "best_p_value": lag_results[best_lag]["p_value"],
                "significant": lag_results[best_lag]["p_value"] < 0.05,
            }

        except Exception as e:
            logger.warning(f"Granger test failed for {var}: {e}")

    return results


def _run_regressions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run OLS regression models testing BTC price drivers.

    Models:
    1. Simple: BTC ~ overdose_deaths + seizures
    2. With controls: + S&P 500, VIX, halving cycle
    3. Log-log specification
    """
    models = {}

    # Model 1: Simple regression on levels
    logger.info("Running Model 1: Simple OLS on BTC price levels")
    models["simple_levels"] = _run_simple_regression(df, use_logs=False)

    # Model 2: Log-log specification (elasticities)
    logger.info("Running Model 2: Log-log specification")
    models["log_log"] = _run_simple_regression(df, use_logs=True)

    # Model 3: With market controls
    logger.info("Running Model 3: With market controls")
    models["with_controls"] = _run_controlled_regression(df)

    # Model 4: Differenced (changes)
    logger.info("Running Model 4: First differences")
    models["first_diff"] = _run_differenced_regression(df)

    return models


def _run_simple_regression(df: pd.DataFrame, use_logs: bool = False) -> Dict[str, Any]:
    """
    Run simple OLS regression: BTC ~ criminal_proxies.

    Args:
        df: Analysis DataFrame.
        use_logs: Whether to use log transformation.

    Returns:
        Dictionary with regression results.
    """
    # Identify available regressors
    crime_vars = []
    for col in df.columns:
        if any(term in col.lower() for term in ["overdose", "cocaine", "fentanyl", "heroin", "seizure"]):
            if "lag" not in col and "pct" not in col:
                crime_vars.append(col)

    if not crime_vars:
        logger.warning("No crime variables available for regression")
        return {}

    # Prepare dependent variable
    y_col = "btc_price_close"
    if y_col not in df.columns:
        return {}

    # Build regression DataFrame
    reg_cols = [y_col] + crime_vars
    reg_df = df[reg_cols].dropna()

    if len(reg_df) < 30:
        logger.warning("Insufficient observations for regression")
        return {}

    # Apply log transformation if requested
    if use_logs:
        for col in reg_cols:
            if (reg_df[col] > 0).all():
                reg_df[col] = np.log(reg_df[col])
            else:
                # Can't log non-positive values
                logger.warning(f"Cannot log-transform {col} (contains non-positive values)")

    y = reg_df[y_col]
    X = reg_df[crime_vars]
    X = sm.add_constant(X)

    try:
        model = OLS(y, X).fit(cov_type="HC1")  # Robust standard errors

        return {
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "f_statistic": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "n_obs": model.nobs,
            "durbin_watson": durbin_watson(model.resid),
            "coefficients": {
                name: {
                    "coef": model.params[name],
                    "se": model.bse[name],
                    "t_stat": model.tvalues[name],
                    "p_value": model.pvalues[name],
                }
                for name in model.params.index
            },
            "summary": model.summary().as_text(),
        }

    except Exception as e:
        logger.error(f"Regression failed: {e}")
        return {}


def _run_controlled_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run regression with market control variables.

    Controls: S&P 500, VIX, halving cycle, time trend.
    """
    # Crime variables
    crime_vars = []
    for col in df.columns:
        if any(term in col.lower() for term in ["overdose", "cocaine", "fentanyl", "heroin", "seizure"]):
            if "lag" not in col and "pct" not in col:
                crime_vars.append(col)

    # Control variables
    control_vars = []
    for control in ["sp500_close", "vix_close", "dxy_close", "halving_cycle", "months_since_halving"]:
        if control in df.columns:
            control_vars.append(control)

    # Add time trend
    df = df.copy()
    df["time_trend"] = range(len(df))
    control_vars.append("time_trend")

    # Build regression
    y_col = "btc_price_close"
    if y_col not in df.columns:
        return {}

    all_vars = [y_col] + crime_vars + control_vars
    reg_df = df[[c for c in all_vars if c in df.columns]].dropna()

    if len(reg_df) < 30:
        logger.warning("Insufficient observations for controlled regression")
        return {}

    y = reg_df[y_col]
    X_vars = [c for c in crime_vars + control_vars if c in reg_df.columns]
    X = reg_df[X_vars]
    X = sm.add_constant(X)

    try:
        model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

        return {
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "f_statistic": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "n_obs": model.nobs,
            "durbin_watson": durbin_watson(model.resid),
            "coefficients": {
                name: {
                    "coef": model.params[name],
                    "se": model.bse[name],
                    "t_stat": model.tvalues[name],
                    "p_value": model.pvalues[name],
                }
                for name in model.params.index
            },
            "summary": model.summary().as_text(),
        }

    except Exception as e:
        logger.error(f"Controlled regression failed: {e}")
        return {}


def _run_differenced_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run regression on first differences (changes).

    ΔBtc ~ ΔOverdose + ΔSeizures
    """
    df = df.copy().sort_values(["year", "month"])

    # Create first differences
    diff_cols = {}

    # BTC
    if "btc_price_close" in df.columns:
        df["d_btc"] = df["btc_price_close"].diff()
        diff_cols["d_btc"] = "btc_price_close"

    # Crime variables
    for col in df.columns:
        if any(term in col.lower() for term in ["overdose", "cocaine", "fentanyl", "heroin", "seizure"]):
            if "lag" not in col and "pct" not in col and not col.startswith("d_"):
                df[f"d_{col}"] = df[col].diff()
                diff_cols[f"d_{col}"] = col

    if "d_btc" not in df.columns or len(diff_cols) < 2:
        return {}

    # Run regression
    y_col = "d_btc"
    x_cols = [c for c in diff_cols.keys() if c != y_col]

    reg_df = df[[y_col] + x_cols].dropna()

    if len(reg_df) < 30:
        return {}

    y = reg_df[y_col]
    X = reg_df[x_cols]
    X = sm.add_constant(X)

    try:
        model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

        return {
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "f_statistic": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "n_obs": model.nobs,
            "durbin_watson": durbin_watson(model.resid),
            "coefficients": {
                name: {
                    "coef": model.params[name],
                    "se": model.bse[name],
                    "t_stat": model.tvalues[name],
                    "p_value": model.pvalues[name],
                }
                for name in model.params.index
            },
            "summary": model.summary().as_text(),
        }

    except Exception as e:
        logger.error(f"Differenced regression failed: {e}")
        return {}


def _run_index_regressions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run regressions using the aggregate drug market index.

    Models:
    1. BTC ~ drug_market_index
    2. BTC ~ drug_market_index + controls
    3. log(BTC) ~ log(index)
    4. d_BTC ~ d_index (first differences)

    Args:
        df: Analysis DataFrame with drug_market_index column.

    Returns:
        Dictionary with model results.
    """
    models = {}

    # Check if index exists
    if "drug_market_index" not in df.columns:
        logger.warning("drug_market_index not found in data")
        return models

    # Model 1: Simple regression - BTC ~ drug_market_index
    logger.info("Running Index Model 1: Simple BTC ~ drug_market_index")
    y_col = "btc_price_close"
    x_cols = ["drug_market_index"]

    if "drug_market_index_simple" in df.columns:
        x_cols.append("drug_market_index_simple")

    reg_df = df[[y_col] + x_cols].dropna()

    if len(reg_df) >= 20:
        y = reg_df[y_col]
        X = sm.add_constant(reg_df[x_cols])

        try:
            model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            models["index_simple"] = {
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "n_obs": model.nobs,
                "durbin_watson": durbin_watson(model.resid),
                "coefficients": {
                    name: {
                        "coef": model.params[name],
                        "se": model.bse[name],
                        "t_stat": model.tvalues[name],
                        "p_value": model.pvalues[name],
                    }
                    for name in model.params.index
                },
                "summary": model.summary().as_text(),
            }
        except Exception as e:
            logger.warning(f"Index simple model failed: {e}")

    # Model 2: With market controls
    logger.info("Running Index Model 2: With market controls")
    control_cols = []
    for col in ["sp500_close", "vix_close", "halving_cycle", "time_trend"]:
        if col in df.columns:
            control_cols.append(col)

    if "time_trend" not in df.columns:
        df = df.copy()
        df["time_trend"] = range(len(df))
        control_cols.append("time_trend")

    x_cols_ctrl = ["drug_market_index"] + control_cols
    reg_df = df[[y_col] + x_cols_ctrl].dropna()

    if len(reg_df) >= 20:
        y = reg_df[y_col]
        X = sm.add_constant(reg_df[x_cols_ctrl])

        try:
            model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            models["index_with_controls"] = {
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "n_obs": model.nobs,
                "durbin_watson": durbin_watson(model.resid),
                "coefficients": {
                    name: {
                        "coef": model.params[name],
                        "se": model.bse[name],
                        "t_stat": model.tvalues[name],
                        "p_value": model.pvalues[name],
                    }
                    for name in model.params.index
                },
                "summary": model.summary().as_text(),
            }
        except Exception as e:
            logger.warning(f"Index with controls model failed: {e}")

    # Model 3: Log-log specification
    logger.info("Running Index Model 3: Log-log")
    if "drug_market_index_simple" in df.columns:
        # Use simple index (always positive) for log
        reg_df = df[[y_col, "drug_market_index_simple"]].dropna()
        reg_df = reg_df[reg_df["drug_market_index_simple"] > 0]

        if len(reg_df) >= 20:
            y = np.log(reg_df[y_col])
            X = sm.add_constant(np.log(reg_df["drug_market_index_simple"]))

            try:
                model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
                models["index_log_log"] = {
                    "r_squared": model.rsquared,
                    "adj_r_squared": model.rsquared_adj,
                    "n_obs": model.nobs,
                    "durbin_watson": durbin_watson(model.resid),
                    "coefficients": {
                        name: {
                            "coef": model.params[name],
                            "se": model.bse[name],
                            "t_stat": model.tvalues[name],
                            "p_value": model.pvalues[name],
                        }
                        for name in model.params.index
                    },
                    "summary": model.summary().as_text(),
                    "interpretation": "Coefficient is elasticity: 1% change in index -> X% change in BTC price",
                }
            except Exception as e:
                logger.warning(f"Index log-log model failed: {e}")

    # Model 4: First differences
    logger.info("Running Index Model 4: First differences")
    df = df.copy().sort_values(["year", "month"])
    df["d_btc"] = df["btc_price_close"].diff()
    df["d_index"] = df["drug_market_index"].diff()

    if "drug_market_index_simple" in df.columns:
        df["d_index_simple"] = df["drug_market_index_simple"].diff()

    reg_df = df[["d_btc", "d_index"]].dropna()

    if len(reg_df) >= 20:
        y = reg_df["d_btc"]
        X = sm.add_constant(reg_df["d_index"])

        try:
            model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            models["index_first_diff"] = {
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "n_obs": model.nobs,
                "durbin_watson": durbin_watson(model.resid),
                "coefficients": {
                    name: {
                        "coef": model.params[name],
                        "se": model.bse[name],
                        "t_stat": model.tvalues[name],
                        "p_value": model.pvalues[name],
                    }
                    for name in model.params.index
                },
                "summary": model.summary().as_text(),
            }
        except Exception as e:
            logger.warning(f"Index first diff model failed: {e}")

    # Model 5: Granger test on index
    logger.info("Running Granger test on drug_market_index")
    if "btc_log_return" in df.columns and "drug_market_index" in df.columns:
        test_df = df[["btc_log_return", "drug_market_index"]].dropna()

        if len(test_df) >= 20:
            try:
                granger_result = grangercausalitytests(
                    test_df[["btc_log_return", "drug_market_index"]],
                    maxlag=6,
                    verbose=False,
                )

                # Find best lag
                lag_pvals = {
                    lag: granger_result[lag][0]["ssr_ftest"][1]
                    for lag in range(1, 7)
                }
                best_lag = min(lag_pvals.keys(), key=lambda k: lag_pvals[k])

                models["index_granger"] = {
                    "test": "drug_market_index -> btc_log_return",
                    "best_lag": best_lag,
                    "best_p_value": lag_pvals[best_lag],
                    "significant": lag_pvals[best_lag] < 0.05,
                    "all_lags": lag_pvals,
                }
            except Exception as e:
                logger.warning(f"Index Granger test failed: {e}")

    return models


def _save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save analysis results to files."""
    import json

    # Save JSON summary (excluding full statsmodels summaries)
    json_safe = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_safe[key] = {}
            for k, v in value.items():
                if isinstance(v, dict) and "summary" in v:
                    v_copy = {kk: vv for kk, vv in v.items() if kk != "summary"}
                    json_safe[key][k] = v_copy
                else:
                    json_safe[key][k] = v
        else:
            json_safe[key] = value

    # Convert numpy types for JSON serialization
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

    json_safe = convert_types(json_safe)

    with open(output_dir / "results.json", "w") as f:
        json.dump(json_safe, f, indent=2)


def _generate_report(results: Dict[str, Any]) -> str:
    """Generate a text report of the analysis results."""
    lines = []
    lines.append("=" * 80)
    lines.append("TIME SERIES ANALYSIS: BTC PRICE AND CRIMINAL ENTERPRISE PROXIES")
    lines.append("=" * 80)
    lines.append("")

    # Data summary
    if "data_summary" in results:
        summary = results["data_summary"]
        lines.append("DATA SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Observations: {summary.get('n_observations', 'N/A')}")
        lines.append(f"Date range: {summary.get('date_range', 'N/A')}")
        lines.append("")

    # Stationarity tests
    if "stationarity_tests" in results:
        lines.append("STATIONARITY TESTS (Augmented Dickey-Fuller)")
        lines.append("-" * 40)
        for var, test in results["stationarity_tests"].items():
            status = "STATIONARY" if test.get("is_stationary") else "NON-STATIONARY"
            lines.append(f"{var}: ADF={test.get('adf_statistic', 0):.3f}, "
                        f"p={test.get('p_value', 1):.4f} [{status}]")
        lines.append("")

    # Granger causality
    if "granger_tests" in results:
        lines.append("GRANGER CAUSALITY TESTS")
        lines.append("-" * 40)
        for test_name, test in results["granger_tests"].items():
            sig = "***" if test.get("significant") else ""
            lines.append(f"{test_name}: p={test.get('best_p_value', 1):.4f} "
                        f"(lag={test.get('best_lag', 0)}) {sig}")
        lines.append("")

    # Regression models
    if "regression_models" in results:
        lines.append("REGRESSION MODELS")
        lines.append("-" * 40)
        for model_name, model in results["regression_models"].items():
            if not model:
                continue
            lines.append(f"\n{model_name.upper()}")
            lines.append(f"R² = {model.get('r_squared', 0):.4f}, "
                        f"Adj R² = {model.get('adj_r_squared', 0):.4f}")
            lines.append(f"N = {model.get('n_obs', 0)}, "
                        f"DW = {model.get('durbin_watson', 0):.3f}")

            if "coefficients" in model:
                lines.append("\nCoefficients:")
                for var, coef in model["coefficients"].items():
                    sig = ""
                    p = coef.get("p_value", 1)
                    if p < 0.01:
                        sig = "***"
                    elif p < 0.05:
                        sig = "**"
                    elif p < 0.1:
                        sig = "*"
                    lines.append(f"  {var}: {coef.get('coef', 0):.4f} "
                                f"(se={coef.get('se', 0):.4f}, p={p:.4f}) {sig}")
            lines.append("")

    lines.append("=" * 80)
    lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.1")
    lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_time_series_analysis()
