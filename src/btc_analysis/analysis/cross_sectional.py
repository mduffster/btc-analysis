"""Phase 1: Cross-sectional and panel regression analysis of BTC premiums."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from btc_analysis.config import get_config
from btc_analysis.processing.merge import clean_panel, create_cross_section, merge_datasets

logger = logging.getLogger(__name__)


def run_phase1_analysis(
    output_dir: Optional[Path] = None,
    year: Optional[int] = None,
) -> dict:
    """
    Run Phase 1 cross-sectional regression analysis.

    Tests the hypothesis that BTC premiums are higher in countries with
    stricter capital controls, controlling for political stability,
    inflation, GDP per capita, and internet penetration.

    Regression specification:
    BTC_Premium_i = β0 + β1*CapitalControls_i + β2*PoliticalStability_i
                  + β3*Inflation_i + β4*log(GDP_pc)_i + β5*InternetPen_i + ε_i

    Args:
        output_dir: Directory to save outputs (tables, plots).
        year: Specific year for cross-section. If None, uses time averages.

    Returns:
        Dictionary containing regression results and diagnostics.
    """
    config = get_config()

    if output_dir is None:
        output_dir = config.paths.outputs_dir / "phase1"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 1 cross-sectional analysis...")

    # Load and prepare data
    panel = merge_datasets()

    if panel.empty:
        logger.error("No data available for analysis")
        return {"error": "No data available"}

    # Clean panel data
    panel = clean_panel(
        panel,
        min_obs=1,
        required_vars=[],  # Be flexible about required vars initially
    )

    # Create cross-section
    cross_section = create_cross_section(panel, year=year, agg_method="mean")

    logger.info(f"Cross-section contains {len(cross_section)} countries")

    # Generate summary statistics
    summary_stats = generate_summary_statistics(cross_section, output_dir)

    # Generate correlation matrix
    corr_matrix = generate_correlation_matrix(cross_section, output_dir)

    # Run main regression
    reg_results = run_main_regression(cross_section, output_dir)

    # Run robustness checks
    robustness = run_robustness_checks(cross_section, output_dir)

    # Run interaction regression
    interaction_results = run_interaction_regression(cross_section, output_dir)

    # Generate coefficient plot
    if reg_results.get("model") is not None:
        generate_coefficient_plot(reg_results["model"], output_dir)

    # Generate interaction coefficient plot
    if interaction_results.get("model") is not None:
        generate_coefficient_plot(
            interaction_results["model"],
            output_dir,
        )
        # Rename to avoid overwriting
        import shutil
        coef_plot = output_dir / "coefficient_plot.png"
        if coef_plot.exists():
            shutil.copy(coef_plot, output_dir / "interaction_coefficient_plot.png")

    # Compile all results
    results = {
        "summary_stats": summary_stats,
        "correlation_matrix": corr_matrix,
        "main_regression": reg_results,
        "interaction_regression": interaction_results,
        "robustness": robustness,
        "n_countries": len(cross_section),
        "year": year if year else "averaged",
    }

    # Save results summary
    save_results_summary(results, output_dir)

    return results


def generate_summary_statistics(
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Generate and save summary statistics table."""
    # Select analysis variables
    analysis_vars = [
        "btc_premium",
        "capital_control_index",
        "kaopen",
        "political_stability",
        "rule_of_law",
        "regulatory_quality",
        "control_of_corruption",
        "gdp_per_capita",
        "internet_penetration",
        "inflation_wb",
    ]

    # Filter to available variables
    available_vars = [v for v in analysis_vars if v in df.columns]

    if not available_vars:
        logger.warning("No analysis variables available for summary statistics")
        return pd.DataFrame()

    subset = df[available_vars]

    # Calculate summary statistics
    stats = subset.describe().T
    stats["missing"] = subset.isnull().sum()
    stats["missing_pct"] = (stats["missing"] / len(df) * 100).round(1)

    # Save to CSV
    stats.to_csv(output_dir / "summary_statistics.csv")
    logger.info(f"Saved summary statistics to {output_dir / 'summary_statistics.csv'}")

    return stats


def generate_correlation_matrix(
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Generate correlation matrix and heatmap."""
    analysis_vars = [
        "btc_premium",
        "capital_control_index",
        "political_stability",
        "rule_of_law",
        "gdp_per_capita",
        "internet_penetration",
        "inflation_wb",
    ]

    available_vars = [v for v in analysis_vars if v in df.columns]

    if len(available_vars) < 2:
        logger.warning("Insufficient variables for correlation matrix")
        return pd.DataFrame()

    subset = df[available_vars].dropna()

    if len(subset) < 10:
        logger.warning(f"Only {len(subset)} observations with complete data for correlation")

    corr = subset.corr()

    # Save correlation matrix
    corr.to_csv(output_dir / "correlation_matrix.csv")

    # Generate heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
    )
    plt.title("Correlation Matrix: BTC Premium and Control Variables")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    logger.info(f"Saved correlation matrix and heatmap to {output_dir}")

    return corr


def run_main_regression(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Run the main OLS regression with robust standard errors.

    Model: btc_premium ~ capital_control_index + political_stability
           + inflation + log_gdp_pc + internet_penetration
    """
    # Define dependent and independent variables
    dep_var = "btc_premium"
    indep_vars = [
        "capital_control_index",
        "political_stability",
        "inflation_wb",
        "log_gdp_pc",
        "internet_penetration",
    ]

    # Check which variables are available
    if dep_var not in df.columns:
        logger.warning(f"Dependent variable '{dep_var}' not available")
        # Try alternative if available
        alt_dep = [c for c in df.columns if "premium" in c.lower()]
        if alt_dep:
            dep_var = alt_dep[0]
            logger.info(f"Using alternative dependent variable: {dep_var}")
        else:
            return {"error": f"Dependent variable not available. Columns: {list(df.columns)}"}

    available_indep = [v for v in indep_vars if v in df.columns]

    if not available_indep:
        logger.warning("No independent variables available")
        # Try alternatives
        alt_indep = []
        if "kaopen" in df.columns:
            alt_indep.append("kaopen")
        if "rule_of_law" in df.columns:
            alt_indep.append("rule_of_law")
        if alt_indep:
            available_indep = alt_indep
        else:
            return {"error": "No independent variables available"}

    logger.info(f"Running regression: {dep_var} ~ {' + '.join(available_indep)}")

    # Prepare regression data
    reg_vars = [dep_var] + available_indep
    reg_data = df[reg_vars].dropna()

    if len(reg_data) < len(available_indep) + 10:
        logger.warning(
            f"Insufficient observations ({len(reg_data)}) for reliable regression"
        )

    # Prepare X and y
    y = reg_data[dep_var]
    X = reg_data[available_indep]
    X = sm.add_constant(X)

    # Run OLS with heteroskedasticity-robust standard errors (HC3)
    model = sm.OLS(y, X).fit(cov_type="HC3")

    # Save regression output
    with open(output_dir / "regression_results.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 1: CROSS-SECTIONAL REGRESSION ANALYSIS\n")
        f.write("BTC Premium vs Capital Controls\n")
        f.write("=" * 80 + "\n\n")
        f.write(model.summary().as_text())
        f.write("\n\n")
        f.write("Notes:\n")
        f.write("- Standard errors are heteroskedasticity-robust (HC3)\n")
        f.write(f"- Number of observations: {len(reg_data)}\n")
        f.write(f"- Dependent variable: {dep_var}\n")

    logger.info(f"Saved regression results to {output_dir / 'regression_results.txt'}")

    # Run diagnostics
    diagnostics = run_regression_diagnostics(model, X, y, output_dir)

    return {
        "model": model,
        "n_obs": len(reg_data),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "std_errors": model.bse.to_dict(),
        "diagnostics": diagnostics,
    }


def run_regression_diagnostics(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> dict:
    """Run regression diagnostics."""
    diagnostics = {}

    # 1. Variance Inflation Factors (multicollinearity)
    try:
        vif_data = []
        for i, col in enumerate(X.columns):
            if col != "const":
                vif = variance_inflation_factor(X.values, i)
                vif_data.append({"variable": col, "VIF": vif})

        vif_df = pd.DataFrame(vif_data)
        vif_df.to_csv(output_dir / "vif.csv", index=False)
        diagnostics["vif"] = vif_df.to_dict("records")

        high_vif = vif_df[vif_df["VIF"] > 5]
        if not high_vif.empty:
            logger.warning(
                f"High VIF detected (>5): {high_vif['variable'].tolist()}"
            )
    except Exception as e:
        logger.warning(f"Could not compute VIF: {e}")

    # 2. Breusch-Pagan test for heteroskedasticity
    try:
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(model.resid, X)
        diagnostics["breusch_pagan"] = {
            "statistic": bp_stat,
            "pvalue": bp_pvalue,
            "heteroskedasticity_detected": bp_pvalue < 0.05,
        }
        if bp_pvalue < 0.05:
            logger.info(
                f"Breusch-Pagan test suggests heteroskedasticity (p={bp_pvalue:.4f}). "
                "Using robust standard errors."
            )
    except Exception as e:
        logger.warning(f"Could not compute Breusch-Pagan test: {e}")

    # 3. Residual plots
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Residuals vs fitted
        axes[0].scatter(model.fittedvalues, model.resid, alpha=0.5)
        axes[0].axhline(y=0, color="r", linestyle="--")
        axes[0].set_xlabel("Fitted Values")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Fitted Values")

        # Q-Q plot
        sm.qqplot(model.resid, line="45", ax=axes[1])
        axes[1].set_title("Q-Q Plot of Residuals")

        plt.tight_layout()
        plt.savefig(output_dir / "residual_diagnostics.png", dpi=150)
        plt.close()

        diagnostics["residual_plots"] = str(output_dir / "residual_diagnostics.png")
    except Exception as e:
        logger.warning(f"Could not generate residual plots: {e}")

    return diagnostics


def run_robustness_checks(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Run robustness checks on the main specification."""
    robustness = {}

    dep_var = "btc_premium" if "btc_premium" in df.columns else None
    if dep_var is None:
        alt_dep = [c for c in df.columns if "premium" in c.lower()]
        if alt_dep:
            dep_var = alt_dep[0]
        else:
            return {"error": "No dependent variable for robustness checks"}

    # 1. Alternative capital control measure (kaopen instead of normalized index)
    if "kaopen" in df.columns:
        indep_vars = ["kaopen"]
        if "political_stability" in df.columns:
            indep_vars.append("political_stability")
        if "log_gdp_pc" in df.columns:
            indep_vars.append("log_gdp_pc")

        reg_data = df[[dep_var] + indep_vars].dropna()
        if len(reg_data) > len(indep_vars) + 5:
            y = reg_data[dep_var]
            X = sm.add_constant(reg_data[indep_vars])
            model_alt = sm.OLS(y, X).fit(cov_type="HC3")
            robustness["alt_capital_control"] = {
                "n_obs": len(reg_data),
                "r_squared": model_alt.rsquared,
                "kaopen_coef": model_alt.params.get("kaopen"),
                "kaopen_pval": model_alt.pvalues.get("kaopen"),
            }

    # 2. Regression with rule_of_law instead of political_stability
    if "rule_of_law" in df.columns and "capital_control_index" in df.columns:
        indep_vars = ["capital_control_index", "rule_of_law"]
        if "log_gdp_pc" in df.columns:
            indep_vars.append("log_gdp_pc")

        reg_data = df[[dep_var] + indep_vars].dropna()
        if len(reg_data) > len(indep_vars) + 5:
            y = reg_data[dep_var]
            X = sm.add_constant(reg_data[indep_vars])
            model_alt = sm.OLS(y, X).fit(cov_type="HC3")
            robustness["alt_governance"] = {
                "n_obs": len(reg_data),
                "r_squared": model_alt.rsquared,
                "capital_control_coef": model_alt.params.get("capital_control_index"),
                "capital_control_pval": model_alt.pvalues.get("capital_control_index"),
            }

    # Save robustness results
    with open(output_dir / "robustness_checks.txt", "w") as f:
        f.write("ROBUSTNESS CHECKS\n")
        f.write("=" * 50 + "\n\n")
        for check_name, results in robustness.items():
            f.write(f"{check_name}:\n")
            for k, v in results.items():
                if isinstance(v, float):
                    f.write(f"  {k}: {v:.4f}\n")
                else:
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

    return robustness


def run_interaction_regression(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Run regression with interaction terms to capture combined effects.

    Tests whether the effect of capital controls on BTC premiums depends on:
    - Inflation level (capital_control × inflation)
    - Political stability (capital_control × political_stability)
    - Economic development (capital_control × log_gdp_pc)

    Model:
    btc_premium ~ capital_control + inflation + political_stability + log_gdp_pc
                + capital_control × inflation
                + capital_control × political_stability
    """
    dep_var = "btc_premium"

    if dep_var not in df.columns:
        return {"error": "Dependent variable not available"}

    # Prepare data - need capital_control_index and key variables
    required = ["capital_control_index", "inflation_wb", "political_stability", "log_gdp_pc"]
    available = [v for v in required if v in df.columns]

    if len(available) < 2:
        return {"error": "Insufficient variables for interaction model"}

    # Create interaction terms
    df_reg = df.copy()

    interactions_created = []

    # Capital controls × Inflation (key interaction)
    if "capital_control_index" in df.columns and "inflation_wb" in df.columns:
        df_reg["cc_x_inflation"] = df_reg["capital_control_index"] * df_reg["inflation_wb"]
        interactions_created.append("cc_x_inflation")

    # Capital controls × Political instability
    if "capital_control_index" in df.columns and "political_stability" in df.columns:
        df_reg["cc_x_instability"] = df_reg["capital_control_index"] * (-df_reg["political_stability"])
        interactions_created.append("cc_x_instability")

    # Capital controls × Low income (inverse of log GDP)
    if "capital_control_index" in df.columns and "log_gdp_pc" in df.columns:
        # Center log_gdp_pc for interpretability
        gdp_mean = df_reg["log_gdp_pc"].mean()
        df_reg["log_gdp_centered"] = df_reg["log_gdp_pc"] - gdp_mean
        df_reg["cc_x_low_income"] = df_reg["capital_control_index"] * (-df_reg["log_gdp_centered"])
        interactions_created.append("cc_x_low_income")

    if not interactions_created:
        return {"error": "Could not create interaction terms"}

    # Build regression with main effects and interactions
    main_effects = [v for v in available if v in df_reg.columns]
    all_vars = main_effects + interactions_created

    reg_data = df_reg[[dep_var] + all_vars].dropna()

    if len(reg_data) < len(all_vars) + 5:
        return {"error": f"Insufficient observations ({len(reg_data)}) for interaction model"}

    y = reg_data[dep_var]
    X = sm.add_constant(reg_data[all_vars])

    # Run OLS with robust standard errors
    model = sm.OLS(y, X).fit(cov_type="HC3")

    # Save results
    with open(output_dir / "interaction_regression.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("INTERACTION MODEL: Capital Controls × Context Variables\n")
        f.write("=" * 80 + "\n\n")
        f.write("Hypothesis: The effect of capital controls on BTC premiums is amplified\n")
        f.write("in high-inflation, politically unstable, or lower-income environments.\n\n")
        f.write("Interaction terms:\n")
        f.write("  - cc_x_inflation: capital_control_index × inflation_wb\n")
        f.write("  - cc_x_instability: capital_control_index × (-political_stability)\n")
        f.write("  - cc_x_low_income: capital_control_index × (-centered_log_gdp)\n\n")
        f.write(model.summary().as_text())
        f.write("\n\nInterpretation:\n")
        f.write("-" * 40 + "\n")

        # Interpret key coefficients
        if "cc_x_inflation" in model.params.index:
            coef = model.params["cc_x_inflation"]
            pval = model.pvalues["cc_x_inflation"]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            f.write(f"Capital Controls × Inflation: {coef:.4f} (p={pval:.4f}) {sig}\n")
            if coef > 0:
                f.write("  → Capital controls have STRONGER effect on BTC premium in high-inflation countries\n")
            else:
                f.write("  → Capital controls have WEAKER effect on BTC premium in high-inflation countries\n")

        if "cc_x_instability" in model.params.index:
            coef = model.params["cc_x_instability"]
            pval = model.pvalues["cc_x_instability"]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            f.write(f"\nCapital Controls × Instability: {coef:.4f} (p={pval:.4f}) {sig}\n")
            if coef > 0:
                f.write("  → Capital controls have STRONGER effect in politically unstable countries\n")
            else:
                f.write("  → Capital controls have WEAKER effect in politically unstable countries\n")

        if "cc_x_low_income" in model.params.index:
            coef = model.params["cc_x_low_income"]
            pval = model.pvalues["cc_x_low_income"]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            f.write(f"\nCapital Controls × Low Income: {coef:.4f} (p={pval:.4f}) {sig}\n")
            if coef > 0:
                f.write("  → Capital controls have STRONGER effect in lower-income countries\n")
            else:
                f.write("  → Capital controls have WEAKER effect in lower-income countries\n")

    logger.info(f"Saved interaction regression to {output_dir / 'interaction_regression.txt'}")

    # Generate interaction plot
    _plot_interaction_effects(df_reg, model, output_dir)

    return {
        "model": model,
        "n_obs": len(reg_data),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "interactions": interactions_created,
    }


def run_panel_regression(
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run panel regression with fixed effects on the full dataset.

    Uses country fixed effects to control for time-invariant unobserved
    heterogeneity, and optionally year fixed effects.

    Models estimated:
    1. Pooled OLS (baseline)
    2. Country Fixed Effects
    3. Country + Year Fixed Effects
    4. Fixed Effects with interactions

    Returns:
        Dictionary containing all model results.
    """
    config = get_config()

    if output_dir is None:
        output_dir = config.paths.outputs_dir / "phase1"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting panel regression analysis...")

    # Load full panel data (not cross-section averages)
    panel = merge_datasets()

    if panel.empty:
        return {"error": "No data available"}

    # Light cleaning - keep more observations
    panel = panel.dropna(subset=["btc_premium", "country_code", "year"])

    logger.info(f"Panel data: {len(panel)} observations, {panel['country_code'].nunique()} countries")

    # Set up panel structure
    panel = panel.set_index(["country_code", "year"])

    # Define variables
    dep_var = "btc_premium"
    base_vars = ["capital_control_index", "inflation_wb", "political_stability"]
    control_vars = ["log_gdp_pc", "internet_penetration"]
    crime_vars = ["criminality_score", "financial_crimes"]  # Key crime variables

    # Check available variables
    available_base = [v for v in base_vars if v in panel.columns]
    available_controls = [v for v in control_vars if v in panel.columns]
    available_crime = [v for v in crime_vars if v in panel.columns]

    if not available_base:
        return {"error": "No explanatory variables available"}

    # Create interaction terms
    if "capital_control_index" in panel.columns and "inflation_wb" in panel.columns:
        panel["cc_x_inflation"] = panel["capital_control_index"] * panel["inflation_wb"]

    if "capital_control_index" in panel.columns and "political_stability" in panel.columns:
        panel["cc_x_instability"] = panel["capital_control_index"] * (-panel["political_stability"])

    # Crime interaction terms
    if "capital_control_index" in panel.columns and "criminality_score" in panel.columns:
        panel["cc_x_crime"] = panel["capital_control_index"] * panel["criminality_score"]

    if "capital_control_index" in panel.columns and "financial_crimes" in panel.columns:
        panel["cc_x_fin_crime"] = panel["capital_control_index"] * panel["financial_crimes"]

    results = {}

    # Prepare regression data
    all_vars = available_base + available_controls
    reg_data = panel[[dep_var] + all_vars].dropna()

    logger.info(f"Regression sample: {len(reg_data)} observations after dropping missing")

    if len(reg_data) < 20:
        return {"error": f"Insufficient observations ({len(reg_data)})"}

    y = reg_data[dep_var]
    X = reg_data[all_vars]

    # =========================================================================
    # Model 1: Pooled OLS
    # =========================================================================
    try:
        model_pooled = PooledOLS(y, sm.add_constant(X)).fit(cov_type="clustered", cluster_entity=True)
        results["pooled_ols"] = {
            "model": model_pooled,
            "r_squared": model_pooled.rsquared,
            "n_obs": model_pooled.nobs,
            "coefficients": dict(model_pooled.params),
            "pvalues": dict(model_pooled.pvalues),
        }
        logger.info(f"Pooled OLS R²: {model_pooled.rsquared:.4f}")
    except Exception as e:
        logger.warning(f"Pooled OLS failed: {e}")

    # =========================================================================
    # Model 2: Country Fixed Effects
    # =========================================================================
    try:
        model_fe = PanelOLS(y, X, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
        results["country_fe"] = {
            "model": model_fe,
            "r_squared_within": model_fe.rsquared_within,
            "r_squared_between": model_fe.rsquared_between,
            "r_squared_overall": model_fe.rsquared_overall,
            "n_obs": model_fe.nobs,
            "n_entities": model_fe.entity_info.total,
            "coefficients": dict(model_fe.params),
            "pvalues": dict(model_fe.pvalues),
        }
        logger.info(f"Country FE R² (within): {model_fe.rsquared_within:.4f}")
    except Exception as e:
        logger.warning(f"Country FE failed: {e}")

    # =========================================================================
    # Model 3: Country + Year Fixed Effects (Two-way FE)
    # =========================================================================
    try:
        model_twoway = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
            cov_type="clustered", cluster_entity=True
        )
        results["twoway_fe"] = {
            "model": model_twoway,
            "r_squared_within": model_twoway.rsquared_within,
            "r_squared_overall": model_twoway.rsquared_overall,
            "n_obs": model_twoway.nobs,
            "coefficients": dict(model_twoway.params),
            "pvalues": dict(model_twoway.pvalues),
        }
        logger.info(f"Two-way FE R² (within): {model_twoway.rsquared_within:.4f}")
    except Exception as e:
        logger.warning(f"Two-way FE failed: {e}")

    # =========================================================================
    # Model 4: Fixed Effects with Interactions
    # =========================================================================
    interaction_vars = [v for v in ["cc_x_inflation", "cc_x_instability"] if v in panel.columns]
    if interaction_vars:
        try:
            all_vars_interact = all_vars + interaction_vars
            reg_data_interact = panel[[dep_var] + all_vars_interact].dropna()
            y_int = reg_data_interact[dep_var]
            X_int = reg_data_interact[all_vars_interact]

            model_fe_interact = PanelOLS(y_int, X_int, entity_effects=True).fit(
                cov_type="clustered", cluster_entity=True
            )
            results["fe_interactions"] = {
                "model": model_fe_interact,
                "r_squared_within": model_fe_interact.rsquared_within,
                "n_obs": model_fe_interact.nobs,
                "coefficients": dict(model_fe_interact.params),
                "pvalues": dict(model_fe_interact.pvalues),
            }
            logger.info(f"FE with interactions R² (within): {model_fe_interact.rsquared_within:.4f}")
        except Exception as e:
            logger.warning(f"FE with interactions failed: {e}")

    # =========================================================================
    # Model 5: Pooled OLS with Crime Variables
    # =========================================================================
    # Use criminality_score which has time-series data (2021, 2023)
    if "criminality_score" in panel.columns:
        try:
            # Only use criminality_score (available for 2021+2023)
            crime_model_vars = available_base + ["criminality_score"]
            crime_reg_data = panel[[dep_var] + crime_model_vars].dropna()

            if len(crime_reg_data) >= 20:
                y_crime = crime_reg_data[dep_var]
                X_crime = crime_reg_data[crime_model_vars]

                model_crime_pooled = PooledOLS(y_crime, sm.add_constant(X_crime)).fit(
                    cov_type="clustered", cluster_entity=True
                )
                results["pooled_crime"] = {
                    "model": model_crime_pooled,
                    "r_squared": model_crime_pooled.rsquared,
                    "n_obs": model_crime_pooled.nobs,
                    "coefficients": dict(model_crime_pooled.params),
                    "pvalues": dict(model_crime_pooled.pvalues),
                }
                logger.info(f"Pooled OLS with crime R²: {model_crime_pooled.rsquared:.4f}, N={model_crime_pooled.nobs}")
            else:
                logger.warning(f"Insufficient observations for crime model: {len(crime_reg_data)}")
        except Exception as e:
            logger.warning(f"Pooled OLS with crime failed: {e}")

    # =========================================================================
    # Model 6: FE with FATF Grey List Status
    # =========================================================================
    # FATF grey list is time-varying and captures regulatory vulnerability
    if "fatf_grey" in panel.columns:
        try:
            fatf_vars = available_base + ["fatf_grey"]
            fatf_data = panel[[dep_var] + fatf_vars].dropna()

            if len(fatf_data) >= 30:
                y_fatf = fatf_data[dep_var]
                X_fatf = fatf_data[fatf_vars]

                # Pooled OLS with FATF (FATF is time-varying so can use in FE too)
                model_fatf_pooled = PooledOLS(y_fatf, sm.add_constant(X_fatf)).fit(
                    cov_type="clustered", cluster_entity=True
                )
                results["pooled_fatf"] = {
                    "model": model_fatf_pooled,
                    "r_squared": model_fatf_pooled.rsquared,
                    "n_obs": model_fatf_pooled.nobs,
                    "coefficients": dict(model_fatf_pooled.params),
                    "pvalues": dict(model_fatf_pooled.pvalues),
                }
                logger.info(f"Pooled OLS with FATF R²: {model_fatf_pooled.rsquared:.4f}")

                # FE with FATF
                model_fatf_fe = PanelOLS(y_fatf, X_fatf, entity_effects=True).fit(
                    cov_type="clustered", cluster_entity=True
                )
                results["fe_fatf"] = {
                    "model": model_fatf_fe,
                    "r_squared_within": model_fatf_fe.rsquared_within,
                    "n_obs": model_fatf_fe.nobs,
                    "coefficients": dict(model_fatf_fe.params),
                    "pvalues": dict(model_fatf_fe.pvalues),
                }
                logger.info(f"FE with FATF R² (within): {model_fatf_fe.rsquared_within:.4f}")

        except Exception as e:
            logger.warning(f"FATF model failed: {e}")

    # =========================================================================
    # Model 7: FE with CC × FATF Interaction
    # =========================================================================
    if "fatf_grey" in panel.columns and "capital_control_index" in panel.columns:
        try:
            panel["cc_x_fatf"] = panel["capital_control_index"] * panel["fatf_grey"]
            fatf_int_vars = available_base + ["fatf_grey", "cc_x_fatf"]
            fatf_int_data = panel[[dep_var] + fatf_int_vars].dropna()

            if len(fatf_int_data) >= 30:
                y_fatf_int = fatf_int_data[dep_var]
                X_fatf_int = fatf_int_data[fatf_int_vars]

                model_fatf_interact = PanelOLS(y_fatf_int, X_fatf_int, entity_effects=True).fit(
                    cov_type="clustered", cluster_entity=True
                )
                results["fe_fatf_interactions"] = {
                    "model": model_fatf_interact,
                    "r_squared_within": model_fatf_interact.rsquared_within,
                    "n_obs": model_fatf_interact.nobs,
                    "coefficients": dict(model_fatf_interact.params),
                    "pvalues": dict(model_fatf_interact.pvalues),
                }
                logger.info(f"FE with CC×FATF R² (within): {model_fatf_interact.rsquared_within:.4f}")

        except Exception as e:
            logger.warning(f"FATF interaction model failed: {e}")

    # =========================================================================
    # Save Results
    # =========================================================================
    _save_panel_results(results, output_dir)

    return results


def _save_panel_results(results: dict, output_dir: Path) -> None:
    """Save panel regression results to file."""
    with open(output_dir / "panel_regression.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PANEL REGRESSION ANALYSIS\n")
        f.write("BTC Premium and Capital Controls - Full Panel\n")
        f.write("=" * 80 + "\n\n")

        # Model comparison table
        f.write("MODEL COMPARISON\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<25} {'N':>8} {'R²':>10} {'CC Coef':>10} {'p-val':>8}\n")
        f.write("-" * 60 + "\n")

        for model_name, model_results in results.items():
            if "error" in model_results:
                continue

            n = model_results.get("n_obs", "N/A")
            r2 = model_results.get("r_squared_within", model_results.get("r_squared", "N/A"))
            if isinstance(r2, float):
                r2_str = f"{r2:.4f}"
            else:
                r2_str = str(r2)

            coefs = model_results.get("coefficients", {})
            pvals = model_results.get("pvalues", {})

            cc_coef = coefs.get("capital_control_index", "N/A")
            cc_pval = pvals.get("capital_control_index", "N/A")

            if isinstance(cc_coef, float):
                cc_str = f"{cc_coef:.4f}"
            else:
                cc_str = str(cc_coef)

            if isinstance(cc_pval, float):
                pval_str = f"{cc_pval:.4f}"
                sig = "***" if cc_pval < 0.01 else "**" if cc_pval < 0.05 else "*" if cc_pval < 0.1 else ""
                pval_str += sig
            else:
                pval_str = str(cc_pval)

            f.write(f"{model_name:<25} {n:>8} {r2_str:>10} {cc_str:>10} {pval_str:>8}\n")

        f.write("-" * 60 + "\n")
        f.write("Significance: *** p<0.01, ** p<0.05, * p<0.1\n\n")

        # Detailed results for each model
        for model_name, model_results in results.items():
            if "model" not in model_results:
                continue

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{model_name.upper().replace('_', ' ')}\n")
            f.write("=" * 80 + "\n")
            f.write(str(model_results["model"].summary))
            f.write("\n")

        # Interpretation
        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")

        if "country_fe" in results and "coefficients" in results["country_fe"]:
            coefs = results["country_fe"]["coefficients"]
            pvals = results["country_fe"]["pvalues"]

            f.write("Country Fixed Effects Model (preferred specification):\n")
            f.write("-" * 50 + "\n")

            for var in coefs:
                coef = coefs[var]
                pval = pvals.get(var, 1.0)
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""

                f.write(f"  {var}: {coef:.4f} (p={pval:.4f}) {sig}\n")

                # Interpretation
                if var == "capital_control_index":
                    if coef > 0:
                        f.write(f"    → Higher capital controls associated with higher BTC premiums\n")
                    else:
                        f.write(f"    → Higher capital controls associated with lower BTC premiums\n")
                elif var == "inflation_wb":
                    if coef > 0:
                        f.write(f"    → Higher inflation associated with higher BTC premiums\n")
                elif var == "cc_x_inflation":
                    if coef > 0:
                        f.write(f"    → Capital control effect amplified in high-inflation environments\n")
                    else:
                        f.write(f"    → Capital control effect dampened in high-inflation environments\n")

    logger.info(f"Saved panel regression results to {output_dir / 'panel_regression.txt'}")


def _plot_interaction_effects(
    df: pd.DataFrame,
    model,
    output_dir: Path,
) -> None:
    """Plot interaction effects showing how capital control effect varies by context."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Effect at low vs high inflation
        if "inflation_wb" in df.columns and "capital_control_index" in df.columns:
            ax = axes[0]
            low_infl = df["inflation_wb"] <= df["inflation_wb"].median()
            high_infl = ~low_infl

            for mask, label, color in [(low_infl, "Low Inflation", "blue"),
                                        (high_infl, "High Inflation", "red")]:
                subset = df[mask].dropna(subset=["capital_control_index", "btc_premium"])
                if len(subset) > 5:
                    ax.scatter(subset["capital_control_index"], subset["btc_premium"],
                              alpha=0.6, label=label, c=color)
                    # Add trend line
                    z = np.polyfit(subset["capital_control_index"], subset["btc_premium"], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(subset["capital_control_index"].min(),
                                        subset["capital_control_index"].max(), 100)
                    ax.plot(x_line, p(x_line), c=color, linestyle="--", alpha=0.8)

            ax.set_xlabel("Capital Control Index")
            ax.set_ylabel("BTC Premium (%)")
            ax.set_title("Capital Controls Effect by Inflation Level")
            ax.legend()

        # Plot 2: Effect at low vs high political stability
        if "political_stability" in df.columns and "capital_control_index" in df.columns:
            ax = axes[1]
            stable = df["political_stability"] >= df["political_stability"].median()
            unstable = ~stable

            for mask, label, color in [(stable, "Politically Stable", "green"),
                                        (unstable, "Politically Unstable", "orange")]:
                subset = df[mask].dropna(subset=["capital_control_index", "btc_premium"])
                if len(subset) > 5:
                    ax.scatter(subset["capital_control_index"], subset["btc_premium"],
                              alpha=0.6, label=label, c=color)
                    z = np.polyfit(subset["capital_control_index"], subset["btc_premium"], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(subset["capital_control_index"].min(),
                                        subset["capital_control_index"].max(), 100)
                    ax.plot(x_line, p(x_line), c=color, linestyle="--", alpha=0.8)

            ax.set_xlabel("Capital Control Index")
            ax.set_ylabel("BTC Premium (%)")
            ax.set_title("Capital Controls Effect by Political Stability")
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "interaction_plots.png", dpi=150)
        plt.close()

        logger.info(f"Saved interaction plots to {output_dir / 'interaction_plots.png'}")

    except Exception as e:
        logger.warning(f"Could not generate interaction plots: {e}")


def generate_coefficient_plot(
    model,
    output_dir: Path,
) -> None:
    """Generate a coefficient plot with confidence intervals."""
    # Extract coefficients (excluding constant)
    coef_df = pd.DataFrame(
        {
            "variable": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "pvalue": model.pvalues.values,
        }
    )

    # Filter out constant
    coef_df = coef_df[coef_df["variable"] != "const"]

    # Calculate confidence intervals
    coef_df["ci_lower"] = coef_df["coefficient"] - 1.96 * coef_df["std_error"]
    coef_df["ci_upper"] = coef_df["coefficient"] + 1.96 * coef_df["std_error"]

    # Significance markers
    coef_df["significant"] = coef_df["pvalue"] < 0.05

    # Plot
    plt.figure(figsize=(10, 6))

    colors = ["#2ecc71" if sig else "#95a5a6" for sig in coef_df["significant"]]

    plt.barh(
        coef_df["variable"],
        coef_df["coefficient"],
        xerr=[
            coef_df["coefficient"] - coef_df["ci_lower"],
            coef_df["ci_upper"] - coef_df["coefficient"],
        ],
        color=colors,
        capsize=5,
        alpha=0.8,
    )

    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    plt.xlabel("Coefficient (with 95% CI)")
    plt.title("Regression Coefficients: BTC Premium Determinants")

    # Add significance note
    plt.figtext(
        0.02,
        0.02,
        "Green = significant at 5% level",
        fontsize=8,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "coefficient_plot.png", dpi=150)
    plt.close()

    logger.info(f"Saved coefficient plot to {output_dir / 'coefficient_plot.png'}")


def save_results_summary(
    results: dict,
    output_dir: Path,
) -> None:
    """Save a high-level results summary."""
    with open(output_dir / "analysis_summary.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 1 ANALYSIS SUMMARY\n")
        f.write("BTC Premium and Capital Controls: Cross-Sectional Analysis\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Number of countries: {results.get('n_countries', 'N/A')}\n")
        f.write(f"Year/Period: {results.get('year', 'N/A')}\n\n")

        main_reg = results.get("main_regression", {})
        if "error" not in main_reg:
            f.write("Main Regression Results:\n")
            f.write("-" * 40 + "\n")
            f.write(f"R-squared: {main_reg.get('r_squared', 'N/A'):.4f}\n")
            f.write(f"Adj. R-squared: {main_reg.get('adj_r_squared', 'N/A'):.4f}\n")
            f.write(f"F-statistic: {main_reg.get('f_statistic', 'N/A'):.4f}\n")
            f.write(f"F p-value: {main_reg.get('f_pvalue', 'N/A'):.4f}\n")
            f.write(f"N observations: {main_reg.get('n_obs', 'N/A')}\n\n")

            f.write("Key Coefficients:\n")
            coefficients = main_reg.get("coefficients", {})
            pvalues = main_reg.get("pvalues", {})
            for var, coef in coefficients.items():
                if var != "const":
                    pval = pvalues.get(var, "N/A")
                    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                    f.write(f"  {var}: {coef:.4f} (p={pval:.4f}) {sig}\n")
        else:
            f.write(f"Regression error: {main_reg.get('error')}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Significance levels: *** p<0.01, ** p<0.05, * p<0.1\n")

    logger.info(f"Saved analysis summary to {output_dir / 'analysis_summary.txt'}")
