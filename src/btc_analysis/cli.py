"""Command-line interface for BTC regulatory arbitrage analysis."""

import logging
import sys
from pathlib import Path

import click

from btc_analysis.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """BTC Regulatory Arbitrage Analysis Pipeline.

    This tool fetches data, processes it, and runs regression analysis
    to test the hypothesis that Bitcoin premiums are driven by capital
    flight demand in countries with restrictive capital controls.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option(
    "--source",
    type=click.Choice(["imf", "chinn-ito", "wgi", "dev", "timeseries", "all"]),
    default="all",
    help="Data source to fetch",
)
@click.option(
    "--start-year",
    type=int,
    default=2015,
    help="Start year for data",
)
@click.option(
    "--end-year",
    type=int,
    default=2024,
    help="End year for data",
)
def fetch(source, start_year, end_year):
    """Fetch data from specified source(s).

    Downloads and saves raw data files to data/raw/ directory.

    Examples:
        btc-analysis fetch --source all
        btc-analysis fetch --source imf --start-year 2018
        btc-analysis fetch --source chinn-ito
    """
    config = get_config()

    click.echo(f"Fetching data from: {source}")
    click.echo(f"Year range: {start_year}-{end_year}")
    click.echo(f"Output directory: {config.paths.raw_dir}")
    click.echo("-" * 50)

    if source in ["imf", "all"]:
        click.echo("\n[IMF Crypto Shadow Rate]")
        try:
            from btc_analysis.data.imf import fetch_crypto_shadow_rate, fetch_cpi_data

            df = fetch_crypto_shadow_rate(start_year=start_year, end_year=end_year)
            click.echo(f"  Shadow rate: {len(df)} records")

            cpi = fetch_cpi_data(start_year=start_year, end_year=end_year)
            click.echo(f"  CPI data: {len(cpi)} records")
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

    if source in ["chinn-ito", "all"]:
        click.echo("\n[Chinn-Ito Capital Controls Index]")
        try:
            from btc_analysis.data.capital_controls import fetch_chinn_ito

            df = fetch_chinn_ito()
            click.echo(f"  Records: {len(df)}")
            if not df.empty:
                years = df["year"].agg(["min", "max"])
                click.echo(f"  Year range: {years['min']}-{years['max']}")
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

    if source in ["wgi", "all"]:
        click.echo("\n[World Bank Governance Indicators]")
        try:
            from btc_analysis.data.world_bank import fetch_governance_indicators

            df = fetch_governance_indicators(start_year=start_year, end_year=end_year)
            click.echo(f"  Records: {len(df)}")
            if not df.empty:
                click.echo(f"  Countries: {df['country_code'].nunique()}")
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

    if source in ["dev", "all"]:
        click.echo("\n[World Bank Development Indicators]")
        try:
            from btc_analysis.data.world_bank import fetch_development_indicators

            df = fetch_development_indicators(start_year=start_year, end_year=end_year)
            click.echo(f"  Records: {len(df)}")
            if not df.empty:
                click.echo(f"  Countries: {df['country_code'].nunique()}")
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

    if source in ["timeseries", "all"]:
        click.echo("\n[Time Series Data - BTC Price]")
        try:
            from btc_analysis.data.btc_price import fetch_btc_price

            df = fetch_btc_price(
                start_date=f"{start_year}-01-01",
                end_date=f"{end_year}-12-31",
            )
            click.echo(f"  BTC price records: {len(df)}")
            if not df.empty:
                click.echo(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

        click.echo("\n[Time Series Data - CDC Overdose Deaths]")
        try:
            from btc_analysis.data.cdc_overdose import fetch_overdose_deaths

            df = fetch_overdose_deaths(start_year=start_year, end_year=end_year)
            click.echo(f"  Overdose records: {len(df)}")
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

        click.echo("\n[Time Series Data - CBP Drug Seizures]")
        try:
            from btc_analysis.data.cbp_seizures import fetch_drug_seizures

            df = fetch_drug_seizures(start_year=start_year, end_year=end_year)
            click.echo(f"  Seizure records: {len(df)}")
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)

    click.echo("\n" + "=" * 50)
    click.echo("Fetch complete. Check data/raw/ for downloaded files.")


@cli.command()
@click.option(
    "--start-year",
    type=int,
    default=2015,
    help="Start year for merged data",
)
@click.option(
    "--end-year",
    type=int,
    default=2023,
    help="End year for merged data",
)
@click.option(
    "--min-obs",
    type=int,
    default=2,
    help="Minimum observations per country",
)
def process(start_year, end_year, min_obs):
    """Process and merge downloaded datasets.

    Merges all available data sources into a single panel dataset
    suitable for regression analysis.

    Output: data/processed/merged_panel.csv
    """
    config = get_config()

    click.echo("Processing and merging datasets...")
    click.echo(f"Year range: {start_year}-{end_year}")
    click.echo(f"Minimum observations per country: {min_obs}")
    click.echo("-" * 50)

    try:
        from btc_analysis.processing.merge import merge_datasets, clean_panel

        # Merge datasets
        click.echo("\nMerging datasets...")
        merged = merge_datasets(start_year=start_year, end_year=end_year)

        if merged.empty:
            click.echo("Warning: Merged dataset is empty", err=True)
            return

        click.echo(f"  Initial merge: {len(merged)} observations")
        click.echo(f"  Countries: {merged['country_code'].nunique()}")
        click.echo(f"  Columns: {list(merged.columns)}")

        # Clean panel
        click.echo("\nCleaning panel...")
        cleaned = clean_panel(merged, min_obs=min_obs)

        click.echo(f"  After cleaning: {len(cleaned)} observations")
        click.echo(f"  Countries: {cleaned['country_code'].nunique()}")

        # Save cleaned version
        output_path = config.paths.processed_dir / "cleaned_panel.csv"
        cleaned.to_csv(output_path, index=False)
        click.echo(f"\nSaved cleaned panel to: {output_path}")

        # Show data summary
        click.echo("\nData Summary:")
        click.echo("-" * 30)
        for col in cleaned.select_dtypes(include=["float64", "int64"]).columns:
            non_null = cleaned[col].notna().sum()
            if non_null > 0:
                click.echo(f"  {col}: {non_null} values, mean={cleaned[col].mean():.2f}")

    except Exception as e:
        click.echo(f"Error during processing: {e}", err=True)
        logger.exception("Processing failed")
        raise click.Abort()


@cli.command()
@click.option(
    "--phase",
    type=click.Choice(["phase1", "panel", "phase2", "phase3", "cash_substitution"]),
    default="phase1",
    help="Analysis phase to run",
)
@click.option(
    "--year",
    type=int,
    default=None,
    help="Specific year for cross-section (default: use averages)",
)
def analyze(phase, year):
    """Run regression analysis.

    Phase 1: Cross-sectional regression of BTC premiums on capital controls
    Panel: Panel regression with country fixed effects (recommended)
    Phase 2: Event study (not yet implemented)
    Phase 3: Time-series analysis (not yet implemented)

    Output: outputs/phase1/ (tables, plots, summary)
    """
    config = get_config()

    click.echo(f"Running {phase} analysis...")
    if year:
        click.echo(f"Year: {year}")
    click.echo("-" * 50)

    if phase == "phase1":
        try:
            from btc_analysis.analysis.cross_sectional import run_phase1_analysis

            results = run_phase1_analysis(year=year)

            if "error" in results:
                click.echo(f"Analysis error: {results['error']}", err=True)
                return

            # Display key results
            click.echo("\n" + "=" * 50)
            click.echo("PHASE 1 RESULTS")
            click.echo("=" * 50)

            click.echo(f"\nCountries analyzed: {results.get('n_countries', 'N/A')}")

            main_reg = results.get("main_regression", {})
            if "model" in main_reg:
                click.echo(f"\nRegression Statistics:")
                click.echo(f"  R-squared: {main_reg['r_squared']:.4f}")
                click.echo(f"  Adj. R-squared: {main_reg['adj_r_squared']:.4f}")
                click.echo(f"  F-statistic: {main_reg['f_statistic']:.4f}")
                click.echo(f"  Observations: {main_reg['n_obs']}")

                click.echo("\nKey Coefficients:")
                for var, coef in main_reg["coefficients"].items():
                    if var != "const":
                        pval = main_reg["pvalues"].get(var, 1.0)
                        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                        click.echo(f"  {var}: {coef:.4f} (p={pval:.4f}) {sig}")

            output_dir = config.paths.outputs_dir / "phase1"
            click.echo(f"\nFull results saved to: {output_dir}")

        except Exception as e:
            click.echo(f"Analysis error: {e}", err=True)
            logger.exception("Analysis failed")
            raise click.Abort()

    elif phase == "panel":
        try:
            from btc_analysis.analysis.cross_sectional import run_panel_regression

            results = run_panel_regression()

            if "error" in results:
                click.echo(f"Analysis error: {results['error']}", err=True)
                return

            # Display key results
            click.echo("\n" + "=" * 60)
            click.echo("PANEL REGRESSION RESULTS")
            click.echo("=" * 60)

            # Model comparison
            click.echo(f"\n{'Model':<25} {'N':>8} {'R²':>10} {'CC Coef':>10} {'p-val':>10}")
            click.echo("-" * 65)

            for model_name, model_results in results.items():
                if "n_obs" not in model_results:
                    continue

                n = model_results.get("n_obs", "N/A")
                r2 = model_results.get("r_squared_within", model_results.get("r_squared", 0))
                coefs = model_results.get("coefficients", {})
                pvals = model_results.get("pvalues", {})

                cc_coef = coefs.get("capital_control_index", None)
                cc_pval = pvals.get("capital_control_index", None)

                r2_str = f"{r2:.4f}" if isinstance(r2, float) else "N/A"
                cc_str = f"{cc_coef:.4f}" if cc_coef is not None else "N/A"

                if cc_pval is not None:
                    sig = "***" if cc_pval < 0.01 else "**" if cc_pval < 0.05 else "*" if cc_pval < 0.1 else ""
                    pval_str = f"{cc_pval:.4f}{sig}"
                else:
                    pval_str = "N/A"

                click.echo(f"{model_name:<25} {n:>8} {r2_str:>10} {cc_str:>10} {pval_str:>10}")

            click.echo("-" * 65)
            click.echo("Significance: *** p<0.01, ** p<0.05, * p<0.1")

            # Show preferred model details
            if "country_fe" in results:
                fe = results["country_fe"]
                click.echo("\n[Country Fixed Effects - Preferred Model]")
                click.echo(f"  Observations: {fe.get('n_obs', 'N/A')}")
                click.echo(f"  Countries: {fe.get('n_entities', 'N/A')}")
                click.echo(f"  R² (within): {fe.get('r_squared_within', 0):.4f}")

                click.echo("\n  Coefficients:")
                for var, coef in fe.get("coefficients", {}).items():
                    pval = fe.get("pvalues", {}).get(var, 1.0)
                    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                    click.echo(f"    {var}: {coef:.4f} (p={pval:.4f}) {sig}")

            output_dir = config.paths.outputs_dir / "phase1"
            click.echo(f"\nFull results saved to: {output_dir}/panel_regression.txt")

        except Exception as e:
            click.echo(f"Analysis error: {e}", err=True)
            logger.exception("Panel analysis failed")
            raise click.Abort()

    elif phase == "phase2":
        click.echo("Phase 2 (Event Study) is not yet implemented.")
        click.echo("Coming soon: Event study analysis")

    elif phase == "phase3":
        try:
            from btc_analysis.analysis.time_series import run_time_series_analysis

            click.echo("\nRunning time series analysis...")
            click.echo("Testing BTC price correlation with criminal enterprise proxies")
            click.echo("-" * 50)

            results = run_time_series_analysis()

            if not results:
                click.echo("Analysis returned no results. Check data availability.", err=True)
                return

            # Display key results
            click.echo("\n" + "=" * 60)
            click.echo("TIME SERIES ANALYSIS RESULTS")
            click.echo("=" * 60)

            # Data summary
            if "data_summary" in results:
                summary = results["data_summary"]
                click.echo(f"\nData: {summary.get('n_observations', 'N/A')} observations")
                click.echo(f"Period: {summary.get('date_range', 'N/A')}")

            # Granger causality results
            if "granger_tests" in results and results["granger_tests"]:
                click.echo("\nGranger Causality Tests:")
                click.echo("-" * 40)
                for test_name, test in results["granger_tests"].items():
                    sig = "***" if test.get("significant") else ""
                    click.echo(f"  {test_name}: p={test.get('best_p_value', 1):.4f} "
                              f"(lag={test.get('best_lag', 0)}) {sig}")

            # Regression results
            if "regression_models" in results:
                for model_name, model in results["regression_models"].items():
                    if not model:
                        continue
                    click.echo(f"\n{model_name.upper()}:")
                    click.echo(f"  R² = {model.get('r_squared', 0):.4f}, "
                              f"N = {model.get('n_obs', 0)}")

                    if "coefficients" in model:
                        crime_coefs = {k: v for k, v in model["coefficients"].items()
                                      if any(term in k.lower() for term in
                                            ["overdose", "cocaine", "fentanyl", "seizure"])}
                        for var, coef in crime_coefs.items():
                            p = coef.get("p_value", 1)
                            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                            click.echo(f"    {var}: {coef.get('coef', 0):.4f} (p={p:.4f}) {sig}")

            output_dir = config.paths.outputs_dir / "timeseries"
            click.echo(f"\nFull results saved to: {output_dir}")

        except Exception as e:
            click.echo(f"Time series analysis error: {e}", err=True)
            logger.exception("Time series analysis failed")
            raise click.Abort()

    elif phase == "cash_substitution":
        try:
            from btc_analysis.analysis.cash_substitution import run_cash_substitution_analysis

            click.echo("\nRunning cash substitution analysis...")
            click.echo("Testing exit liquidity hypothesis")
            click.echo("-" * 50)

            results = run_cash_substitution_analysis()

            if not results:
                click.echo("Analysis returned no results. Check data availability.", err=True)
                return

            # Display key results
            click.echo("\n" + "=" * 60)
            click.echo("CASH SUBSTITUTION ANALYSIS RESULTS")
            click.echo("=" * 60)

            # Data summary
            if "data_summary" in results:
                summary = results["data_summary"]
                click.echo(f"\nData: {summary.get('n_observations', 'N/A')} observations")
                click.echo(f"Period: {summary.get('date_range', 'N/A')}")

            # Regime analysis
            if "regime_analysis" in results:
                regime = results["regime_analysis"]
                click.echo("\nREGIME ANALYSIS:")
                click.echo("-" * 40)

                if "ratio_trend" in regime:
                    trend = regime["ratio_trend"]
                    click.echo(f"Cash/Drug ratio trend: {trend.get('monthly_pct_change', 0):.2f}% per month")
                    click.echo(f"  Declining: {trend.get('is_declining', False)}")

                for key in ["pre_institutional_residual_correlation",
                           "post_institutional_residual_correlation"]:
                    if key in regime:
                        data = regime[key]
                        sig = "***" if data["p_value"] < 0.01 else "**" if data["p_value"] < 0.05 else "*" if data["p_value"] < 0.1 else ""
                        click.echo(f"  {data['label']}: r = {data['correlation']:.4f} "
                                  f"(p={data['p_value']:.4f}) {sig}")

            # Lagged models (PRIMARY CAUSAL FINDING)
            if "lagged_models" in results:
                lm = results["lagged_models"]
                click.echo("\nLAGGED MODELS (Primary Causal Spec):")
                click.echo("-" * 40)
                for lag_key in ["lag_1", "lag_2", "lag_3"]:
                    if lag_key in lm:
                        lag_data = lm[lag_key]
                        sig = "***" if lag_data.get("interaction_pvalue", 1) < 0.01 else "**" if lag_data.get("interaction_pvalue", 1) < 0.05 else "*" if lag_data.get("interaction_pvalue", 1) < 0.1 else ""
                        click.echo(f"  {lag_key}: coef={lag_data.get('interaction_coef', 0):+.4f} "
                                  f"(p={lag_data.get('interaction_pvalue', 1):.4f}) {sig}")
                if "lag_1" in lm and lm["lag_1"].get("interaction_pvalue", 1) < 0.1:
                    click.echo(f"\n  ** Lag-1 is significant: demand effect, not exit selling **")

            # Volume interaction (contemporaneous - for comparison)
            if "volume_interaction" in results:
                vi = results["volume_interaction"]
                click.echo("\nCONTEMPORANEOUS (for comparison):")
                click.echo("-" * 40)

                for key in ["correlation_low_volume", "correlation_high_volume"]:
                    if key in vi:
                        data = vi[key]
                        sig = "***" if data["p_value"] < 0.01 else "**" if data["p_value"] < 0.05 else "*" if data["p_value"] < 0.1 else ""
                        click.echo(f"  {data['label']}: r = {data['correlation']:.4f} "
                                  f"(p={data['p_value']:.4f}) {sig}")

                if "interaction_model" in vi:
                    model = vi["interaction_model"]
                    interact_coef = model["coefficients"].get("subst_x_volume", {})
                    click.echo(f"\n  Interaction coefficient: {interact_coef.get('coefficient', 0):.4f} "
                              f"(p={interact_coef.get('p_value', 1):.4f})")

                if "economic_significance" in vi:
                    click.echo(f"\n  {vi['economic_significance']['interpretation']}")

            # Stress tests
            if "stress_tests" in results:
                st = results["stress_tests"]
                click.echo("\nSTRESS TESTS:")
                click.echo("-" * 40)

                tests_passed = 0
                tests_total = 0

                if "bootstrap" in st:
                    tests_total += 1
                    passed = st["bootstrap"]["ci_excludes_zero"]
                    tests_passed += int(passed)
                    click.echo(f"  Bootstrap CI excludes zero: {'PASS' if passed else 'FAIL'}")

                if "placebo_test" in st:
                    tests_total += 1
                    passed = st["placebo_test"]["passes"]
                    tests_passed += int(passed)
                    click.echo(f"  Placebo test (p < 0.05): {'PASS' if passed else 'FAIL'}")

                if "leave_one_out" in st:
                    tests_total += 1
                    passed = st["leave_one_out"]["all_negative"]
                    tests_passed += int(passed)
                    click.echo(f"  Leave-one-out all negative: {'PASS' if passed else 'FAIL'}")

                if "falsification_sp500" in st:
                    tests_total += 1
                    passed = st["falsification_sp500"]["passes"]
                    tests_passed += int(passed)
                    click.echo(f"  Falsification (S&P 500 not significant): {'PASS' if passed else 'FAIL'}")

                click.echo(f"\n  Tests passed: {tests_passed}/{tests_total}")

            output_dir = config.paths.outputs_dir / "cash_substitution"
            click.echo(f"\nFull results saved to: {output_dir}")

        except Exception as e:
            click.echo(f"Cash substitution analysis error: {e}", err=True)
            logger.exception("Cash substitution analysis failed")
            raise click.Abort()


@cli.command()
def report():
    """Generate summary report of all analyses.

    Compiles results from all completed analysis phases into
    a single summary report.
    """
    config = get_config()

    click.echo("Generating summary report...")

    output_dir = config.paths.outputs_dir
    report_path = output_dir / "full_report.txt"

    sections = []

    # Phase 1 results
    phase1_summary = output_dir / "phase1" / "analysis_summary.txt"
    if phase1_summary.exists():
        sections.append(("PHASE 1: CROSS-SECTIONAL ANALYSIS", phase1_summary.read_text()))

    if not sections:
        click.echo("No analysis results found. Run 'btc-analysis analyze' first.")
        return

    # Write combined report
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BTC REGULATORY ARBITRAGE ANALYSIS - FULL REPORT\n")
        f.write("=" * 80 + "\n\n")

        for title, content in sections:
            f.write(f"\n{title}\n")
            f.write("-" * len(title) + "\n")
            f.write(content)
            f.write("\n")

    click.echo(f"Report saved to: {report_path}")


@cli.command()
def status():
    """Show status of data files and analysis outputs."""
    config = get_config()

    click.echo("BTC Analysis Pipeline Status")
    click.echo("=" * 50)

    # Check raw data files
    click.echo("\nRaw Data Files (Cross-sectional):")
    raw_files = [
        ("IMF Shadow Rate", "imf_crypto_shadow_rate.csv"),
        ("IMF CPI", "imf_cpi.csv"),
        ("Chinn-Ito Index", "chinn_ito.csv"),
        ("World Bank WGI", "world_bank_wgi.csv"),
        ("World Bank Dev", "world_bank_dev.csv"),
        ("FATF Status", "fatf_status.csv"),
        ("Crime Index", "crime_index.csv"),
    ]

    for name, filename in raw_files:
        path = config.paths.raw_dir / filename
        if path.exists():
            size = path.stat().st_size / 1024
            click.echo(f"  [x] {name}: {size:.1f} KB")
        else:
            click.echo(f"  [ ] {name}: Not found")

    # Check time series raw data files
    click.echo("\nRaw Data Files (Time Series):")
    ts_raw_files = [
        ("BTC Price", "btc_price.csv"),
        ("CDC Overdose Deaths", "cdc_overdose_deaths.csv"),
        ("CBP Drug Seizures", "cbp_drug_seizures.csv"),
    ]

    for name, filename in ts_raw_files:
        path = config.paths.raw_dir / filename
        if path.exists():
            size = path.stat().st_size / 1024
            click.echo(f"  [x] {name}: {size:.1f} KB")
        else:
            click.echo(f"  [ ] {name}: Not found")

    # Check processed data
    click.echo("\nProcessed Data:")
    processed_files = [
        ("Merged Panel", "merged_panel.csv"),
        ("Cleaned Panel", "cleaned_panel.csv"),
        ("Time Series Panel", "timeseries_panel.csv"),
    ]

    for name, filename in processed_files:
        path = config.paths.processed_dir / filename
        if path.exists():
            size = path.stat().st_size / 1024
            click.echo(f"  [x] {name}: {size:.1f} KB")
        else:
            click.echo(f"  [ ] {name}: Not found")

    # Check outputs
    click.echo("\nAnalysis Outputs (Cross-sectional):")
    output_files = [
        ("Phase 1 Results", "phase1/regression_results.txt"),
        ("Phase 1 Summary", "phase1/analysis_summary.txt"),
        ("Panel Regression", "phase1/panel_regression.txt"),
        ("Correlation Heatmap", "phase1/correlation_heatmap.png"),
        ("Coefficient Plot", "phase1/coefficient_plot.png"),
    ]

    for name, filepath in output_files:
        path = config.paths.outputs_dir / filepath
        if path.exists():
            click.echo(f"  [x] {name}")
        else:
            click.echo(f"  [ ] {name}")

    click.echo("\nAnalysis Outputs (Time Series):")
    ts_output_files = [
        ("TS Analysis Report", "timeseries/time_series_analysis.txt"),
        ("TS Results JSON", "timeseries/results.json"),
    ]

    for name, filepath in ts_output_files:
        path = config.paths.outputs_dir / filepath
        if path.exists():
            click.echo(f"  [x] {name}")
        else:
            click.echo(f"  [ ] {name}")


if __name__ == "__main__":
    cli()
