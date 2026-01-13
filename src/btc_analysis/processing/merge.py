"""Data processing and merging utilities."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from btc_analysis.config import get_config
from btc_analysis.data import (
    fetch_crypto_shadow_rate,
    fetch_chinn_ito,
    fetch_governance_indicators,
)
from btc_analysis.data.world_bank import fetch_development_indicators
from btc_analysis.data.crime import fetch_organized_crime_index
from btc_analysis.data.aml import create_fatf_panel

logger = logging.getLogger(__name__)

# ISO2 to ISO3 country code mapping
ISO2_TO_ISO3 = {
    "AD": "AND", "AE": "ARE", "AF": "AFG", "AG": "ATG", "AL": "ALB",
    "AM": "ARM", "AO": "AGO", "AR": "ARG", "AS": "ASM", "AT": "AUT",
    "AU": "AUS", "AW": "ABW", "AZ": "AZE", "BA": "BIH", "BB": "BRB",
    "BD": "BGD", "BE": "BEL", "BF": "BFA", "BG": "BGR", "BH": "BHR",
    "BI": "BDI", "BJ": "BEN", "BN": "BRN", "BO": "BOL", "BR": "BRA",
    "BS": "BHS", "BT": "BTN", "BW": "BWA", "BY": "BLR", "BZ": "BLZ",
    "CA": "CAN", "CD": "COD", "CF": "CAF", "CG": "COG", "CH": "CHE",
    "CI": "CIV", "CL": "CHL", "CM": "CMR", "CN": "CHN", "CO": "COL",
    "CR": "CRI", "CU": "CUB", "CV": "CPV", "CY": "CYP", "CZ": "CZE",
    "DE": "DEU", "DJ": "DJI", "DK": "DNK", "DM": "DMA", "DO": "DOM",
    "DZ": "DZA", "EC": "ECU", "EE": "EST", "EG": "EGY", "ER": "ERI",
    "ES": "ESP", "ET": "ETH", "FI": "FIN", "FJ": "FJI", "FM": "FSM",
    "FR": "FRA", "GA": "GAB", "GB": "GBR", "GD": "GRD", "GE": "GEO",
    "GH": "GHA", "GM": "GMB", "GN": "GIN", "GQ": "GNQ", "GR": "GRC",
    "GT": "GTM", "GW": "GNB", "GY": "GUY", "HK": "HKG", "HN": "HND",
    "HR": "HRV", "HT": "HTI", "HU": "HUN", "ID": "IDN", "IE": "IRL",
    "IL": "ISR", "IN": "IND", "IQ": "IRQ", "IR": "IRN", "IS": "ISL",
    "IT": "ITA", "JM": "JAM", "JO": "JOR", "JP": "JPN", "KE": "KEN",
    "KG": "KGZ", "KH": "KHM", "KI": "KIR", "KM": "COM", "KN": "KNA",
    "KP": "PRK", "KR": "KOR", "KW": "KWT", "KZ": "KAZ", "LA": "LAO",
    "LB": "LBN", "LC": "LCA", "LI": "LIE", "LK": "LKA", "LR": "LBR",
    "LS": "LSO", "LT": "LTU", "LU": "LUX", "LV": "LVA", "LY": "LBY",
    "MA": "MAR", "MC": "MCO", "MD": "MDA", "ME": "MNE", "MG": "MDG",
    "MH": "MHL", "MK": "MKD", "ML": "MLI", "MM": "MMR", "MN": "MNG",
    "MO": "MAC", "MR": "MRT", "MT": "MLT", "MU": "MUS", "MV": "MDV",
    "MW": "MWI", "MX": "MEX", "MY": "MYS", "MZ": "MOZ", "NA": "NAM",
    "NE": "NER", "NG": "NGA", "NI": "NIC", "NL": "NLD", "NO": "NOR",
    "NP": "NPL", "NR": "NRU", "NZ": "NZL", "OM": "OMN", "PA": "PAN",
    "PE": "PER", "PG": "PNG", "PH": "PHL", "PK": "PAK", "PL": "POL",
    "PT": "PRT", "PW": "PLW", "PY": "PRY", "QA": "QAT", "RO": "ROU",
    "RS": "SRB", "RU": "RUS", "RW": "RWA", "SA": "SAU", "SB": "SLB",
    "SC": "SYC", "SD": "SDN", "SE": "SWE", "SG": "SGP", "SI": "SVN",
    "SK": "SVK", "SL": "SLE", "SM": "SMR", "SN": "SEN", "SO": "SOM",
    "SR": "SUR", "SS": "SSD", "ST": "STP", "SV": "SLV", "SY": "SYR",
    "SZ": "SWZ", "TD": "TCD", "TG": "TGO", "TH": "THA", "TJ": "TJK",
    "TL": "TLS", "TM": "TKM", "TN": "TUN", "TO": "TON", "TR": "TUR",
    "TT": "TTO", "TV": "TUV", "TW": "TWN", "TZ": "TZA", "UA": "UKR",
    "UG": "UGA", "US": "USA", "UY": "URY", "UZ": "UZB", "VC": "VCT",
    "VE": "VEN", "VN": "VNM", "VU": "VUT", "WS": "WSM", "XK": "XKX",
    "YE": "YEM", "ZA": "ZAF", "ZM": "ZMB", "ZW": "ZWE",
}


def merge_datasets(
    output_path: Optional[Path] = None,
    start_year: int = 2015,
    end_year: int = 2023,
) -> pd.DataFrame:
    """
    Merge all datasets into a single panel for analysis.

    Combines:
    - IMF Crypto Shadow Rate (BTC premium)
    - Chinn-Ito capital control index
    - World Bank governance indicators
    - World Bank development indicators

    The merge is performed on country code and year.

    Args:
        output_path: Path to save the merged CSV.
        start_year: Start year to filter data.
        end_year: End year to filter data.

    Returns:
        Merged panel DataFrame ready for analysis.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.processed_dir / "merged_panel.csv"

    logger.info("Loading datasets for merging...")

    # Load each dataset
    shadow_rate = fetch_crypto_shadow_rate(start_year=start_year, end_year=end_year)
    chinn_ito = fetch_chinn_ito()
    wgi = fetch_governance_indicators(start_year=start_year, end_year=end_year)
    dev_indicators = fetch_development_indicators(start_year=start_year, end_year=end_year)

    # Standardize country codes across all datasets
    shadow_rate = _normalize_country_codes(shadow_rate)
    chinn_ito = _normalize_country_codes(chinn_ito)
    wgi = _normalize_country_codes(wgi)
    dev_indicators = _normalize_country_codes(dev_indicators)

    # Filter by year range
    chinn_ito = chinn_ito[
        (chinn_ito["year"] >= start_year) & (chinn_ito["year"] <= end_year)
    ]

    # Start with the base dataset (shadow rate or create country-year skeleton)
    if not shadow_rate.empty and "country_code" in shadow_rate.columns:
        base = shadow_rate[["country_code", "year", "btc_premium"]].copy()
    else:
        # Create skeleton from other datasets
        logger.info("Creating country-year skeleton from available datasets...")
        base = _create_skeleton(chinn_ito, wgi, start_year, end_year)

    logger.info(f"Base dataset shape: {base.shape}")

    # Merge Chinn-Ito capital controls
    if not chinn_ito.empty:
        chinn_ito_cols = ["country_code", "year"]
        if "kaopen" in chinn_ito.columns:
            chinn_ito_cols.append("kaopen")
        if "kaopen_norm" in chinn_ito.columns:
            chinn_ito_cols.append("kaopen_norm")
        if "capital_control_index" in chinn_ito.columns:
            chinn_ito_cols.append("capital_control_index")

        chinn_ito_subset = chinn_ito[chinn_ito_cols].drop_duplicates()

        base = pd.merge(
            base,
            chinn_ito_subset,
            on=["country_code", "year"],
            how="left",
        )
        logger.info(f"After Chinn-Ito merge: {base.shape}")

    # Merge governance indicators
    if not wgi.empty:
        wgi_cols = ["country_code", "year"] + [
            c for c in wgi.columns
            if c not in ["country_code", "country_name", "year"]
        ]
        wgi_subset = wgi[wgi_cols].drop_duplicates(subset=["country_code", "year"])

        base = pd.merge(
            base,
            wgi_subset,
            on=["country_code", "year"],
            how="left",
        )
        logger.info(f"After WGI merge: {base.shape}")

    # Merge development indicators
    if not dev_indicators.empty:
        dev_cols = ["country_code", "year"] + [
            c for c in dev_indicators.columns
            if c not in ["country_code", "country_name", "year"]
        ]
        dev_subset = dev_indicators[dev_cols].drop_duplicates(subset=["country_code", "year"])

        base = pd.merge(
            base,
            dev_subset,
            on=["country_code", "year"],
            how="left",
        )
        logger.info(f"After dev indicators merge: {base.shape}")

    # Fetch and merge crime data (now with time series: 2021, 2023, 2025)
    crime_data = fetch_organized_crime_index()
    if not crime_data.empty:
        crime_data = _normalize_country_codes(crime_data)
        crime_vars = [
            "criminality_score", "resilience_score", "financial_crimes",
            "human_trafficking", "human_smuggling", "arms_trafficking",
            "heroin_trade", "cocaine_trade", "cannabis_trade", "synthetic_drugs",
            "cybercrime", "flora_crimes", "fauna_crimes", "resource_crimes"
        ]

        # Check if we have time-varying data
        if "year" in crime_data.columns and crime_data["year"].nunique() > 1:
            logger.info(f"Crime data has {crime_data['year'].nunique()} years: {sorted(crime_data['year'].unique())}")
            crime_cols = ["country_code", "year"] + [c for c in crime_vars if c in crime_data.columns]
            crime_subset = crime_data[crime_cols].drop_duplicates(subset=["country_code", "year"])

            # Merge on country_code and year
            base = pd.merge(
                base,
                crime_subset,
                on=["country_code", "year"],
                how="left",
            )
        else:
            # Fallback: treat as time-invariant
            crime_cols = ["country_code"] + [c for c in crime_vars if c in crime_data.columns]
            crime_subset = crime_data[crime_cols].drop_duplicates(subset=["country_code"])
            base = pd.merge(
                base,
                crime_subset,
                on=["country_code"],
                how="left",
            )
        logger.info(f"After crime data merge: {base.shape}")

    # Create and merge FATF status indicators
    countries = base["country_code"].unique().tolist()
    fatf_data = create_fatf_panel(countries, start_year=start_year, end_year=end_year)
    base = pd.merge(
        base,
        fatf_data,
        on=["country_code", "year"],
        how="left",
    )
    # Fill NaN with 0 (countries not on FATF lists)
    for col in ["fatf_black", "fatf_grey", "fatf_any"]:
        if col in base.columns:
            base[col] = base[col].fillna(0).astype(int)
    logger.info(f"After FATF merge: {base.shape}")

    # Add country names back
    base = _add_country_names(base, chinn_ito, wgi)

    # Save merged dataset
    base.to_csv(output_path, index=False)
    logger.info(f"Saved merged panel to {output_path}")

    return base


def _normalize_country_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize country codes to ISO3 format."""
    if df.empty or "country_code" not in df.columns:
        return df

    df = df.copy()
    df["country_code"] = df["country_code"].astype(str).str.upper().str.strip()

    # Convert ISO2 codes to ISO3
    def to_iso3(code):
        code = str(code).upper().strip()
        # If it's already ISO3 (3 letters), keep it
        if len(code) == 3 and code.isalpha():
            return code
        # If it's ISO2, convert
        if len(code) == 2 and code in ISO2_TO_ISO3:
            return ISO2_TO_ISO3[code]
        return code

    df["country_code"] = df["country_code"].apply(to_iso3)

    # Filter to valid ISO3 codes (3 letters)
    valid_mask = df["country_code"].str.match(r"^[A-Z]{3}$", na=False)
    invalid_codes = df.loc[~valid_mask, "country_code"].unique()
    if len(invalid_codes) > 0:
        logger.debug(f"Filtered out invalid country codes: {invalid_codes[:10]}...")

    return df[valid_mask]


def _create_skeleton(
    chinn_ito: pd.DataFrame,
    wgi: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Create a country-year skeleton from available datasets."""
    countries = set()

    if not chinn_ito.empty and "country_code" in chinn_ito.columns:
        countries.update(chinn_ito["country_code"].dropna().unique())

    if not wgi.empty and "country_code" in wgi.columns:
        countries.update(wgi["country_code"].dropna().unique())

    years = range(start_year, end_year + 1)

    skeleton = pd.DataFrame(
        [(c, y) for c in countries for y in years],
        columns=["country_code", "year"],
    )

    # Add placeholder for BTC premium (to be filled if data available)
    skeleton["btc_premium"] = np.nan

    return skeleton


def _add_country_names(
    df: pd.DataFrame,
    chinn_ito: pd.DataFrame,
    wgi: pd.DataFrame,
) -> pd.DataFrame:
    """Add country names from source datasets."""
    country_names = {}

    if not chinn_ito.empty and "country_name" in chinn_ito.columns:
        for _, row in chinn_ito[["country_code", "country_name"]].drop_duplicates().iterrows():
            if pd.notna(row["country_code"]) and pd.notna(row["country_name"]):
                country_names[row["country_code"]] = row["country_name"]

    if not wgi.empty and "country_name" in wgi.columns:
        for _, row in wgi[["country_code", "country_name"]].drop_duplicates().iterrows():
            if pd.notna(row["country_code"]) and pd.notna(row["country_name"]):
                if row["country_code"] not in country_names:
                    country_names[row["country_code"]] = row["country_name"]

    df = df.copy()
    df["country_name"] = df["country_code"].map(country_names)

    # Reorder columns
    cols = ["country_code", "country_name", "year"]
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]

    return df


def clean_panel(
    df: pd.DataFrame,
    min_obs: int = 2,
    required_vars: Optional[list] = None,
) -> pd.DataFrame:
    """
    Clean the merged panel for analysis.

    Performs:
    - Drops countries with insufficient observations
    - Handles missing values
    - Creates derived variables
    - Winsorizes extreme values

    Args:
        df: Input panel DataFrame.
        min_obs: Minimum observations required per country.
        required_vars: List of variables that must be non-missing.

    Returns:
        Cleaned panel DataFrame.
    """
    if required_vars is None:
        required_vars = ["capital_control_index", "political_stability"]

    df = df.copy()

    # Log initial state
    n_countries = df["country_code"].nunique()
    n_obs = len(df)
    logger.info(f"Initial panel: {n_countries} countries, {n_obs} observations")

    # Count observations per country
    obs_count = df.groupby("country_code").size()
    valid_countries = obs_count[obs_count >= min_obs].index
    df = df[df["country_code"].isin(valid_countries)]

    logger.info(
        f"After min_obs filter ({min_obs}): "
        f"{df['country_code'].nunique()} countries, {len(df)} observations"
    )

    # Drop observations missing required variables
    for var in required_vars:
        if var in df.columns:
            before = len(df)
            df = df[df[var].notna()]
            after = len(df)
            if before > after:
                logger.info(f"Dropped {before - after} obs missing {var}")

    # Create log transformations for skewed variables
    if "gdp_per_capita" in df.columns:
        df["log_gdp_pc"] = np.log(df["gdp_per_capita"].clip(lower=1))

    if "btc_premium" in df.columns:
        # Log premium (handling potential negatives)
        df["log_btc_premium"] = np.log(df["btc_premium"].clip(lower=0.001) + 1)

    # Winsorize extreme values at 1st and 99th percentiles
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["year", "country_code"]:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)

    # Final summary
    logger.info(
        f"Final panel: {df['country_code'].nunique()} countries, "
        f"{len(df)} observations"
    )

    # Report missing data
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_report = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if not missing_report.empty:
        logger.info(f"Missing data (%):\n{missing_report.head(10)}")

    return df


def create_cross_section(
    df: pd.DataFrame,
    year: Optional[int] = None,
    agg_method: str = "mean",
) -> pd.DataFrame:
    """
    Create a cross-sectional dataset from the panel.

    Either selects a specific year or aggregates across years.

    Args:
        df: Panel DataFrame.
        year: Specific year to select. If None, aggregates across years.
        agg_method: Aggregation method ("mean", "median", "last").

    Returns:
        Cross-sectional DataFrame (one row per country).
    """
    if year is not None:
        return df[df["year"] == year].copy()

    # Aggregate across years
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "year"]

    if agg_method == "mean":
        agg = df.groupby("country_code")[numeric_cols].mean()
    elif agg_method == "median":
        agg = df.groupby("country_code")[numeric_cols].median()
    elif agg_method == "last":
        agg = df.sort_values("year").groupby("country_code")[numeric_cols].last()
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")

    agg = agg.reset_index()

    # Add country names
    country_names = df[["country_code", "country_name"]].drop_duplicates()
    agg = pd.merge(agg, country_names, on="country_code", how="left")

    return agg
