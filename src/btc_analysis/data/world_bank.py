"""World Bank data fetchers for governance and development indicators."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from btc_analysis.config import get_config

logger = logging.getLogger(__name__)

# World Bank indicator codes for Worldwide Governance Indicators
WGI_INDICATORS = {
    "CC.EST": "control_of_corruption",      # Control of Corruption
    "GE.EST": "government_effectiveness",   # Government Effectiveness
    "PV.EST": "political_stability",        # Political Stability and Absence of Violence
    "RQ.EST": "regulatory_quality",         # Regulatory Quality
    "RL.EST": "rule_of_law",                # Rule of Law
    "VA.EST": "voice_accountability",       # Voice and Accountability
}

# Additional development indicators
DEV_INDICATORS = {
    "NY.GDP.PCAP.CD": "gdp_per_capita",           # GDP per capita (current US$)
    "IT.NET.USER.ZS": "internet_penetration",     # Individuals using the Internet (% of pop)
    "FP.CPI.TOTL.ZG": "inflation_wb",             # Inflation, consumer prices (annual %)
    "FM.LBL.BMNY.GD.ZS": "broad_money_gdp",       # Broad money (% of GDP) - financial depth
}


def fetch_governance_indicators(
    output_path: Optional[Path] = None,
    start_year: int = 2010,
    end_year: int = 2023,
) -> pd.DataFrame:
    """
    Fetch World Bank Worldwide Governance Indicators (WGI).

    The WGI measures six dimensions of governance:
    - Voice and Accountability
    - Political Stability and Absence of Violence
    - Government Effectiveness
    - Regulatory Quality
    - Rule of Law
    - Control of Corruption

    Values range from approximately -2.5 (weak) to 2.5 (strong) governance.

    Source: https://databank.worldbank.org/source/worldwide-governance-indicators

    Args:
        output_path: Path to save the CSV output.
        start_year: Start year for the data range.
        end_year: End year for the data range.

    Returns:
        DataFrame with governance indicators by country and year.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "world_bank_wgi.csv"

    # Check for existing data
    if output_path.exists():
        logger.info(f"Loading existing WGI data from {output_path}")
        return pd.read_csv(output_path)

    all_data = []

    for indicator_code, indicator_name in WGI_INDICATORS.items():
        logger.info(f"Fetching {indicator_name} ({indicator_code})...")
        df = _fetch_wb_indicator(
            indicator_code,
            start_year,
            end_year,
            config.api.world_bank_base_url,
            config.api.timeout,
        )
        if df is not None and not df.empty:
            df["indicator"] = indicator_name
            all_data.append(df)

    if not all_data:
        logger.error("Failed to fetch any governance indicators")
        return _create_placeholder_wgi()

    # Combine all indicators
    combined = pd.concat(all_data, ignore_index=True)

    # Pivot to wide format (one column per indicator)
    pivoted = combined.pivot_table(
        index=["country_code", "country_name", "year"],
        columns="indicator",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivoted.columns.name = None

    pivoted.to_csv(output_path, index=False)
    logger.info(f"Saved WGI data to {output_path}")

    return pivoted


def fetch_development_indicators(
    output_path: Optional[Path] = None,
    start_year: int = 2010,
    end_year: int = 2023,
) -> pd.DataFrame:
    """
    Fetch additional World Bank development indicators.

    Includes GDP per capita, internet penetration, and financial depth measures
    to use as control variables in the regression.

    Args:
        output_path: Path to save the CSV output.
        start_year: Start year for the data range.
        end_year: End year for the data range.

    Returns:
        DataFrame with development indicators by country and year.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "world_bank_dev.csv"

    if output_path.exists():
        logger.info(f"Loading existing development indicators from {output_path}")
        return pd.read_csv(output_path)

    all_data = []

    for indicator_code, indicator_name in DEV_INDICATORS.items():
        logger.info(f"Fetching {indicator_name} ({indicator_code})...")
        df = _fetch_wb_indicator(
            indicator_code,
            start_year,
            end_year,
            config.api.world_bank_base_url,
            config.api.timeout,
        )
        if df is not None and not df.empty:
            df["indicator"] = indicator_name
            all_data.append(df)

    if not all_data:
        logger.error("Failed to fetch any development indicators")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    pivoted = combined.pivot_table(
        index=["country_code", "country_name", "year"],
        columns="indicator",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivoted.columns.name = None

    pivoted.to_csv(output_path, index=False)
    logger.info(f"Saved development indicators to {output_path}")

    return pivoted


def _fetch_wb_indicator(
    indicator: str,
    start_year: int,
    end_year: int,
    base_url: str,
    timeout: int,
) -> Optional[pd.DataFrame]:
    """
    Fetch a single indicator from World Bank API.

    Args:
        indicator: World Bank indicator code (e.g., "NY.GDP.PCAP.CD")
        start_year: Start year
        end_year: End year
        base_url: World Bank API base URL
        timeout: Request timeout in seconds

    Returns:
        DataFrame with columns: country_code, country_name, year, value
    """
    # World Bank API endpoint
    url = f"{base_url}/country/all/indicator/{indicator}"

    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": 20000,  # Get all data in one request
    }

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()

        data = response.json()

        # World Bank API returns [metadata, data] structure
        if not data or len(data) < 2 or not data[1]:
            logger.warning(f"No data returned for indicator {indicator}")
            return None

        records = []
        for item in data[1]:
            if item.get("value") is not None:
                records.append(
                    {
                        "country_code": item.get("country", {}).get("id", ""),
                        "country_name": item.get("country", {}).get("value", ""),
                        "year": int(item.get("date", 0)),
                        "value": float(item.get("value")),
                    }
                )

        return pd.DataFrame(records)

    except requests.RequestException as e:
        logger.error(f"Failed to fetch indicator {indicator}: {e}")
        return None
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing response for indicator {indicator}: {e}")
        return None


def _create_placeholder_wgi() -> pd.DataFrame:
    """Create placeholder DataFrame with expected WGI schema."""
    columns = ["country_code", "country_name", "year"] + list(WGI_INDICATORS.values())
    return pd.DataFrame(columns=columns)


def fetch_all_world_bank(
    output_path: Optional[Path] = None,
    start_year: int = 2010,
    end_year: int = 2023,
) -> pd.DataFrame:
    """
    Fetch all World Bank indicators (governance + development) and merge.

    Args:
        output_path: Path to save the merged CSV.
        start_year: Start year for the data range.
        end_year: End year for the data range.

    Returns:
        DataFrame with all World Bank indicators merged.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "world_bank_all.csv"

    wgi = fetch_governance_indicators(start_year=start_year, end_year=end_year)
    dev = fetch_development_indicators(start_year=start_year, end_year=end_year)

    if wgi.empty and dev.empty:
        return pd.DataFrame()

    if wgi.empty:
        return dev

    if dev.empty:
        return wgi

    # Merge on country and year
    merged = pd.merge(
        wgi,
        dev,
        on=["country_code", "country_name", "year"],
        how="outer",
    )

    merged.to_csv(output_path, index=False)
    logger.info(f"Saved merged World Bank data to {output_path}")

    return merged
