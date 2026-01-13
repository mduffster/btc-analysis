"""Capital control index data fetchers."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from btc_analysis.config import get_config

logger = logging.getLogger(__name__)

# ISO3 to ISO2 country code mapping for common countries
ISO3_TO_ISO2 = {
    "USA": "US", "GBR": "GB", "DEU": "DE", "FRA": "FR", "JPN": "JP",
    "CHN": "CN", "IND": "IN", "BRA": "BR", "RUS": "RU", "CAN": "CA",
    "AUS": "AU", "MEX": "MX", "KOR": "KR", "IDN": "ID", "TUR": "TR",
    "SAU": "SA", "ARG": "AR", "ZAF": "ZA", "NGA": "NG", "EGY": "EG",
    "PAK": "PK", "BGD": "BD", "VNM": "VN", "IRN": "IR", "THA": "TH",
    "MYS": "MY", "PHL": "PH", "COL": "CO", "POL": "PL", "UKR": "UA",
    "VEN": "VE", "PER": "PE", "CHL": "CL", "KAZ": "KZ", "IRQ": "IQ",
    "ROU": "RO", "NLD": "NL", "BEL": "BE", "GRC": "GR", "CZE": "CZ",
    "PRT": "PT", "SWE": "SE", "HUN": "HU", "AUT": "AT", "CHE": "CH",
    "ISR": "IL", "SGP": "SG", "HKG": "HK", "NOR": "NO", "DNK": "DK",
    "FIN": "FI", "IRL": "IE", "NZL": "NZ", "ARE": "AE", "KWT": "KW",
    "QAT": "QA", "TWN": "TW",
}


def fetch_chinn_ito(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Fetch the Chinn-Ito Financial Openness Index.

    The Chinn-Ito index (KAOPEN) measures a country's degree of capital account
    openness. Higher values indicate more open capital accounts (less restrictive).
    The index is based on binary dummy variables from the IMF's Annual Report on
    Exchange Arrangements and Exchange Restrictions (AREAER).

    Source: http://web.pdx.edu/~ito/Chinn-Ito_website.htm
    Coverage: 182 countries, 1970-2022

    Args:
        output_path: Path to save the CSV output.

    Returns:
        DataFrame with columns: country_code, country_name, year, kaopen, ka_open_norm
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "chinn_ito.csv"

    # Check for existing local file first
    if output_path.exists():
        logger.info(f"Loading existing Chinn-Ito data from {output_path}")
        return pd.read_csv(output_path)

    # Download from source
    url = config.api.chinn_ito_url

    logger.info(f"Downloading Chinn-Ito index from {url}...")

    try:
        response = requests.get(url, timeout=config.api.timeout)
        response.raise_for_status()

        # Save raw file
        raw_file = config.paths.raw_dir / "kaopen_2022.xls"
        with open(raw_file, "wb") as f:
            f.write(response.content)

        # Read Excel file
        df = pd.read_excel(raw_file, sheet_name=0)

        # Standardize column names
        df = _standardize_chinn_ito(df)

        # Save processed version
        df.to_csv(output_path, index=False)
        logger.info(f"Saved Chinn-Ito data to {output_path}")

        return df

    except requests.RequestException as e:
        logger.error(f"Failed to download Chinn-Ito data: {e}")
        logger.info("Attempting alternative download method...")

        # Try alternative URL or provide manual instructions
        return _create_placeholder_chinn_ito()

    except Exception as e:
        logger.error(f"Error processing Chinn-Ito data: {e}")
        return _create_placeholder_chinn_ito()


def _standardize_chinn_ito(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Chinn-Ito DataFrame columns and format."""
    df = df.copy()

    # Rename specific columns we need
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower == "ccode":
            rename_map[col] = "country_code"
        elif col_lower == "country_name":
            rename_map[col] = "country_name"
        elif col_lower == "country":
            rename_map[col] = "country_name"
        elif col_lower == "year":
            rename_map[col] = "year"
        elif col_lower == "kaopen":
            rename_map[col] = "kaopen"
        # ka_open is the normalized version (0-1 scale) - keep as separate column
        elif col_lower == "ka_open":
            rename_map[col] = "kaopen_norm_orig"

    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["country_name", "year", "kaopen"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns in Chinn-Ito data: {missing}")
        logger.info(f"Available columns: {list(df.columns)}")

    # Use original normalized version if available, otherwise compute
    if "kaopen_norm_orig" in df.columns:
        df["kaopen_norm"] = df["kaopen_norm_orig"]
    elif "kaopen" in df.columns:
        ka_min = df["kaopen"].min()
        ka_max = df["kaopen"].max()
        df["kaopen_norm"] = (df["kaopen"] - ka_min) / (ka_max - ka_min)

    # Create capital control index (inverse of openness)
    # Higher values = more capital controls
    if "kaopen_norm" in df.columns:
        df["capital_control_index"] = 1 - df["kaopen_norm"]

    # Add ISO codes if country name available
    if "country_name" in df.columns and "country_code" not in df.columns:
        df["country_code"] = df["country_name"].map(_country_name_to_iso3)

    return df


def _country_name_to_iso3(name: str) -> str:
    """Map country name to ISO3 code."""
    name_to_iso = {
        "United States": "USA",
        "United Kingdom": "GBR",
        "Germany": "DEU",
        "France": "FRA",
        "Japan": "JPN",
        "China": "CHN",
        "India": "IND",
        "Brazil": "BRA",
        "Russia": "RUS",
        "Russian Federation": "RUS",
        "Canada": "CAN",
        "Australia": "AUS",
        "Mexico": "MX",
        "Korea": "KOR",
        "Korea, Rep.": "KOR",
        "South Korea": "KOR",
        "Indonesia": "IDN",
        "Turkey": "TUR",
        "Saudi Arabia": "SAU",
        "Argentina": "ARG",
        "South Africa": "ZAF",
        "Nigeria": "NGA",
        "Egypt": "EGY",
        "Pakistan": "PAK",
        "Bangladesh": "BGD",
        "Vietnam": "VNM",
        "Iran": "IRN",
        "Thailand": "THA",
        "Malaysia": "MYS",
        "Philippines": "PHL",
        "Colombia": "COL",
        "Poland": "POL",
        "Ukraine": "UKR",
        "Venezuela": "VEN",
        "Peru": "PER",
        "Chile": "CHL",
        "Kazakhstan": "KAZ",
        "Iraq": "IRQ",
        "Romania": "ROU",
        "Netherlands": "NLD",
        "Belgium": "BEL",
        "Greece": "GRC",
        "Czech Republic": "CZE",
        "Portugal": "PRT",
        "Sweden": "SWE",
        "Hungary": "HUN",
        "Austria": "AUT",
        "Switzerland": "CHE",
        "Israel": "ISR",
        "Singapore": "SGP",
        "Hong Kong": "HKG",
        "Norway": "NOR",
        "Denmark": "DNK",
        "Finland": "FIN",
        "Ireland": "IRL",
        "New Zealand": "NZL",
        "United Arab Emirates": "ARE",
        "Kuwait": "KWT",
        "Qatar": "QAT",
        "Taiwan": "TWN",
    }
    return name_to_iso.get(name, "")


def _create_placeholder_chinn_ito() -> pd.DataFrame:
    """Create placeholder DataFrame with expected schema."""
    logger.warning(
        "Creating placeholder Chinn-Ito data. Please download manually from: "
        "http://web.pdx.edu/~ito/Chinn-Ito_website.htm"
    )
    return pd.DataFrame(
        columns=[
            "country_code",
            "country_name",
            "year",
            "kaopen",
            "kaopen_norm",
            "capital_control_index",
        ]
    )


def fetch_fkrsu(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Fetch Fernández-Klein-Rebucci-Schindler-Uribe Capital Controls Dataset.

    This dataset provides detailed capital control restrictions on both inflows
    and outflows across 10 asset categories for 100 countries.

    Source: www.columbia.edu/~mu2166/fkrsu
    Coverage: 100 countries, 1995-2021

    Args:
        output_path: Path to save the CSV output.

    Returns:
        DataFrame with detailed capital control measures by asset type.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "fkrsu.csv"

    # Check for existing local file
    if output_path.exists():
        logger.info(f"Loading existing FKRSU data from {output_path}")
        return pd.read_csv(output_path)

    # The FKRSU dataset is typically distributed as Excel/Stata files
    # from the Columbia University website
    fkrsu_urls = [
        "https://www.columbia.edu/~mu2166/fkrsu/fkrsu_dataset.xlsx",
        "https://www.columbia.edu/~mu2166/fkrsu/fkrsu.xlsx",
    ]

    for url in fkrsu_urls:
        try:
            logger.info(f"Attempting to download FKRSU data from {url}...")
            response = requests.get(url, timeout=config.api.timeout)
            if response.status_code == 200:
                raw_file = config.paths.raw_dir / "fkrsu_raw.xlsx"
                with open(raw_file, "wb") as f:
                    f.write(response.content)

                df = pd.read_excel(raw_file)
                df = _standardize_fkrsu(df)
                df.to_csv(output_path, index=False)
                logger.info(f"Saved FKRSU data to {output_path}")
                return df

        except Exception as e:
            logger.warning(f"Could not fetch from {url}: {e}")
            continue

    logger.warning(
        "Could not download FKRSU data automatically. Please download from: "
        "www.columbia.edu/~mu2166/fkrsu"
    )
    return pd.DataFrame(
        columns=[
            "country_code",
            "country_name",
            "year",
            "kai",  # Capital inflow restrictions index
            "kao",  # Capital outflow restrictions index
            "ka",   # Overall capital account restrictions
        ]
    )


def _standardize_fkrsu(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize FKRSU DataFrame columns."""
    column_map = {
        "ifscode": "imf_code",
        "country": "country_name",
        "year": "year",
        "kai": "kai",
        "kao": "kao",
        "ka": "ka",
    }

    df = df.rename(
        columns={k: v for k, v in column_map.items() if k.lower() in [c.lower() for c in df.columns]}
    )

    return df
