"""Crime and AML risk data fetchers."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from btc_analysis.config import get_config

logger = logging.getLogger(__name__)

# Country name to ISO3 mapping for common variations
COUNTRY_NAME_TO_ISO3 = {
    "United States": "USA",
    "United States of America": "USA",
    "United Kingdom": "GBR",
    "Russia": "RUS",
    "Russian Federation": "RUS",
    "South Korea": "KOR",
    "Republic of Korea": "KOR",
    "Korea, Republic of": "KOR",
    "North Korea": "PRK",
    "China": "CHN",
    "Taiwan": "TWN",
    "Hong Kong": "HKG",
    "Macau": "MAC",
    "Vietnam": "VNM",
    "Viet Nam": "VNM",
    "Iran": "IRN",
    "Syria": "SYR",
    "Venezuela": "VEN",
    "Turkey": "TUR",
    "Türkiye": "TUR",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "UAE": "ARE",
    "United Arab Emirates": "ARE",
    "Saudi Arabia": "SAU",
    "South Africa": "ZAF",
    "Argentina": "ARG",
    "Brazil": "BRA",
    "Mexico": "MEX",
    "Colombia": "COL",
    "Peru": "PER",
    "Chile": "CHL",
    "Nigeria": "NGA",
    "Kenya": "KEN",
    "Egypt": "EGY",
    "Morocco": "MAR",
    "Algeria": "DZA",
    "Tunisia": "TUN",
    "Libya": "LBY",
    "Pakistan": "PAK",
    "India": "IND",
    "Bangladesh": "BGD",
    "Indonesia": "IDN",
    "Malaysia": "MYS",
    "Philippines": "PHL",
    "Thailand": "THA",
    "Singapore": "SGP",
    "Japan": "JPN",
    "Australia": "AUS",
    "New Zealand": "NZL",
    "Canada": "CAN",
    "Germany": "DEU",
    "France": "FRA",
    "Italy": "ITA",
    "Spain": "ESP",
    "Netherlands": "NLD",
    "Belgium": "BEL",
    "Switzerland": "CHE",
    "Austria": "AUT",
    "Sweden": "SWE",
    "Norway": "NOR",
    "Denmark": "DNK",
    "Finland": "FIN",
    "Poland": "POL",
    "Ukraine": "UKR",
    "Romania": "ROU",
    "Hungary": "HUN",
    "Greece": "GRC",
    "Portugal": "PRT",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Georgia": "GEO",
    "Kazakhstan": "KAZ",
    "Uzbekistan": "UZB",
    "Belarus": "BLR",
}


def fetch_organized_crime_index(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Fetch the Global Organized Crime Index data.

    The index measures criminality (how much crime) and resilience (how well
    countries combat it). Higher criminality scores indicate more organized crime.

    Source: Global Initiative Against Transnational Organized Crime
    URL: https://ocindex.net/

    Crime categories include:
    - Human trafficking
    - Human smuggling
    - Arms trafficking
    - Flora/fauna crimes
    - Non-renewable resource crimes
    - Heroin trade
    - Cocaine trade
    - Cannabis trade
    - Synthetic drug trade
    - Cyber-dependent crimes
    - Financial crimes

    Args:
        output_path: Path to save the CSV output.

    Returns:
        DataFrame with columns: country_code, country_name, year,
        criminality_score, resilience_score, and crime subcategories.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "organized_crime_index.csv"

    # Check for existing data
    if output_path.exists():
        logger.info(f"Loading existing crime index data from {output_path}")
        return pd.read_csv(output_path)

    # Download from source
    url = "https://ocindex.net/assets/downloads/global_oc_index.xlsx"

    logger.info(f"Downloading Global Organized Crime Index from {url}...")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Save raw file
        raw_file = config.paths.raw_dir / "global_oc_index.xlsx"
        with open(raw_file, "wb") as f:
            f.write(response.content)

        # Read and process
        df = _process_crime_index_excel(raw_file)

        if not df.empty:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved crime index data to {output_path}")

        return df

    except requests.RequestException as e:
        logger.error(f"Failed to download crime index: {e}")
        return _create_placeholder_crime_index()

    except Exception as e:
        logger.error(f"Error processing crime index data: {e}")
        return _create_placeholder_crime_index()


def _process_crime_index_excel(filepath: Path) -> pd.DataFrame:
    """Process the Global OC Index Excel file - load all available years."""
    try:
        xlsx = pd.ExcelFile(filepath)
        logger.info(f"Excel sheets: {xlsx.sheet_names}")

        # Map sheet names to years
        year_sheets = {}
        for sheet_name in xlsx.sheet_names:
            # Extract year from sheet name (e.g., "2021_dataset" -> 2021)
            for year in [2021, 2023, 2025]:
                if str(year) in sheet_name and "dataset" in sheet_name.lower():
                    year_sheets[year] = sheet_name
                    break

        logger.info(f"Found year sheets: {year_sheets}")

        if not year_sheets:
            # Fallback: try to use any sheet with country data
            for sheet_name in xlsx.sheet_names:
                try:
                    temp_df = pd.read_excel(xlsx, sheet_name=sheet_name)
                    if "country" in str(temp_df.columns).lower() or len(temp_df) > 100:
                        logger.info(f"Using fallback sheet: {sheet_name}")
                        df = _standardize_crime_columns(temp_df)
                        return df
                except Exception:
                    continue
            return pd.DataFrame()

        # Load all years and combine into panel
        all_years = []
        for year, sheet_name in year_sheets.items():
            try:
                temp_df = pd.read_excel(xlsx, sheet_name=sheet_name)
                temp_df = _standardize_crime_columns(temp_df)
                temp_df["year"] = year
                all_years.append(temp_df)
                logger.info(f"Loaded {len(temp_df)} countries for year {year}")
            except Exception as e:
                logger.warning(f"Could not load sheet {sheet_name}: {e}")

        if not all_years:
            return pd.DataFrame()

        df = pd.concat(all_years, ignore_index=True)
        logger.info(f"Combined crime data: {len(df)} total observations across {len(year_sheets)} years")

        return df

    except Exception as e:
        logger.error(f"Error processing crime index Excel: {e}")
        return pd.DataFrame()


def _standardize_crime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and add ISO3 codes."""
    df = df.copy()

    # Common column name mappings (case-insensitive)
    # Note: 2021 dataset uses commas (e.g., "Criminality avg,") while 2023/2025 use periods
    column_map = {
        "country": "country_name",
        "country_name": "country_name",
        "name": "country_name",
        "iso": "country_code",
        "iso3": "country_code",
        "iso_code": "country_code",
        "country_code": "country_code",
        "criminality": "criminality_score",
        "criminality_score": "criminality_score",
        "overall_criminality": "criminality_score",
        "criminality_avg.": "criminality_score",  # 2023/2025 format
        "criminality_avg,": "criminality_score",  # 2021 format
        "resilience": "resilience_score",  # 2021 format (just "Resilience")
        "resilience_score": "resilience_score",
        "overall_resilience": "resilience_score",
        "resilience_avg.": "resilience_score",  # 2025 format
        "resilience_avg,": "resilience_score",  # 2023 format
        "year": "year",
        # Crime subcategories
        "human_trafficking": "human_trafficking",
        "human_smuggling": "human_smuggling",
        "arms_trafficking": "arms_trafficking",
        "flora_crimes": "flora_crimes",
        "fauna_crimes": "fauna_crimes",
        "resource_crimes": "resource_crimes",
        "non_renewable_resource_crimes": "resource_crimes",
        "non-renewable_resource_crimes": "resource_crimes",
        "heroin_trade": "heroin_trade",
        "cocaine_trade": "cocaine_trade",
        "cannabis_trade": "cannabis_trade",
        "synthetic_drugs": "synthetic_drugs",
        "synthetic_drug_trade": "synthetic_drugs",
        "cybercrime": "cybercrime",
        "cyber_dependent_crimes": "cybercrime",
        "cyber-dependent_crimes": "cybercrime",
        "financial_crimes": "financial_crimes",
    }

    # Rename columns
    new_cols = {}
    for col in df.columns:
        col_lower = col.lower().strip().replace(" ", "_").replace("-", "_")
        if col_lower in column_map:
            new_cols[col] = column_map[col_lower]

    df = df.rename(columns=new_cols)

    # Add ISO3 codes if not present
    if "country_code" not in df.columns and "country_name" in df.columns:
        df["country_code"] = df["country_name"].map(_get_iso3_code)

    # Add year column if not present (assume most recent)
    if "year" not in df.columns:
        df["year"] = 2023  # Most recent index year

    # Filter to rows with valid data
    if "criminality_score" in df.columns:
        df = df[df["criminality_score"].notna()]

    logger.info(f"Processed crime data: {len(df)} countries")

    return df


def _get_iso3_code(country_name: str) -> Optional[str]:
    """Convert country name to ISO3 code using pycountry."""
    if pd.isna(country_name):
        return None

    name = str(country_name).strip()

    # Direct lookup in our mapping first
    if name in COUNTRY_NAME_TO_ISO3:
        return COUNTRY_NAME_TO_ISO3[name]

    # Handle special cases
    special_cases = {
        "Congo, Dem. Rep.": "COD",
        "Congo, Rep.": "COG",
        "Democratic Republic of the Congo": "COD",
        "Republic of the Congo": "COG",
        "Ivory Coast": "CIV",
        "Côte d'Ivoire": "CIV",
        "eSwatini": "SWZ",
        "Swaziland": "SWZ",
        "Myanmar": "MMR",
        "Burma": "MMR",
        "Laos": "LAO",
        "Lao PDR": "LAO",
        "Sao Tome and Principe": "STP",
        "São Tomé and Príncipe": "STP",
        "Cape Verde": "CPV",
        "Cabo Verde": "CPV",
        "Timor-Leste": "TLS",
        "East Timor": "TLS",
        "Palestine": "PSE",
        "North Macedonia": "MKD",
        "Macedonia": "MKD",
        "Moldova": "MDA",
        "Kyrgyzstan": "KGZ",
        "Tajikistan": "TJK",
        "Turkmenistan": "TKM",
        "Bosnia and Herzegovina": "BIH",
        "Serbia": "SRB",
        "Montenegro": "MNE",
        "Kosovo": "XKX",
        "Trinidad and Tobago": "TTO",
        "Dominican Republic": "DOM",
        "El Salvador": "SLV",
        "Costa Rica": "CRI",
        "Panama": "PAN",
        "Guatemala": "GTM",
        "Honduras": "HND",
        "Nicaragua": "NIC",
        "Bolivia": "BOL",
        "Paraguay": "PRY",
        "Uruguay": "URY",
        "Ecuador": "ECU",
        "Guyana": "GUY",
        "Suriname": "SUR",
        "Brunei": "BRN",
        "Cambodia": "KHM",
        "Sri Lanka": "LKA",
        "Nepal": "NPL",
        "Afghanistan": "AFG",
        "Iraq": "IRQ",
        "Jordan": "JOR",
        "Lebanon": "LBN",
        "Kuwait": "KWT",
        "Qatar": "QAT",
        "Bahrain": "BHR",
        "Oman": "OMN",
        "Yemen": "YEM",
        "Cyprus": "CYP",
        "Malta": "MLT",
        "Iceland": "ISL",
        "Luxembourg": "LUX",
        "Slovenia": "SVN",
        "Slovakia": "SVK",
        "Estonia": "EST",
        "Latvia": "LVA",
        "Lithuania": "LTU",
        "Croatia": "HRV",
        "Bulgaria": "BGR",
        "Albania": "ALB",
        "Armenia": "ARM",
        "Azerbaijan": "AZE",
        "Ghana": "GHA",
        "Senegal": "SEN",
        "Mali": "MLI",
        "Burkina Faso": "BFA",
        "Niger": "NER",
        "Benin": "BEN",
        "Togo": "TGO",
        "Guinea": "GIN",
        "Sierra Leone": "SLE",
        "Liberia": "LBR",
        "Gambia": "GMB",
        "Guinea-Bissau": "GNB",
        "Mauritius": "MUS",
        "Madagascar": "MDG",
        "Mozambique": "MOZ",
        "Zimbabwe": "ZWE",
        "Zambia": "ZMB",
        "Malawi": "MWI",
        "Namibia": "NAM",
        "Botswana": "BWA",
        "Angola": "AGO",
        "Cameroon": "CMR",
        "Gabon": "GAB",
        "Chad": "TCD",
        "Central African Republic": "CAF",
        "Burundi": "BDI",
        "Rwanda": "RWA",
        "Uganda": "UGA",
        "Tanzania": "TZA",
        "Ethiopia": "ETH",
        "Eritrea": "ERI",
        "Djibouti": "DJI",
        "Somalia": "SOM",
        "South Sudan": "SSD",
        "Sudan": "SDN",
        "Lesotho": "LSO",
        "Comoros": "COM",
        "Equatorial Guinea": "GNQ",
        "Papua New Guinea": "PNG",
        "Fiji": "FJI",
        "Solomon Islands": "SLB",
        "Vanuatu": "VUT",
        "Samoa": "WSM",
        "Tonga": "TON",
        "Micronesia": "FSM",
        "Mongolia": "MNG",
        "Jamaica": "JAM",
        "Haiti": "HTI",
        "Cuba": "CUB",
        "Bahamas": "BHS",
        "Barbados": "BRB",
    }

    if name in special_cases:
        return special_cases[name]

    # Try pycountry
    try:
        import pycountry

        # Try exact match
        country = pycountry.countries.get(name=name)
        if country:
            return country.alpha_3

        # Try fuzzy search
        results = pycountry.countries.search_fuzzy(name)
        if results:
            return results[0].alpha_3

    except Exception:
        pass

    # Try case-insensitive match in our mapping
    name_lower = name.lower()
    for key, value in COUNTRY_NAME_TO_ISO3.items():
        if key.lower() == name_lower:
            return value

    return None


def _create_placeholder_crime_index() -> pd.DataFrame:
    """Create placeholder DataFrame with expected schema."""
    logger.warning(
        "Creating placeholder crime index data. Please download manually from: "
        "https://ocindex.net/downloads"
    )
    return pd.DataFrame(
        columns=[
            "country_code",
            "country_name",
            "year",
            "criminality_score",
            "resilience_score",
            "financial_crimes",
        ]
    )


def fetch_basel_aml_index(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Fetch Basel AML Index data.

    The Basel AML Index measures money laundering and terrorist financing
    risk for countries. Higher scores indicate higher risk.

    Note: This may require manual download from Basel Institute.

    Args:
        output_path: Path to save the CSV output.

    Returns:
        DataFrame with AML risk scores by country.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "basel_aml_index.csv"

    # Check for existing data
    if output_path.exists():
        logger.info(f"Loading existing Basel AML data from {output_path}")
        return pd.read_csv(output_path)

    # Basel AML Index typically requires registration to download
    logger.warning(
        "Basel AML Index requires manual download from: "
        "https://index.baselgovernance.org/download"
    )

    return pd.DataFrame(
        columns=["country_code", "country_name", "year", "aml_risk_score"]
    )
