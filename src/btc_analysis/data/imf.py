"""IMF data fetchers for crypto shadow rates and CPI data."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from btc_analysis.config import get_config

logger = logging.getLogger(__name__)

# Currency ISO to Country ISO3 mapping
CURRENCY_TO_COUNTRY = {
    "AED": "ARE", "ARS": "ARG", "AUD": "AUS", "BRL": "BRA", "BYN": "BLR",
    "CAD": "CAN", "CHF": "CHE", "CLP": "CHL", "CNY": "CHN", "COP": "COL",
    "CZK": "CZE", "EUR": "EUR", "GBP": "GBR", "GEL": "GEO", "HKD": "HKG",
    "IDR": "IDN", "ILS": "ISR", "INR": "IND", "JPY": "JPN", "KES": "KEN",
    "KGS": "KGZ", "KRW": "KOR", "KZT": "KAZ", "MXN": "MEX", "MYR": "MYS",
    "NGN": "NGA", "NZD": "NZL", "PEN": "PER", "PHP": "PHL", "PLN": "POL",
    "RUB": "RUS", "SGD": "SGP", "THB": "THA", "TRY": "TUR", "TWD": "TWN",
    "UAH": "UKR", "UGX": "UGA", "USD": "USA", "VND": "VNM", "ZAR": "ZAF",
    "ZMW": "ZMB",
}


def fetch_crypto_shadow_rate(
    output_path: Optional[Path] = None,
    start_year: int = 2015,
    end_year: int = 2023,
) -> pd.DataFrame:
    """
    Fetch IMF Crypto Shadow Rate (parallel exchange rate) data.

    This dataset provides Bitcoin premiums by country, measuring the price of BTC
    on local markets relative to the US market. Higher premiums indicate willingness
    to pay more for foreign currency access in capital-controlled countries.

    Source: Graf von Luckner, Koepke, & Sgherri (2024), "Crypto as a Marketplace
    for Capital Flight," IMF Working Paper

    The data is available through the IMF's SDMX API. The crypto shadow rate
    dataflow is part of the IMF's experimental/research datasets.

    Args:
        output_path: Path to save the CSV output. If None, uses default from config.
        start_year: Start year for the data range.
        end_year: End year for the data range.

    Returns:
        DataFrame with columns: country_code, country_name, year, btc_premium
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "imf_crypto_shadow_rate.csv"

    # The IMF Crypto Shadow Rate data may be available through:
    # 1. Direct IMF SDMX API with specific dataflow ID
    # 2. IMF DataMapper / IFS database
    # 3. Supplementary data from the working paper

    # First, try to fetch from the IMF's data portal
    # The exact dataflow ID needs to be discovered from IMF's catalog
    base_url = config.api.imf_base_url

    # Try to get the dataflow catalog to find crypto-related datasets
    catalog_url = f"{base_url}/dataflow/IMF"

    try:
        logger.info("Fetching IMF dataflow catalog to locate crypto shadow rate data...")
        response = requests.get(
            catalog_url,
            headers={"Accept": "application/vnd.sdmx.structure+json;version=1.0"},
            timeout=config.api.timeout,
        )
        response.raise_for_status()

        # Parse the catalog to find relevant dataflows
        catalog = response.json()
        dataflows = catalog.get("data", {}).get("dataflows", [])

        # Look for crypto or parallel exchange rate related dataflows
        crypto_flows = [
            df
            for df in dataflows
            if any(
                term in str(df).lower()
                for term in ["crypto", "bitcoin", "parallel", "shadow"]
            )
        ]

        if crypto_flows:
            logger.info(f"Found potential crypto dataflows: {crypto_flows}")
        else:
            logger.warning("No crypto-specific dataflows found in IMF catalog")

    except requests.RequestException as e:
        logger.warning(f"Could not fetch IMF catalog: {e}")

    # Alternative approach: The working paper data may be available as supplementary
    # material or through the IMF's research data portal

    # For now, create a placeholder structure and document the expected format
    # The actual data would need to be:
    # 1. Downloaded from IMF working paper supplementary materials
    # 2. Fetched once the exact SDMX dataflow ID is identified
    # 3. Or manually compiled from the paper's methodology

    logger.info("Checking for local crypto shadow rate data files...")

    # Check for Excel file from IMF working paper (primary source)
    excel_patterns = list(config.paths.raw_dir.glob("*Crypto*Exchange*.xlsx"))
    if excel_patterns:
        excel_file = excel_patterns[0]
        logger.info(f"Found crypto exchange rate Excel file: {excel_file}")
        df = _load_crypto_excel(excel_file, start_year, end_year)
        if df is not None and not df.empty:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed crypto shadow rate data to {output_path}")
            return df

    # Check for CSV files
    local_sources = [
        config.paths.raw_dir / "crypto_shadow_rate_manual.csv",
        config.paths.raw_dir / "imf_crypto_shadow_rate.csv",
        config.paths.base_dir / "crypto_shadow_rate.csv",
    ]

    for source in local_sources:
        if source.exists():
            logger.info(f"Loading existing crypto shadow rate data from {source}")
            df = pd.read_csv(source)
            return _standardize_shadow_rate(df)

    # Return empty DataFrame with expected schema if no data available
    logger.warning(
        "Could not fetch crypto shadow rate data automatically. "
        "Please download manually from IMF working paper supplementary materials."
    )
    return pd.DataFrame(
        columns=["country_code", "country_name", "year", "month", "btc_premium"]
    )


def _load_crypto_excel(
    filepath: Path,
    start_year: int,
    end_year: int,
) -> Optional[pd.DataFrame]:
    """
    Load and process the IMF Crypto Parallel Exchange Rates Excel file.

    Args:
        filepath: Path to the Excel file.
        start_year: Start year to filter.
        end_year: End year to filter.

    Returns:
        Processed DataFrame with country_code, year, btc_premium.
    """
    try:
        # Read the Data sheet
        df = pd.read_excel(filepath, sheet_name="Data")

        # Rename columns for easier handling
        df = df.rename(columns={
            "Date": "date",
            "Currency ISO": "currency_iso",
            "Crypto-based Parallel Exchange Rate Premium (in %)": "btc_premium",
            "Trade volume in local market (in USD)": "trade_volume_usd",
        })

        # Convert date and extract year/month
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        # Filter by year range
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

        # Map currency to country code
        df["country_code"] = df["currency_iso"].map(CURRENCY_TO_COUNTRY)

        # Drop rows without valid country mapping or premium data
        df = df[df["country_code"].notna() & df["btc_premium"].notna()]

        # Aggregate to yearly averages (weighted by trade volume if available)
        if "trade_volume_usd" in df.columns and df["trade_volume_usd"].notna().any():
            # Weighted average by trade volume
            def weighted_avg(group):
                weights = group["trade_volume_usd"].fillna(1)
                if weights.sum() > 0:
                    return (group["btc_premium"] * weights).sum() / weights.sum()
                return group["btc_premium"].mean()

            yearly = df.groupby(["country_code", "year"]).apply(
                weighted_avg, include_groups=False
            ).reset_index(name="btc_premium")
        else:
            # Simple average
            yearly = df.groupby(["country_code", "year"])["btc_premium"].mean().reset_index()

        logger.info(
            f"Loaded crypto premium data: {len(yearly)} country-year observations, "
            f"{yearly['country_code'].nunique()} countries"
        )

        return yearly

    except Exception as e:
        logger.error(f"Error loading crypto Excel file: {e}")
        return None


def _fetch_ifs_exchange_rates(start_year: int, end_year: int) -> Optional[pd.DataFrame]:
    """
    Fetch exchange rate data from IMF IFS as a proxy/supplement.

    This provides official vs parallel exchange rate gaps where available.
    """
    config = get_config()

    # IFS indicators for exchange rates
    # ENDA_XDC_USD_RATE - Official exchange rate
    # EDNA_XDC_USD_RATE - End of period rate

    ifs_url = f"{config.api.imf_base_url}/data/IFS/A..ENDA_XDC_USD_RATE"

    try:
        response = requests.get(
            ifs_url,
            params={
                "startPeriod": str(start_year),
                "endPeriod": str(end_year),
            },
            headers={"Accept": "application/vnd.sdmx.data+json;version=1.0"},
            timeout=config.api.timeout,
        )

        if response.status_code == 200:
            data = response.json()
            return _parse_sdmx_json(data)

    except requests.RequestException as e:
        logger.warning(f"Could not fetch IFS exchange rate data: {e}")

    return None


def _parse_sdmx_json(data: dict) -> pd.DataFrame:
    """Parse SDMX JSON response into a DataFrame."""
    records = []

    try:
        dataset = data.get("data", {}).get("dataSets", [{}])[0]
        series = dataset.get("series", {})
        structure = data.get("data", {}).get("structure", {})

        # Get dimension information
        dimensions = structure.get("dimensions", {}).get("series", [])
        time_dims = structure.get("dimensions", {}).get("observation", [])

        for series_key, series_data in series.items():
            # Parse the series key to get country/indicator info
            key_parts = series_key.split(":")

            observations = series_data.get("observations", {})
            for time_idx, obs_data in observations.items():
                if obs_data and len(obs_data) > 0:
                    value = obs_data[0]
                    records.append(
                        {
                            "series_key": series_key,
                            "time_index": int(time_idx),
                            "value": value,
                        }
                    )

    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Error parsing SDMX JSON: {e}")

    return pd.DataFrame(records)


def _standardize_shadow_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for the shadow rate dataset."""
    # Map various possible column names to standard names
    column_map = {
        "iso3": "country_code",
        "iso": "country_code",
        "country": "country_name",
        "date": "year",
        "premium": "btc_premium",
        "shadow_rate": "btc_premium",
        "parallel_rate": "btc_premium",
    }

    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    return df


def fetch_cpi_data(
    output_path: Optional[Path] = None,
    start_year: int = 2010,
    end_year: int = 2023,
) -> pd.DataFrame:
    """
    Fetch CPI (Consumer Price Index) data from IMF IFS.

    Used to calculate inflation rates as a control variable.

    Args:
        output_path: Path to save the CSV output.
        start_year: Start year for the data range.
        end_year: End year for the data range.

    Returns:
        DataFrame with columns: country_code, year, cpi_index, inflation_rate
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "imf_cpi.csv"

    # IFS indicator: PCPI_IX - Consumer Price Index
    ifs_url = f"{config.api.imf_base_url}/data/IFS/A..PCPI_IX"

    try:
        logger.info("Fetching CPI data from IMF IFS...")
        response = requests.get(
            ifs_url,
            params={
                "startPeriod": str(start_year),
                "endPeriod": str(end_year),
            },
            headers={"Accept": "application/vnd.sdmx.data+json;version=1.0"},
            timeout=config.api.timeout,
        )
        response.raise_for_status()

        data = response.json()
        df = _parse_cpi_response(data, start_year)

        if not df.empty:
            # Calculate year-over-year inflation rate
            df = df.sort_values(["country_code", "year"])
            df["inflation_rate"] = df.groupby("country_code")["cpi_index"].pct_change() * 100

            df.to_csv(output_path, index=False)
            logger.info(f"Saved CPI data to {output_path}")
            return df

    except requests.RequestException as e:
        logger.error(f"Failed to fetch CPI data: {e}")

    return pd.DataFrame(columns=["country_code", "year", "cpi_index", "inflation_rate"])


def _parse_cpi_response(data: dict, start_year: int) -> pd.DataFrame:
    """Parse IMF CPI response into a standardized DataFrame."""
    records = []

    try:
        dataset = data.get("data", {}).get("dataSets", [{}])[0]
        series = dataset.get("series", {})
        structure = data.get("data", {}).get("structure", {})

        # Get country codes from dimensions
        dimensions = structure.get("dimensions", {}).get("series", [])
        country_dim = next((d for d in dimensions if d.get("id") == "REF_AREA"), None)
        countries = {
            str(i): v.get("id")
            for i, v in enumerate(country_dim.get("values", []))
        } if country_dim else {}

        # Get time periods
        time_dims = structure.get("dimensions", {}).get("observation", [])
        time_dim = time_dims[0] if time_dims else {}
        time_periods = {
            str(i): v.get("id")
            for i, v in enumerate(time_dim.get("values", []))
        }

        for series_key, series_data in series.items():
            key_parts = series_key.split(":")
            country_idx = key_parts[0] if key_parts else "0"
            country_code = countries.get(country_idx, "UNK")

            observations = series_data.get("observations", {})
            for time_idx, obs_data in observations.items():
                if obs_data and len(obs_data) > 0:
                    value = obs_data[0]
                    year_str = time_periods.get(time_idx, "")
                    try:
                        year = int(year_str)
                        records.append(
                            {
                                "country_code": country_code,
                                "year": year,
                                "cpi_index": value,
                            }
                        )
                    except ValueError:
                        continue

    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Error parsing CPI response: {e}")

    return pd.DataFrame(records)
