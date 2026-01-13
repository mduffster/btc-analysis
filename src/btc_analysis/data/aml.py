"""AML risk and sanctions data."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from btc_analysis.config import get_config

logger = logging.getLogger(__name__)

# FATF Grey List - Jurisdictions under Increased Monitoring
# Source: https://www.fatf-gafi.org/en/countries/black-and-grey-lists.html
# Updated: October 2024

FATF_BLACK_LIST = {
    # High-Risk Jurisdictions Subject to Call for Action
    "PRK",  # North Korea (Democratic People's Republic of Korea)
    "IRN",  # Iran
    "MMR",  # Myanmar
}

# Grey list with approximate dates added (for time-varying indicator)
FATF_GREY_LIST_HISTORY = {
    # Country: (ISO3, year_added, year_removed or None)
    "ALB": (2020, 2023),  # Albania - removed Feb 2023
    "ARE": (2022, 2024),  # UAE - removed Feb 2024
    "BGD": (2022, None),  # Barbados - on list (using BGD as placeholder)
    "BGR": (2023, None),  # Bulgaria - added Oct 2023
    "BFA": (2021, None),  # Burkina Faso
    "CMR": (2023, None),  # Cameroon - added Oct 2023
    "HRV": (2023, None),  # Croatia - added Jun 2023
    "COD": (2022, None),  # Democratic Republic of Congo
    "HTI": (2021, None),  # Haiti
    "JAM": (2020, 2024),  # Jamaica - removed Jun 2024
    "JOR": (2021, 2023),  # Jordan - removed Feb 2023
    "KEN": (2024, None),  # Kenya - added Feb 2024
    "LBN": (2024, None),  # Lebanon - added Oct 2024
    "MLI": (2021, None),  # Mali
    "MCO": (2023, None),  # Monaco - added Jun 2023
    "MOZ": (2023, None),  # Mozambique - added Oct 2023
    "NAM": (2024, None),  # Namibia - added Feb 2024
    "NGA": (2023, None),  # Nigeria - added Feb 2023
    "PAK": (2018, 2022),  # Pakistan - removed Oct 2022
    "PAN": (2019, 2023),  # Panama - removed Oct 2023
    "PHL": (2021, None),  # Philippines
    "SEN": (2021, None),  # Senegal
    "ZAF": (2023, None),  # South Africa - added Feb 2023
    "SSD": (2021, None),  # South Sudan
    "SYR": (2010, None),  # Syria
    "TZA": (2022, None),  # Tanzania
    "TUR": (2021, 2024),  # Turkey - removed Jun 2024
    "UGA": (2020, 2023),  # Uganda - removed Feb 2023
    "VEN": (2023, None),  # Venezuela - added Oct 2023
    "VNM": (2023, None),  # Vietnam - added Jun 2023
    "YEM": (2010, None),  # Yemen
    "ZWE": (2019, 2023),  # Zimbabwe - removed Feb 2023
}

# Current grey list as of late 2024
FATF_GREY_LIST_CURRENT = {
    "BGR",  # Bulgaria
    "BFA",  # Burkina Faso
    "CMR",  # Cameroon
    "HRV",  # Croatia
    "COD",  # Democratic Republic of Congo
    "HTI",  # Haiti
    "KEN",  # Kenya
    "LBN",  # Lebanon
    "MLI",  # Mali
    "MCO",  # Monaco
    "MOZ",  # Mozambique
    "NAM",  # Namibia
    "NGA",  # Nigeria
    "PHL",  # Philippines
    "SEN",  # Senegal
    "ZAF",  # South Africa
    "SSD",  # South Sudan
    "SYR",  # Syria
    "TZA",  # Tanzania
    "VEN",  # Venezuela
    "VNM",  # Vietnam
    "YEM",  # Yemen
}


def get_fatf_status(country_code: str, year: int) -> dict:
    """
    Get FATF list status for a country in a given year.

    Returns dict with:
        - fatf_black: 1 if on black list, 0 otherwise
        - fatf_grey: 1 if on grey list that year, 0 otherwise
        - fatf_any: 1 if on either list, 0 otherwise
    """
    # Black list (relatively stable)
    is_black = 1 if country_code in FATF_BLACK_LIST else 0

    # Grey list (time-varying)
    is_grey = 0
    if country_code in FATF_GREY_LIST_HISTORY:
        year_added, year_removed = FATF_GREY_LIST_HISTORY[country_code]
        if year >= year_added:
            if year_removed is None or year < year_removed:
                is_grey = 1

    return {
        "fatf_black": is_black,
        "fatf_grey": is_grey,
        "fatf_any": 1 if (is_black or is_grey) else 0,
    }


def create_fatf_panel(
    countries: list,
    start_year: int = 2019,
    end_year: int = 2023,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create a panel dataset with FATF list indicators.

    Args:
        countries: List of ISO3 country codes
        start_year: Start year
        end_year: End year
        output_path: Path to save CSV

    Returns:
        DataFrame with country_code, year, fatf_black, fatf_grey, fatf_any
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "fatf_status.csv"

    rows = []
    for country in countries:
        for year in range(start_year, end_year + 1):
            status = get_fatf_status(country, year)
            rows.append({
                "country_code": country,
                "year": year,
                **status,
            })

    df = pd.DataFrame(rows)

    # Summary
    grey_count = df.groupby("year")["fatf_grey"].sum()
    logger.info(f"FATF grey list countries by year:\n{grey_count.to_string()}")

    df.to_csv(output_path, index=False)
    logger.info(f"Saved FATF status data to {output_path}")

    return df


def fetch_sanctions_exposure(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Create sanctions exposure indicator.

    Countries under comprehensive sanctions have strong incentive for crypto adoption.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "sanctions_exposure.csv"

    # Countries under comprehensive or significant US/EU sanctions
    # Higher scores = more sanctions pressure
    SANCTIONS_SCORES = {
        # Comprehensive sanctions (highest pressure)
        "PRK": 10,  # North Korea
        "IRN": 10,  # Iran
        "SYR": 9,   # Syria
        "CUB": 8,   # Cuba
        "VEN": 8,   # Venezuela (sectoral)

        # Significant sanctions
        "RUS": 9,   # Russia (post-2022)
        "BLR": 7,   # Belarus (post-2022)
        "MMR": 6,   # Myanmar

        # Partial/targeted sanctions
        "ZWE": 4,   # Zimbabwe
        "SDN": 5,   # Sudan
        "SSD": 4,   # South Sudan
        "LBY": 4,   # Libya
        "YEM": 3,   # Yemen (partial)
        "NIC": 3,   # Nicaragua

        # Historical (for time-varying)
        "AFG": 5,   # Afghanistan (post-2021)
    }

    # Time-varying sanctions (major changes)
    SANCTIONS_CHANGES = {
        "RUS": {"pre_2022": 3, "post_2022": 9},  # Major increase after Ukraine invasion
        "BLR": {"pre_2022": 2, "post_2022": 7},
        "AFG": {"pre_2021": 2, "post_2021": 5},
    }

    rows = []
    for country, base_score in SANCTIONS_SCORES.items():
        for year in range(2019, 2024):
            score = base_score

            # Apply time-varying adjustments
            if country in SANCTIONS_CHANGES:
                changes = SANCTIONS_CHANGES[country]
                if "pre_2022" in changes and year < 2022:
                    score = changes["pre_2022"]
                elif "post_2022" in changes and year >= 2022:
                    score = changes["post_2022"]
                elif "pre_2021" in changes and year < 2021:
                    score = changes["pre_2021"]
                elif "post_2021" in changes and year >= 2021:
                    score = changes["post_2021"]

            rows.append({
                "country_code": country,
                "year": year,
                "sanctions_score": score,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved sanctions exposure data to {output_path}")

    return df
