"""Data fetching modules for BTC analysis."""

from .imf import fetch_crypto_shadow_rate, fetch_cpi_data
from .capital_controls import fetch_chinn_ito, fetch_fkrsu
from .world_bank import fetch_governance_indicators
from .crime import fetch_organized_crime_index
from .aml import create_fatf_panel, get_fatf_status
from .cdc_overdose import fetch_overdose_deaths, get_national_monthly_overdose
from .cbp_seizures import fetch_drug_seizures
from .cbp_currency import fetch_currency_seizures
from .btc_price import (
    fetch_btc_price,
    get_monthly_btc_price,
    get_btc_halving_dates,
    add_halving_controls,
)

__all__ = [
    # Cross-sectional data
    "fetch_crypto_shadow_rate",
    "fetch_cpi_data",
    "fetch_chinn_ito",
    "fetch_fkrsu",
    "fetch_governance_indicators",
    "fetch_organized_crime_index",
    "create_fatf_panel",
    "get_fatf_status",
    # Time series data
    "fetch_overdose_deaths",
    "get_national_monthly_overdose",
    "fetch_drug_seizures",
    "fetch_currency_seizures",
    "fetch_btc_price",
    "get_monthly_btc_price",
    "get_btc_halving_dates",
    "add_halving_controls",
]
