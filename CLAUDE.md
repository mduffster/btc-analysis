# BTC Criminal Substitution Analysis

## Core Thesis

Criminal utility is why BTC didn't die on the vine. It provided baseline demand that sustained the network until institutional adoption (MicroStrategy, Aug 2020). 

## What We Clearly Observe

1. **Cash seizure ratios are declining** - DOJ cash seizures peaked 2009-2013 ($67M/month avg), now below pre-BTC levels ($33M/month). This is almost certainly crypto substitution for fiat.

2. **Pre-MicroStrategy signal exists** - Criminal substitution effect is detectable before institutional noise:
   - DOJ pre-MicroStrategy: -0.356 (p=0.0001)***
   - Combined LE pre-MicroStrategy: -0.297 (p=0.027)**

3. **Post-MicroStrategy signal disappears** - Not because criminal demand went away, but because retail/institutional flows drown it out. This is expected.

## Data Assets

### Law Enforcement Seizures

| Dataset | Source | Period | Location |
|---------|--------|--------|----------|
| DOJ cash seizures (monthly) | CATS FOIA | 1996-2025 | `data/raw/doj_cash_seizures_monthly.csv` |
| DOJ full CATS database | FOIA | 1996-2025 | `data/raw/foiaCATS.zip` (664MB, 1.35M records) |
| CBP currency seizures (raw) | CBP.gov | FY19-FY26 | `data/raw/cbp_currency/*.csv` (6 files) |
| CBP drug seizures (raw) | CBP.gov | FY19-FY26 | `data/raw/cbp/*.csv` (6 files) |
| CBP currency (processed) | CBP.gov | 2019-2025 | `data/raw/cbp_currency_seizures.csv` |
| CBP drugs (processed) | CBP.gov | 2019-2025 | `data/raw/cbp_drug_seizures.csv` |
| Combined LE seizures | DOJ + CBP | 2019-2025 | `data/processed/combined_criminal_seizures.csv` |

### Crypto & Market Data

| Dataset | Source | Period | Location |
|---------|--------|--------|----------|
| BTC prices (full history) | CryptoCompare | 2010-2025 | `data/raw/btc_prices_cryptocompare.csv` |
| BTC prices (yfinance) | Yahoo Finance | 2014-2025 | `data/raw/btc_price.csv` |
| USDT supply history | DefiLlama | 2017-2025 | `data/raw/stablecoins/usdt_defillama.json` |
| USDC supply history | DefiLlama | 2018-2025 | `data/raw/stablecoins/usdc_defillama.json` |
| Tron stablecoin supply | DefiLlama | 2019-2025 | `data/raw/stablecoins/tron_stablecoins.json` |
| Tornado Cash TVL | DefiLlama | 2019-2025 | `data/raw/stablecoins/tornado_cash_tvl.json` |
| Stablecoin (processed) | DefiLlama | 2017-2025 | `data/processed/stablecoin_monthly.csv` |
| Privacy tools (processed) | DefiLlama | 2019-2025 | `data/processed/privacy_tools_monthly.csv` |

### Cross-Country / Regulatory (Phase 1 - Regulatory Arbitrage)

| Dataset | Source | Period | Location |
|---------|--------|--------|----------|
| IMF Crypto Shadow Rate | IMF | varies | `data/raw/imf_crypto_shadow_rate.csv` |
| Chinn-Ito Financial Openness | Academic | 1970-2022 | `data/raw/chinn_ito.csv` |
| World Bank Governance (WGI) | World Bank | varies | `data/raw/world_bank_wgi.csv` |
| World Bank Development | World Bank | varies | `data/raw/world_bank_dev.csv` |
| FATF Status | FATF | current | `data/raw/fatf_status.csv` |
| Organized Crime Index | GI-TOC | varies | `data/raw/organized_crime_index.csv` |

### Health / Drug Market Proxy

| Dataset | Source | Period | Location |
|---------|--------|--------|----------|
| CDC Overdose Deaths | CDC | varies | `data/raw/cdc_overdose_deaths.csv` |

### Processed Panels

| Dataset | Description | Location |
|---------|-------------|----------|
| cash_substitution_panel.csv | CBP seizure ratios + BTC | `data/processed/` |
| cash_substitution_normalized.csv | Z-scored substitution metrics | `data/processed/` |
| merged_panel.csv | Cross-country merged data | `data/processed/` |
| timeseries_panel.csv | Time series analysis panel | `data/processed/` |

## Key Analysis Files

- `src/btc_analysis/analysis/combined_substitution.py` - Full analytical suite (regressions, lags, structural breaks, controls)
- `src/btc_analysis/analysis/time_trend_model.py` - Secular trend analysis with 1996 baseline
- `outputs/combined_substitution/full_analysis.txt` - Latest results

## Structural Breaks

| Event | Date | Effect on Seizures |
|-------|------|-------------------|
| BTC Launch | 2009-01 | +246% (p=0.03) |
| MicroStrategy | 2020-08 | -80% (p=0.008) |
| ETF Approval | 2024-01 | -96% (p=0.002) |

## Interpretation Cautions

1. **Seizure timing** - Law enforcement data has lag between criminal activity and recorded seizure
2. **Secular trend vs cyclical** - Cash-to-crypto shift is monotonic, not month-to-month oscillation
3. **BTC as rails, not storage** - Criminals may use BTC to transfer then convert to stablecoins
4. **Post-2020 noise** - Don't over-interpret contemporaneous effects in institutional era

## Next Steps (if continuing)

- Focus analysis on pre-MicroStrategy period where signal is cleanest
- Model the secular substitution trend properly (not just month-to-month variation)
- Consider BTC as on-ramp/off-ramp rather than holding asset
