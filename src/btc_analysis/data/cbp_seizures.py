"""CBP Drug Seizure Data fetcher.

Fetches drug seizure statistics from U.S. Customs and Border Protection.

Source: https://www.cbp.gov/newsroom/stats/drug-seizure-statistics
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from btc_analysis.config import get_config

logger = logging.getLogger(__name__)

# CBP stats page URL
CBP_STATS_URL = "https://www.cbp.gov/newsroom/stats/drug-seizure-statistics"


def fetch_drug_seizures(
    output_path: Optional[Path] = None,
    start_year: int = 2017,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Fetch drug seizure data from CBP.

    CBP publishes monthly drug seizure statistics by drug type.
    Data includes seizures at ports of entry and between ports.

    Note: CBP data may require web scraping or manual download as they
    don't provide a formal API.

    Args:
        output_path: Path to save the CSV output. If None, uses default from config.
        start_year: Start year for the data range.
        end_year: End year for the data range.

    Returns:
        DataFrame with monthly drug seizure data.
    """
    config = get_config()

    if output_path is None:
        output_path = config.paths.raw_dir / "cbp_drug_seizures.csv"

    # Check for CBP subdirectory with CSV files
    cbp_dir = config.paths.raw_dir / "cbp"
    if cbp_dir.exists():
        cbp_files = list(cbp_dir.glob("*.csv"))
        if cbp_files:
            logger.info(f"Found {len(cbp_files)} CBP CSV files in {cbp_dir}")
            df = _load_cbp_csv_files(cbp_files)
            if df is not None and not df.empty:
                df = _filter_by_year(df, start_year, end_year)
                df.to_csv(output_path, index=False)
                logger.info(f"Saved processed CBP data to {output_path}")
                return df

    # Check for local Excel/CSV files (manual download)
    local_files = list(config.paths.raw_dir.glob("*CBP*seizure*"))
    local_files += list(config.paths.raw_dir.glob("*drug*seizure*"))

    for local_file in local_files:
        logger.info(f"Found local seizure data file: {local_file}")
        df = _load_local_seizure_file(local_file)
        if df is not None and not df.empty:
            df = _filter_by_year(df, start_year, end_year)
            df.to_csv(output_path, index=False)
            return df

    # Try to scrape CBP website for downloadable data links
    try:
        logger.info(f"Attempting to fetch CBP seizure data from {CBP_STATS_URL}")
        df = _scrape_cbp_data()
        if df is not None and not df.empty:
            df = _filter_by_year(df, start_year, end_year)
            df.to_csv(output_path, index=False)
            return df
    except Exception as e:
        logger.warning(f"Could not scrape CBP data: {e}")

    # Check for existing output file
    if output_path.exists():
        logger.info(f"Loading cached seizure data from {output_path}")
        return pd.read_csv(output_path, parse_dates=["date"] if "date" in pd.read_csv(output_path, nrows=1).columns else None)

    # If no data available, create synthetic data structure with instructions
    logger.warning(
        "Could not automatically fetch CBP seizure data. "
        "Please manually download from: https://www.cbp.gov/newsroom/stats/drug-seizure-statistics "
        "and save to data/raw/ directory."
    )

    return _create_empty_df()


def _load_cbp_csv_files(files: list) -> Optional[pd.DataFrame]:
    """
    Load and combine multiple CBP CSV files.

    Args:
        files: List of Path objects to CBP CSV files.

    Returns:
        Combined and processed DataFrame.
    """
    all_data = []

    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath.name}")
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Error loading {filepath}: {e}")

    if not all_data:
        return None

    # Combine all files
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined {len(combined)} total records from {len(all_data)} files")

    # Process the combined data
    return _process_cbp_data(combined)


def _process_cbp_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw CBP data into monthly national totals by drug type.

    Args:
        df: Raw DataFrame from CBP CSV files.

    Returns:
        Processed DataFrame with monthly national totals.
    """
    logger.info(f"Processing CBP data with columns: {df.columns.tolist()}")

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Expected columns: fy, month_(abbv), drug_type, sum_qty_(lbs)
    fy_col = "fy" if "fy" in df.columns else None
    month_col = next((c for c in df.columns if "month" in c), None)
    drug_col = next((c for c in df.columns if "drug" in c), None)
    qty_col = next((c for c in df.columns if "qty" in c or "lbs" in c), None)

    if not all([fy_col, month_col, drug_col, qty_col]):
        logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
        return _create_empty_df()

    # Convert quantity to numeric
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")

    # Map month abbreviations to numbers
    month_abbr_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    df["month_num"] = df[month_col].str.upper().map(month_abbr_map)

    # Clean FY column - extract just the year number
    df[fy_col] = df[fy_col].astype(str).str.extract(r"(\d{4})")[0]
    df[fy_col] = pd.to_numeric(df[fy_col], errors="coerce")

    # Convert fiscal year + month to calendar year + month
    # Federal FY starts October 1: FY2024 Oct = Calendar Oct 2023
    def fy_to_calendar(row):
        fy = row[fy_col]
        month = row["month_num"]
        if pd.isna(fy) or pd.isna(month):
            return None, None
        fy = int(fy)
        month = int(month)
        # Oct-Dec of FY are in prior calendar year
        if month >= 10:
            return fy - 1, month
        else:
            return fy, month

    cal_dates = df.apply(fy_to_calendar, axis=1, result_type="expand")
    df["year"] = cal_dates[0]
    df["month"] = cal_dates[1]

    # Drop rows with invalid dates
    df = df.dropna(subset=["year", "month"])
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    # Standardize drug type names
    drug_map = {
        "cocaine": "cocaine_lbs",
        "fentanyl": "fentanyl_lbs",
        "heroin": "heroin_lbs",
        "methamphetamine": "meth_lbs",
        "marijuana": "marijuana_lbs",
        "other drugs**": "other_drugs_lbs",
        "other drugs": "other_drugs_lbs",
    }

    def map_drug_type(drug):
        drug_lower = str(drug).lower().strip()
        for pattern, mapped in drug_map.items():
            if pattern in drug_lower:
                return mapped
        return f"{drug_lower.replace(' ', '_')}_lbs"

    df["drug_category"] = df[drug_col].apply(map_drug_type)

    # Aggregate to national monthly totals by drug type
    monthly = df.groupby(["year", "month", "drug_category"])[qty_col].sum().reset_index()

    # Pivot drug types to columns
    pivot = monthly.pivot_table(
        index=["year", "month"],
        columns="drug_category",
        values=qty_col,
        aggfunc="sum",
    ).reset_index()

    # Remove column name from pivot
    pivot.columns.name = None

    # Create total column
    drug_cols = [c for c in pivot.columns if c.endswith("_lbs")]
    if drug_cols:
        pivot["total_seizures_lbs"] = pivot[drug_cols].sum(axis=1)

    # Create date column
    pivot["date"] = pd.to_datetime(
        pivot["year"].astype(str) + "-" +
        pivot["month"].astype(str).str.zfill(2) + "-01"
    )

    # Sort by date
    pivot = pivot.sort_values("date").reset_index(drop=True)

    # Remove duplicates (from overlapping files)
    pivot = pivot.drop_duplicates(subset=["year", "month"], keep="last")

    logger.info(f"Processed CBP data: {len(pivot)} months, drugs: {drug_cols}")

    return pivot


def _scrape_cbp_data() -> Optional[pd.DataFrame]:
    """
    Attempt to scrape CBP drug seizure statistics page.

    Returns:
        DataFrame with seizure data if successful, None otherwise.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Research purposes - academic analysis)"
    }

    response = requests.get(CBP_STATS_URL, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Look for data tables on the page
    tables = soup.find_all("table")

    for table in tables:
        df = _parse_html_table(table)
        if df is not None and "seizure" in str(df.columns).lower():
            return df

    # Look for downloadable Excel/CSV links
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if any(ext in href.lower() for ext in [".xlsx", ".xls", ".csv"]):
            if "drug" in href.lower() or "seizure" in href.lower():
                logger.info(f"Found potential data download: {href}")
                # Could implement download here

    return None


def _parse_html_table(table) -> Optional[pd.DataFrame]:
    """Parse an HTML table element into a DataFrame."""
    try:
        # Extract headers
        headers = []
        header_row = table.find("thead")
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all("th")]

        # Extract rows
        rows = []
        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells and len(cells) > 1:
                rows.append(cells)

        if rows:
            if headers:
                df = pd.DataFrame(rows, columns=headers[:len(rows[0])])
            else:
                df = pd.DataFrame(rows)
            return df

    except Exception as e:
        logger.warning(f"Error parsing HTML table: {e}")

    return None


def _load_local_seizure_file(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load seizure data from a local file.

    Args:
        filepath: Path to the data file (Excel or CSV).

    Returns:
        Processed DataFrame or None if loading fails.
    """
    try:
        if filepath.suffix in [".xlsx", ".xls"]:
            # Try to load Excel file
            xlsx = pd.ExcelFile(filepath)
            logger.info(f"Excel sheets available: {xlsx.sheet_names}")

            # Look for monthly data sheet
            for sheet_name in xlsx.sheet_names:
                if "month" in sheet_name.lower() or "data" in sheet_name.lower():
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    return _standardize_seizure_columns(df)

            # Default to first sheet
            df = pd.read_excel(filepath)
            return _standardize_seizure_columns(df)

        elif filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
            return _standardize_seizure_columns(df)

    except Exception as e:
        logger.error(f"Error loading seizure file {filepath}: {e}")

    return None


def _standardize_seizure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names in seizure data.

    Args:
        df: Raw DataFrame with potentially varied column names.

    Returns:
        DataFrame with standardized column names.
    """
    # Common column name mappings
    column_map = {
        "fiscal year": "fiscal_year",
        "fy": "fiscal_year",
        "month": "month",
        "cocaine": "cocaine_lbs",
        "marijuana": "marijuana_lbs",
        "heroin": "heroin_lbs",
        "methamphetamine": "meth_lbs",
        "fentanyl": "fentanyl_lbs",
        "total": "total_seizures_lbs",
    }

    # Lowercase all columns for matching
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Apply mappings
    rename_dict = {}
    for orig_col in df.columns:
        for pattern, new_name in column_map.items():
            if pattern in orig_col:
                rename_dict[orig_col] = new_name
                break

    df = df.rename(columns=rename_dict)

    # Convert fiscal year + month to calendar date if needed
    if "fiscal_year" in df.columns and "month" in df.columns:
        df = _fiscal_to_calendar(df)

    return df


def _fiscal_to_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert fiscal year + month to calendar year/month/date.

    Federal fiscal year starts October 1.
    FY2024 Month 1 = October 2023.

    Args:
        df: DataFrame with fiscal_year and month columns.

    Returns:
        DataFrame with year, month, date columns added.
    """
    # Fiscal year month mapping
    # FY month 1 = October (calendar month 10)
    # FY month 12 = September (calendar month 9)
    fy_month_to_cal = {
        1: (10, -1),   # Oct of prior year
        2: (11, -1),   # Nov of prior year
        3: (12, -1),   # Dec of prior year
        4: (1, 0),     # Jan of FY year
        5: (2, 0),     # Feb
        6: (3, 0),     # Mar
        7: (4, 0),     # Apr
        8: (5, 0),     # May
        9: (6, 0),     # Jun
        10: (7, 0),    # Jul
        11: (8, 0),    # Aug
        12: (9, 0),    # Sep
    }

    def convert_row(row):
        fy = int(row["fiscal_year"])
        fy_month = int(row["month"])

        if fy_month in fy_month_to_cal:
            cal_month, year_adj = fy_month_to_cal[fy_month]
            cal_year = fy + year_adj
        else:
            # Fallback
            cal_month = fy_month
            cal_year = fy

        return pd.Series({
            "year": cal_year,
            "month": cal_month,
        })

    cal_dates = df.apply(convert_row, axis=1)
    df["year"] = cal_dates["year"]
    df["month"] = cal_dates["month"]

    # Create date column
    df["date"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" +
        df["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )

    return df


def _filter_by_year(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Filter DataFrame by year range."""
    if "year" in df.columns:
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    return df


def _create_empty_df() -> pd.DataFrame:
    """Create empty DataFrame with expected schema."""
    return pd.DataFrame(columns=[
        "date", "year", "month",
        "cocaine_lbs", "marijuana_lbs", "heroin_lbs",
        "meth_lbs", "fentanyl_lbs", "total_seizures_lbs",
    ])


def create_manual_cbp_data() -> pd.DataFrame:
    """
    Create DataFrame from manually compiled CBP data.

    This is used when automatic fetching fails.
    Data compiled from CBP public reports:
    https://www.cbp.gov/newsroom/stats/drug-seizure-statistics

    Returns:
        DataFrame with monthly seizure data.
    """
    # CBP publishes annual and monthly summaries
    # Key drugs tracked: Cocaine, Fentanyl, Heroin, Methamphetamine, Marijuana

    # Example structure - would need manual data entry from CBP reports
    data = [
        # FY2017-FY2024 monthly data would go here
        # Format: date, cocaine_lbs, fentanyl_lbs, heroin_lbs, meth_lbs, marijuana_lbs
    ]

    if data:
        df = pd.DataFrame(data, columns=[
            "date", "cocaine_lbs", "fentanyl_lbs",
            "heroin_lbs", "meth_lbs", "marijuana_lbs"
        ])
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        return df

    return _create_empty_df()
