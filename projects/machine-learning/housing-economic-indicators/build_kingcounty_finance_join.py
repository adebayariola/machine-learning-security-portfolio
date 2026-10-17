from __future__ import annotations

from pathlib import Path

import pandas as pd
import rdata


ROOT = Path("/path/to/local/Documents/Codex/2026-10-06/us")
DATA_DIR = ROOT / "work" / "data"
OUT_DIR = ROOT / "outputs"

SALES_RDA = DATA_DIR / "kingco_sales.rda"
OUTPUT_CSV = OUT_DIR / "kingco_sales_last10yrs_with_financial_indicators.csv"
SUMMARY_TXT = OUT_DIR / "kingco_sales_financial_join_summary.txt"


def load_kingco_sales() -> pd.DataFrame:
    parsed = rdata.parser.parse_file(SALES_RDA)
    converted = rdata.conversion.convert(parsed)
    sales = converted["kingco_sales"].copy()

    # R Date values are days since 1970-01-01.
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], unit="D", origin="1970-01-01")
    return sales


def read_fred_csv(filename: str, series_id: str, output_name: str) -> pd.DataFrame:
    fred = pd.read_csv(DATA_DIR / filename)
    fred["DATE"] = pd.to_datetime(fred["observation_date"], errors="coerce")
    fred[series_id] = pd.to_numeric(fred[series_id], errors="coerce")
    fred = fred.dropna(subset=["DATE", series_id]).copy()
    return fred.rename(columns={series_id: output_name})


def monthly_average(filename: str, series_id: str, output_name: str) -> pd.DataFrame:
    fred = read_fred_csv(filename, series_id, output_name)
    fred["sale_month_start"] = fred["DATE"].dt.to_period("M").dt.to_timestamp()
    return fred.groupby("sale_month_start", as_index=False)[output_name].mean()


def monthly_last(filename: str, series_id: str, output_name: str) -> pd.DataFrame:
    fred = read_fred_csv(filename, series_id, output_name)
    fred["sale_month_start"] = fred["DATE"].dt.to_period("M").dt.to_timestamp()
    return fred[["sale_month_start", output_name]].drop_duplicates("sale_month_start", keep="last")


def annual_last(filename: str, series_id: str, output_name: str) -> pd.DataFrame:
    fred = read_fred_csv(filename, series_id, output_name)
    fred["sale_year"] = fred["DATE"].dt.year
    return fred[["sale_year", output_name]].drop_duplicates("sale_year", keep="last")


def asof_by_month(df: pd.DataFrame, external: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Attach the most recent external value available by the sale month."""
    left = df.sort_values("sale_month_start").copy()
    right = external.sort_values("sale_month_start").copy()
    merged = pd.merge_asof(left, right, on="sale_month_start", direction="backward")
    return merged.sort_index()


def asof_by_date(df: pd.DataFrame, external: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Attach the most recent external value available by the exact sale date."""
    left = df.sort_values("sale_date").copy()
    right = external[["DATE", value_col]].sort_values("DATE").copy()
    merged = pd.merge_asof(left, right, left_on="sale_date", right_on="DATE", direction="backward")
    return merged.drop(columns=["DATE"]).sort_index()


def main() -> None:
    df = load_kingco_sales()
    raw_rows = len(df)

    df = df.drop_duplicates(subset=["sale_id", "pinx", "sale_date", "sale_price"]).copy()
    deduped_rows = len(df)

    cutoff_date = df["sale_date"].max() - pd.DateOffset(years=10)
    df = df.query("sale_date >= @cutoff_date").copy()

    df["sale_year"] = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month
    df["sale_month_start"] = df["sale_date"].dt.to_period("M").dt.to_timestamp()
    df["is_seattle"] = (
        df["city"].astype("string").str.strip().str.upper().eq("SEATTLE").astype("int8")
    )

    mortgage = monthly_average("MORTGAGE30US.csv", "MORTGAGE30US", "mortgage_30yr_rate")
    unemployment = monthly_last("SEAT653URN.csv", "SEAT653URN", "unemployment_rate_seattle_msa")
    cpi = read_fred_csv("CUUSA423SA0.csv", "CUUSA423SA0", "cpi_seattle_all_items")
    income = read_fred_csv("PCPI53033.csv", "PCPI53033", "king_county_per_capita_income")

    # Mortgage and unemployment are effectively monthly for modeling.
    # CPI is semiannual in this FRED series and income is annual, so use the
    # latest available value at the sale date rather than exact date matching.
    df = asof_by_month(df, mortgage, "mortgage_30yr_rate")
    df = asof_by_month(df, unemployment, "unemployment_rate_seattle_msa")
    df = asof_by_date(df, cpi, "cpi_seattle_all_items")
    df = asof_by_date(df, income, "king_county_per_capita_income")

    indicator_cols = [
        "mortgage_30yr_rate",
        "cpi_seattle_all_items",
        "unemployment_rate_seattle_msa",
        "king_county_per_capita_income",
    ]

    df = df.drop(columns=["sale_month_start"])
    df.to_csv(OUTPUT_CSV, index=False)

    missing = (df[indicator_cols].isna().mean() * 100).round(2)
    summary = [
        "King County housing sales plus financial/economic indicators",
        f"Source housing rows: {raw_rows:,}",
        f"Rows after duplicate removal: {deduped_rows:,}",
        f"Last-10-years cutoff: {cutoff_date.date()}",
        f"Final rows: {len(df):,}",
        f"Final columns: {len(df.columns):,}",
        f"Sale-date range: {df['sale_date'].min().date()} to {df['sale_date'].max().date()}",
        "",
        "Added indicators:",
        *[f"- {col}: {missing[col]}% missing" for col in indicator_cols],
        "",
        "Join keys:",
        "- Mortgage, CPI, unemployment: sale month",
        "- King County income: sale year",
    ]
    SUMMARY_TXT.write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved summary: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
