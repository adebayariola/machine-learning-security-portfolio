"""
Join King County housing sales with external economic indicators from FRED.

Usage:
    python join_external_economic_data.py \
        --housing-csv /path/to/kc_housing_cleaned.csv \
        --output-csv /path/to/kc_housing_with_economic_indicators.csv

Expected housing column:
    sale_date

Adds:
    mortgage_30yr_rate
    cpi_seattle_all_items
    unemployment_rate_seattle_msa
    king_county_per_capita_income
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


@dataclass(frozen=True)
class FredSeries:
    series_id: str
    output_name: str
    frequency: str


SERIES = [
    FredSeries("MORTGAGE30US", "mortgage_30yr_rate", "weekly_to_monthly"),
    FredSeries("CUUSA423SA0", "cpi_seattle_all_items", "monthly"),
    FredSeries("SEAT653URN", "unemployment_rate_seattle_msa", "monthly"),
    FredSeries("PCPI53033", "king_county_per_capita_income", "annual"),
]


def read_fred_series(series: FredSeries) -> pd.DataFrame:
    url = FRED_CSV.format(series_id=series.series_id)
    df = pd.read_csv(url)
    df["DATE"] = pd.to_datetime(df["observation_date"])
    df[series.series_id] = pd.to_numeric(df[series.series_id], errors="coerce")
    df = df.dropna(subset=[series.series_id]).copy()

    if series.frequency == "weekly_to_monthly":
        df["sale_month"] = df["DATE"].dt.to_period("M").dt.to_timestamp()
        return (
            df.groupby("sale_month", as_index=False)[series.series_id]
            .mean()
            .rename(columns={series.series_id: series.output_name})
        )

    if series.frequency == "monthly":
        df["sale_month"] = df["DATE"].dt.to_period("M").dt.to_timestamp()
        monthly = df[["sale_month", series.series_id]].copy()
        monthly = monthly.rename(columns={series.series_id: series.output_name})
        return monthly.drop_duplicates(subset=["sale_month"], keep="last")

    if series.frequency == "annual":
        df["sale_year"] = df["DATE"].dt.year
        annual = df[["sale_year", series.series_id]].copy()
        annual = annual.rename(columns={series.series_id: series.output_name})
        return annual.drop_duplicates(subset=["sale_year"], keep="last")

    raise ValueError(f"Unsupported frequency: {series.frequency}")


def add_time_keys(housing: pd.DataFrame) -> pd.DataFrame:
    if "sale_date" not in housing.columns:
        raise KeyError("Housing CSV must contain a 'sale_date' column.")

    housing = housing.copy()
    housing["sale_date"] = pd.to_datetime(housing["sale_date"], errors="coerce")
    housing = housing.dropna(subset=["sale_date"]).copy()
    housing["sale_month"] = housing["sale_date"].dt.to_period("M").dt.to_timestamp()
    housing["sale_year"] = housing["sale_date"].dt.year
    return housing


def merge_external_indicators(housing_csv: str, output_csv: str) -> None:
    housing = pd.read_csv(housing_csv, low_memory=False)
    merged = add_time_keys(housing)

    for series in SERIES:
        external = read_fred_series(series)
        key = "sale_year" if series.frequency == "annual" else "sale_month"
        merged = merged.merge(external, on=key, how="left")

    indicator_cols = [series.output_name for series in SERIES]
    missing = merged[indicator_cols].isna().mean().sort_values(ascending=False)
    print("Missing share after merge:")
    print((missing * 100).round(2).astype(str) + "%")

    merged.to_csv(output_csv, index=False)
    print(f"Saved merged file: {output_csv}")
    print(f"Rows: {len(merged):,}")
    print(f"Columns: {len(merged.columns):,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--housing-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    merge_external_indicators(args.housing_csv, args.output_csv)


if __name__ == "__main__":
    main()
