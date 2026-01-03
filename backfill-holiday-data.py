"""
Backfill Swedish holiday data from Svenska Dagar API into Hopsworks feature store.

Svenska Dagar API: https://sholiday.faboul.se
- Free, no API key required
- Provides Swedish holidays, red days, work-free days, days before holidays
"""

import hopsworks
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

from util import fetch_swedish_holidays, clean_column_names

load_dotenv()


def create_holiday_feature_group(fs):
    """Create or get the holiday feature group in Hopsworks."""

    holiday_fg = fs.get_or_create_feature_group(
        name="swedish_holidays_fg",
        version=1,
        primary_key=["date"],
        event_time="date",
        online_enabled=False,
        description="Swedish holidays and special days from Svenska Dagar API"
    )

    return holiday_fg


def ingest_holiday_data(years: list, project):
    """
    Ingest holiday data for specified years into Hopsworks.

    Args:
        years: List of years to fetch (e.g., [2025, 2026])
        project: Hopsworks project
    """
    fs = project.get_feature_store()
    holiday_fg = create_holiday_feature_group(fs)

    all_data = []

    for year in years:
        print(f"Fetching holiday data for {year}...")
        df = fetch_swedish_holidays(year)

        if df.empty:
            print(f"  No data for {year}")
            continue

        print(f"  Fetched {len(df)} days")
        all_data.append(df)

    if not all_data:
        print("No holiday data to ingest")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Ensure proper types
    combined_df["week"] = combined_df["week"].astype("Int64")
    combined_df["day_of_week"] = combined_df["day_of_week"].astype("Int64")
    combined_df["is_work_free"] = combined_df["is_work_free"].astype(bool)
    combined_df["is_red_day"] = combined_df["is_red_day"].astype(bool)
    combined_df["is_day_before_holiday"] = combined_df["is_day_before_holiday"].astype(bool)

    # Make date timezone-aware for Hopsworks
    combined_df["date"] = pd.to_datetime(combined_df["date"]).dt.tz_localize("Europe/Stockholm").dt.tz_convert("UTC")

    print(f"\nIngesting {len(combined_df)} holiday records to Hopsworks...")
    holiday_fg.insert(clean_column_names(combined_df), write_options={"wait_for_job": False})
    print("Holiday data ingestion complete!")


def main():
    """Backfill holiday data for relevant years."""

    project = hopsworks.login(
        project=os.environ["HOPSWORKS_PROJECT"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"]
    )

    # Fetch 2025 and 2026 to cover our data range plus future predictions
    years = [2025, 2026]

    print(f"\nBackfilling Swedish holiday data for years: {years}")
    print("=" * 60)

    ingest_holiday_data(years, project)


if __name__ == "__main__":
    main()
