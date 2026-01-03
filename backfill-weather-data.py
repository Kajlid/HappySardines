"""
Backfill historical weather data from Open-Meteo API into Hopsworks feature store.

This script fetches hourly weather data for the Östergötland region (where otraf operates)
and stores it in a feature group that can be joined with trip data for ML training.

Open-Meteo API: https://open-meteo.com/en/docs/historical-weather-api
- Free, no API key required
- Historical data available back to 1940
- 5-day processing delay for most recent data
"""

import hopsworks
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

from util import (
    fetch_historical_weather,
    clean_column_names,
    DEFAULT_WEATHER_LAT,
    DEFAULT_WEATHER_LON,
)

load_dotenv()


def create_weather_feature_group(fs):
    """Create or get the weather feature group in Hopsworks."""

    weather_fg = fs.get_or_create_feature_group(
        name="weather_hourly_fg",
        version=1,
        primary_key=["timestamp", "latitude", "longitude"],
        event_time="timestamp",
        online_enabled=False,
        description="Hourly weather data from Open-Meteo for Östergötland region"
    )

    return weather_fg


def ingest_weather_data(
    start_date: str,
    end_date: str,
    project,
    latitude: float = DEFAULT_WEATHER_LAT,
    longitude: float = DEFAULT_WEATHER_LON
):
    """
    Ingest weather data for a date range into Hopsworks.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        project: Hopsworks project
        latitude: Location latitude
        longitude: Location longitude
    """
    fs = project.get_feature_store()
    weather_fg = create_weather_feature_group(fs)

    print(f"Fetching weather data from {start_date} to {end_date}...")
    df = fetch_historical_weather(start_date, end_date, latitude, longitude)

    if df.empty:
        print("No weather data to ingest")
        return

    # Ensure proper types for Hopsworks
    df["temperature_2m"] = pd.to_numeric(df["temperature_2m"], errors="coerce")
    df["relative_humidity_2m"] = pd.to_numeric(df["relative_humidity_2m"], errors="coerce")
    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
    df["rain"] = pd.to_numeric(df["rain"], errors="coerce")
    df["snowfall"] = pd.to_numeric(df["snowfall"], errors="coerce")
    df["cloud_cover"] = pd.to_numeric(df["cloud_cover"], errors="coerce")
    df["wind_speed_10m"] = pd.to_numeric(df["wind_speed_10m"], errors="coerce")
    df["wind_gusts_10m"] = pd.to_numeric(df["wind_gusts_10m"], errors="coerce")
    df["weather_code"] = pd.to_numeric(df["weather_code"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["hour"] = df["hour"].astype("Int64")

    print(f"  Fetched {len(df)} hourly weather records")
    print(f"Ingesting weather data to Hopsworks...")

    weather_fg.insert(clean_column_names(df), write_options={"wait_for_job": False})
    print("Weather data ingestion complete!")


def main():
    """Backfill weather data for the same date range as trip data."""

    project = hopsworks.login(
        project=os.environ["HOPSWORKS_PROJECT"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"]
    )

    # Match the date range used in backfill-trafikverket-data.py
    start_date = "2025-11-01"
    end_date = "2025-12-31"

    # Open-Meteo has ~5 day delay, so cap at 5 days ago
    max_date = datetime.now() - timedelta(days=5)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    if end_dt > max_date:
        end_date = max_date.strftime("%Y-%m-%d")
        print(f"Adjusted end_date to {end_date} (Open-Meteo 5-day delay)")

    print(f"\nBackfilling weather data from {start_date} to {end_date}")
    print(f"Location: Östergötland region (lat={DEFAULT_WEATHER_LAT}, lon={DEFAULT_WEATHER_LON})")
    print(f"(Coordinates based on centroid of otraf GTFS stops)")
    print("=" * 60)

    ingest_weather_data(start_date, end_date, project)


if __name__ == "__main__":
    main()
