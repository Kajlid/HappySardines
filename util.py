import requests
import time
import xml.etree.ElementTree as ET
import pandas as pd
from haversine import haversine, Unit
from datetime import datetime, timedelta
import great_expectations as ge

MAX_RETRIES = 3
WAIT_SECONDS = 5  # wait between retries

# Open-Meteo API endpoints
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Östergötland region center (centroid of otraf GTFS stops)
DEFAULT_WEATHER_LAT = 58.13
DEFAULT_WEATHER_LON = 15.90

# Weather variables to fetch
HOURLY_WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "weather_code",
]

# Svenska Dagar API for Swedish holidays
SVENSKA_DAGAR_API_URL = "https://sholiday.faboul.se/dagar/v2.1"

def fetch_json(url):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"Retrying in {WAIT_SECONDS} seconds...")
                time.sleep(WAIT_SECONDS)
            else:
                raise RuntimeError(f"Failed to fetch data after {MAX_RETRIES} attempts.")

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove whitespaces in beginning of column names to make sure they are compatible with Hopsworks"""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()            # lowercase everything
        .str.replace(" ", "_")  # replace spaces with underscores
    )
    # Hopsworks requires names to start with a letter:
    df.columns = [
        col if col[0].isalpha() else f"f_{col.lstrip('_')}"
        for col in df.columns
    ]
    return df

def validate_with_expectations(df, expectation_suite, name="dataset"):
    """
    Run Great Expectations validation before ingestion.
    Prints results and raises an error if validation fails.
    """
    ge_df = ge.from_pandas(df)
    validation_result = ge_df.validate(expectation_suite=expectation_suite)
    
    if not validation_result.success:
        print(f"Validation failed for {name}")
        for res in validation_result.results:
            if not res.success:
                print(f" - {res.expectation_config.expectation_type} failed: {res.result}")
        raise ValueError(f"Validation failed for {name}.")
    else:
        print(f"Validation passed for {name}")  

def distance_m(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two points."""
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)

# For Trafikverket situation data
def build_situation_query(day_start: str, day_end: str, api_key: str) -> str:
    return f"""
    <REQUEST>
      <LOGIN authenticationkey="{api_key}" />
      <QUERY objecttype="Situation" schemaversion="1">
        <FILTER>
          <AND>
            <LT name="Deviation.StartTime" value="{day_end}" />
            <OR>
              <GT name="Deviation.EndTime" value="{day_start}" />
              <EQ name="Deviation.ValidUntilFurtherNotice" value="true" />
            </OR>
          </AND>
        </FILTER>

        <INCLUDE>Id</INCLUDE>
        <INCLUDE>Deleted</INCLUDE>

        <INCLUDE>Deviation.Id</INCLUDE>
        <INCLUDE>Deviation.StartTime</INCLUDE>
        <INCLUDE>Deviation.EndTime</INCLUDE>
        <INCLUDE>Deviation.ValidUntilFurtherNotice</INCLUDE>
        <INCLUDE>Deviation.Geometry</INCLUDE>
        <INCLUDE>Deviation.Header</INCLUDE>
        <INCLUDE>Deviation.LocationDescriptor</INCLUDE>
        <INCLUDE>Deviation.RoadNumber</INCLUDE>
        <INCLUDE>Deviation.RoadNumberNumeric</INCLUDE>
        <INCLUDE>Deviation.MessageCode</INCLUDE>
        <INCLUDE>Deviation.MessageType</INCLUDE>
        <INCLUDE>Deviation.SeverityCode</INCLUDE>
        <INCLUDE>Deviation.SeverityText</INCLUDE>

      </QUERY>
    </REQUEST>
    """


def fetch_situations(day_start: str, day_end: str, url:str, api_key:str) -> ET.Element:
    xml_query = build_situation_query(day_start, day_end, api_key)

    response = requests.post(
        url,
        data=xml_query.encode("utf-8"),
        headers={"Content-Type": "application/xml"},
        timeout=60
    )

    if response.status_code != 200:
        print(f"API Error {response.status_code}: {response.text}")
        response.raise_for_status()

    return ET.fromstring(response.content)


# Weather data functions
def fetch_historical_weather(
    start_date: str,
    end_date: str,
    latitude: float = DEFAULT_WEATHER_LAT,
    longitude: float = DEFAULT_WEATHER_LON
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo Archive API.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        latitude: Location latitude (default: Östergötland centroid)
        longitude: Location longitude (default: Östergötland centroid)

    Returns:
        DataFrame with hourly weather data
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_WEATHER_VARIABLES),
        "timezone": "Europe/Stockholm",
    }

    response = requests.get(OPENMETEO_ARCHIVE_URL, params=params, timeout=60)

    if response.status_code != 200:
        print(f"Weather API Error {response.status_code}: {response.text[:500]}")
        response.raise_for_status()

    return _parse_weather_response(response.json(), latitude, longitude)


def fetch_weather_forecast(
    latitude: float = DEFAULT_WEATHER_LAT,
    longitude: float = DEFAULT_WEATHER_LON,
    past_days: int = 1,
    forecast_days: int = 7
) -> pd.DataFrame:
    """
    Fetch weather forecast (and recent past) from Open-Meteo Forecast API.

    Args:
        latitude: Location latitude
        longitude: Location longitude
        past_days: Number of past days to include (max 92)
        forecast_days: Number of forecast days (max 16)

    Returns:
        DataFrame with hourly weather data
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_WEATHER_VARIABLES),
        "timezone": "Europe/Stockholm",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }

    response = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=60)

    if response.status_code != 200:
        print(f"Weather API Error {response.status_code}: {response.text[:500]}")
        response.raise_for_status()

    return _parse_weather_response(response.json(), latitude, longitude)


def _parse_weather_response(data: dict, latitude: float, longitude: float) -> pd.DataFrame:
    """Parse Open-Meteo API response into DataFrame."""
    hourly = data.get("hourly", {})

    if not hourly:
        return pd.DataFrame()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly.get("temperature_2m"),
        "relative_humidity_2m": hourly.get("relative_humidity_2m"),
        "precipitation": hourly.get("precipitation"),
        "rain": hourly.get("rain"),
        "snowfall": hourly.get("snowfall"),
        "cloud_cover": hourly.get("cloud_cover"),
        "wind_speed_10m": hourly.get("wind_speed_10m"),
        "wind_gusts_10m": hourly.get("wind_gusts_10m"),
        "weather_code": hourly.get("weather_code"),
    })

    df["latitude"] = latitude
    df["longitude"] = longitude
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    # Make timezone-aware (UTC for Hopsworks)
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"])
        .dt.tz_localize("Europe/Stockholm")
        .dt.tz_convert("UTC")
    )

    return df


# Swedish holiday data functions
def fetch_swedish_holidays(year: int, month: int = None) -> pd.DataFrame:
    """
    Fetch Swedish holiday data from Svenska Dagar API.

    Args:
        year: Year to fetch (e.g., 2025)
        month: Optional month (1-12). If None, fetches entire year.

    Returns:
        DataFrame with daily holiday information
    """
    if month:
        url = f"{SVENSKA_DAGAR_API_URL}/{year}/{month:02d}"
    else:
        url = f"{SVENSKA_DAGAR_API_URL}/{year}"

    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        print(f"Svenska Dagar API Error {response.status_code}: {response.text[:200]}")
        response.raise_for_status()

    data = response.json()
    days = data.get("dagar", [])

    if not days:
        return pd.DataFrame()

    rows = []
    for day in days:
        rows.append({
            "date": day.get("datum"),
            "weekday": day.get("veckodag"),
            "week": int(day.get("vecka", 0)),
            "day_of_week": int(day.get("dag i vecka", 0)),
            "is_work_free": day.get("arbetsfri dag") == "Ja",
            "is_red_day": day.get("röd dag") == "Ja",
            "is_day_before_holiday": day.get("dag före arbetsfri helgdag") == "Ja",
            "holiday_name": day.get("helgdag"),
            "flag_day": day.get("flaggdag"),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    return df


def get_holiday_features_for_date(date: datetime) -> dict:
    """
    Get holiday features for a specific date.

    Args:
        date: Date to get features for

    Returns:
        Dictionary with holiday features
    """
    url = f"{SVENSKA_DAGAR_API_URL}/{date.year}/{date.month:02d}/{date.day:02d}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return _empty_holiday_features()

        data = response.json()
        days = data.get("dagar", [])

        if not days:
            return _empty_holiday_features()

        day = days[0]
        return {
            "is_work_free": day.get("arbetsfri dag") == "Ja",
            "is_red_day": day.get("röd dag") == "Ja",
            "is_day_before_holiday": day.get("dag före arbetsfri helgdag") == "Ja",
            "is_holiday": day.get("helgdag") is not None,
            "holiday_name": day.get("helgdag"),
        }
    except Exception as e:
        print(f"Error fetching holiday data: {e}")
        return _empty_holiday_features()


def _empty_holiday_features() -> dict:
    """Return empty holiday features dict."""
    return {
        "is_work_free": False,
        "is_red_day": False,
        "is_day_before_holiday": False,
        "is_holiday": False,
        "holiday_name": None,
    }
