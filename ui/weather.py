"""
Weather forecast fetching for HappySardines predictions.

Uses Open-Meteo API to get weather forecasts.
"""

import requests
from datetime import datetime

# Open-Meteo API
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Weather variables needed for prediction
WEATHER_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
]


def get_weather_for_prediction(lat: float, lon: float, target_datetime: datetime) -> dict:
    """
    Get weather forecast for a specific location and time.

    Args:
        lat: Latitude
        lon: Longitude
        target_datetime: Target datetime for prediction

    Returns:
        Dict with weather features for the model
    """
    try:
        # Determine if we need forecast or recent past
        now = datetime.now()
        days_ahead = (target_datetime.date() - now.date()).days

        # Open-Meteo provides up to 16 days forecast
        if days_ahead > 16:
            print(f"Warning: Date too far in future, using defaults")
            return _default_weather()

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(WEATHER_VARIABLES),
            "timezone": "Europe/Stockholm",
            "forecast_days": max(2, days_ahead + 1),  # At least today + tomorrow
        }

        # Include past days if looking at today
        if days_ahead <= 0:
            params["past_days"] = 1

        response = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=30)

        if response.status_code != 200:
            print(f"Weather API error: {response.status_code}")
            return _default_weather()

        data = response.json()
        hourly = data.get("hourly", {})

        if not hourly:
            return _default_weather()

        # Find the matching hour in the response
        times = hourly.get("time", [])
        target_str = target_datetime.strftime("%Y-%m-%dT%H:00")

        try:
            idx = times.index(target_str)
        except ValueError:
            # Try to find closest hour
            target_hour = target_datetime.hour
            target_date = target_datetime.strftime("%Y-%m-%d")

            for i, t in enumerate(times):
                if t.startswith(target_date) and f"T{target_hour:02d}:" in t:
                    idx = i
                    break
            else:
                print(f"Could not find matching time for {target_datetime}")
                return _default_weather()

        return {
            "temperature_2m": hourly.get("temperature_2m", [None])[idx] or 10.0,
            "precipitation": hourly.get("precipitation", [None])[idx] or 0.0,
            "cloud_cover": hourly.get("cloud_cover", [None])[idx] or 50.0,
            "wind_speed_10m": hourly.get("wind_speed_10m", [None])[idx] or 5.0,
        }

    except Exception as e:
        print(f"Error fetching weather: {e}")
        return _default_weather()


def _default_weather() -> dict:
    """Return default weather values."""
    return {
        "temperature_2m": 10.0,  # Typical Swedish temp
        "precipitation": 0.0,
        "cloud_cover": 50.0,
        "wind_speed_10m": 5.0,
    }
