"""
Swedish holiday lookup for HappySardines predictions.

Uses Svenska Dagar API to get holiday information.
"""

import requests
from datetime import datetime

# Svenska Dagar API
SVENSKA_DAGAR_API_URL = "https://sholiday.faboul.se/dagar/v2.1"


def get_holiday_features(target_datetime: datetime) -> dict:
    """
    Get holiday features for a specific date.

    Args:
        target_datetime: Target datetime

    Returns:
        Dict with holiday features for the model
    """
    try:
        date = target_datetime.date()
        url = f"{SVENSKA_DAGAR_API_URL}/{date.year}/{date.month:02d}/{date.day:02d}"

        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            print(f"Holiday API error: {response.status_code}")
            return _default_holidays(target_datetime)

        data = response.json()
        days = data.get("dagar", [])

        if not days:
            return _default_holidays(target_datetime)

        day = days[0]

        return {
            "is_work_free": day.get("arbetsfri dag") == "Ja",
            "is_red_day": day.get("röd dag") == "Ja",
            "is_day_before_holiday": day.get("dag före arbetsfri helgdag") == "Ja",
            "holiday_name": day.get("helgdag"),
            "day_of_week": int(day.get("dag i vecka", target_datetime.weekday() + 1)) - 1,  # Convert to 0-indexed
        }

    except Exception as e:
        print(f"Error fetching holiday data: {e}")
        return _default_holidays(target_datetime)


def _default_holidays(target_datetime: datetime) -> dict:
    """Return default holiday values based on day of week."""
    day_of_week = target_datetime.weekday()

    # Weekends are typically work-free
    is_weekend = day_of_week >= 5

    return {
        "is_work_free": is_weekend,
        "is_red_day": day_of_week == 6,  # Sundays are red days
        "is_day_before_holiday": False,
        "holiday_name": None,
        "day_of_week": day_of_week,
    }
