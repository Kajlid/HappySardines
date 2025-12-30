from util import fetch_json, clean_column_names, haversine_km
import os
from dotenv import load_dotenv
import geohash
import pandas as pd
import hopsworks
from datetime import timedelta, datetime
import gc
import time
import requests

# For a given bus trip and minute window, count the number of nearby events such that:
# • Start time is within the next 0-60 minutes, OR
# • End time was within the last 0-60 minutes,
# Exclude ongoing events,
# • Only events within X km of the bus location.
# time-windowed spatial join

# Feature groups:
# 1. event_raw_fg
# event_id	# Ticketmaster event id (PK)
# start_time	Event start datetime (UTC)
# end_time	# Event end datetime (UTC)
# lat	Latitude
# lon	Longitude
# geohash	Precision 6-7
# city	
# classification
	
# Primary key: event_id
# Event time: start_time
# Online: False

# 2. trip_event_agg_fg
# trip_id	Trip identifier
# window_start	Minute window (same as vehicle features)
# events_starting_next_60m	Count
# events_ended_prev_60m	Count
# events_total_60m	Sum
# geohash	Optional for debugging

# Primary key: trip_id, window_start
# Event time: window_start
# Online: False


# Configuration
load_dotenv()
# countryCode = os.getenv("countryCode")
ticketmaster_key = os.getenv("TICKETMASTER_API_KEY")
TICKETMASTER_BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"

hopsworks_key = os.getenv("HOPSWORKS_API_KEY")
hopsworks_project = os.getenv("HOPSWORKS_PROJECT")

EVENT_RADIUS_KM = 20
GEOHASH_PRECISION = 4
MAX_EVENTS_PER_CALL = 200
EVENT_CALL_CACHE = {}
TM_REQUEST_TIMEOUT = 30  # seconds
TM_MAX_RETRIES = 5
TM_INITIAL_BACKOFF = 2  # seconds


def make_cache_key(geohash, start_dt):
    """Create cache key based on geohash and hour bucket."""
    hour_bucket = start_dt.replace(minute=0, second=0, microsecond=0)
    return (geohash, hour_bucket)

def fetch_ticketmaster_events(start_dt, end_dt, lat, lon, radius_km, debug=False):
    """
    Fetch events from Ticketmaster with rate-limit handling.
    
    Implements exponential backoff for 429 (Too Many Requests) responses.
    Raises RuntimeError if all retries are exhausted.
    """
    start_ts = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_ts = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        f"{TICKETMASTER_BASE_URL}"
        f"?apikey={ticketmaster_key}"
        f"&radius={radius_km}"
        f"&unit=km"
        f"&latlong={lat},{lon}"
        f"&startDateTime={start_ts}"
        f"&endDateTime={end_ts}"
        f"&size={MAX_EVENTS_PER_CALL}"
    )
    
    if debug:
        print(f"  URL: {url[:100]}...")
    
    backoff = TM_INITIAL_BACKOFF
    
    for attempt in range(1, TM_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TM_REQUEST_TIMEOUT)
            
            if debug:
                print(f"  HTTP Status: {resp.status_code}")
            
            # Handle rate limiting
            if resp.status_code == 429:
                if attempt < TM_MAX_RETRIES:
                    print(f"Rate limited (429). Attempt {attempt}/{TM_MAX_RETRIES}. Waiting {backoff}s before retry...")
                    time.sleep(backoff)
                    backoff *= 2  # exponential backoff
                    continue
                else:
                    raise RuntimeError(f"Rate limited after {TM_MAX_RETRIES} retries. Giving up.")
            
            # Raise for other HTTP errors
            resp.raise_for_status()
            
            # Success
            data = resp.json()
            events = data.get("_embedded", {}).get("events", [])
            if debug:
                print(f"  Found {len(events)} events")
            return events
            
        except requests.exceptions.RequestException as e:
            if attempt < TM_MAX_RETRIES:
                print(f"Request failed (attempt {attempt}/{TM_MAX_RETRIES}): {e}")
                time.sleep(backoff)
                backoff *= 2
            else:
                raise RuntimeError(f"Failed to fetch Ticketmaster events after {TM_MAX_RETRIES} attempts: {e}")
    
    return []


def normalize_event(e):
    venue = e["_embedded"]["venues"][0]

    start = pd.to_datetime(e["dates"]["start"]["dateTime"], utc=True)
    end = pd.to_datetime(
        e["dates"].get("end", {}).get("dateTime"), utc=True
    ) if e["dates"].get("end") else None

    lat = float(venue["location"]["latitude"])
    lon = float(venue["location"]["longitude"])

    event_id = e["id"]

    return (
        {
            "event_id": event_id,
            "start_time": start,
            "end_time": end,
            "classification": (
                e["classifications"][0]["segment"]["name"]
                if "classifications" in e else None
            )
        },
        {
            "event_id": event_id,
            "lat": lat,
            "lon": lon,
            "geohash": geohash.encode(lat, lon, precision=GEOHASH_PRECISION),
            "city": venue.get("city", {}).get("name"),
        }
    )



def aggregate_events_for_trip_windows(trip_df, events_df, radius_km):
    """
    Aggregate events near trip locations within specific temporal windows.
    
    For each trip-minute window, count events where:
    - Event is within radius_km spatially
    - Event time window overlaps with [window_start - 60min, window_start + 60min]
    
    This handles multi-day events (e.g., festivals) correctly.
    """
    results = []

    for _, row in trip_df.iterrows():
        ws = row.window_start
        lat, lon = row.lat_mean, row.lon_mean

        if pd.isna(lat) or pd.isna(lon):
            continue

        # Spatial filtering using geohash neighbors
        bus_geohash = geohash.encode(lat, lon, precision=GEOHASH_PRECISION)
        neighbors = geohash.expand(bus_geohash)

        candidates = events_df[events_df["geohash"].isin(neighbors)]

        # Spatial distance filter
        candidates = candidates[
            candidates.apply(
                lambda e, lat=lat, lon=lon: haversine_km(lat, lon, e.lat, e.lon) <= radius_km,
                axis=1
            )
        ]

        # Temporal window: event overlaps with [ws - 60min, ws + 60min]
        # An event [start_time, end_time] overlaps with window [ws - 60min, ws + 60min] if:
        #   event.start_time <= ws + 60min AND event.end_time >= ws - 60min
        window_start = ws - timedelta(minutes=60)
        window_end = ws + timedelta(minutes=60)

        # Ensure event times are timezone-aware for comparison
        event_start = pd.to_datetime(candidates['start_time'], utc=True)
        event_end = pd.to_datetime(candidates['end_time'], utc=True)

        nearby_events = candidates[
            (event_start <= window_end) &
            (event_end >= window_start)
        ]

        results.append({
            "trip_id": row.trip_id,
            "window_start": ws,
            "number_of_events_nearby": len(nearby_events),
        })

    return pd.DataFrame(results)


def ingest_raw_events(sample_points, fs):
    """
    Fetch and ingest raw events from Ticketmaster for sample locations.
    Returns both schedule and location dataframes merged on event_id.
    """
    schedule_rows = []
    location_rows = []
    
    # Debug first few requests to understand why no events are returned
    debug_count = 0

    for _, row in sample_points.iterrows():
        start = row.window_start - timedelta(hours=1)
        end = row.window_start + timedelta(hours=1)

        # Cache API responses to minimize API calls
        cache_key = make_cache_key(row.geohash, start)

        if cache_key in EVENT_CALL_CACHE:
            print(f"Cache hit for geohash {row.geohash}")
            events = EVENT_CALL_CACHE[cache_key]
        else:
            debug = debug_count < 3  # Debug first 3 requests
            print(f"Fetching events for location ({row.lat_mean:.4f}, {row.lon_mean:.4f}), window: {start} to {end}")
            events = fetch_ticketmaster_events(
                start, end,
                row.lat_mean, row.lon_mean,
                EVENT_RADIUS_KM,
                debug=debug
            )
            EVENT_CALL_CACHE[cache_key] = events
            debug_count += 1

        for e in events:
            sched, loc = normalize_event(e)
            schedule_rows.append(sched)
            location_rows.append(loc)

    if not schedule_rows or not location_rows:
        print("No events found; skipping event ingestion.")
        return None

    event_schedule_df = (
        pd.DataFrame(schedule_rows)
        .drop_duplicates("event_id")
    )

    event_location_df = (
        pd.DataFrame(location_rows)
        .drop_duplicates("event_id")
    )

    # -----------------------------------------------------------------
    # Store event schedule (temporal information)
    # -----------------------------------------------------------------
    event_schedule_fg = fs.get_or_create_feature_group(
        name="event_schedule_fg",
        version=1,
        primary_key=["event_id"],
        event_time="start_time",
        online_enabled=False,
        description="Event temporal schedule (start and end times)"
    )

    event_schedule_fg.insert(
        clean_column_names(event_schedule_df),
        write_options={"wait_for_job": True}
    )

    # -----------------------------------------------------------------
    # Store event location (spatial information)
    # -----------------------------------------------------------------
    event_location_fg = fs.get_or_create_feature_group(
        name="event_location_fg",
        version=1,
        primary_key=["event_id"],
        online_enabled=False,
        description="Event location information (lat, lon, city, geohash)"
    )

    event_location_fg.insert(
        clean_column_names(event_location_df),
        write_options={"wait_for_job": True}
    )
    
    # Merge and return for aggregation
    events_df = event_schedule_df.merge(
        event_location_df,
        on="event_id",
        how="inner"
    )
    
    return events_df



def main():
    """
    Ingest events for a given date range.
    """
    project = hopsworks.login(project=hopsworks_project, api_key_value=hopsworks_key)
    fs = project.get_feature_store()
    
    # Parse date range
    start_date = datetime.strptime("2025-11-01", "%Y-%m-%d")
    end_date = datetime.strptime("2025-12-15", "%Y-%m-%d")
    
    # Convert to UTC-aware timestamps for comparison with trip data
    start_date = pd.Timestamp(start_date, tz='UTC')
    end_date = pd.Timestamp(end_date, tz='UTC')
    
    # Load vehicle trip features
    vehicle_trip_agg_fg = fs.get_or_create_feature_group(
        name="vehicle_trip_agg_fg",  
        version=2,
    )
    
    trip_df = vehicle_trip_agg_fg.read()
    print(f"Loaded {len(trip_df)} total trip records from feature group")
    print(f"Trip data date range: {trip_df['window_start'].min()} to {trip_df['window_start'].max()}")
    
    # Ensure window_start is timezone-aware UTC for comparison
    trip_df['window_start'] = pd.to_datetime(trip_df['window_start'], utc=True)
    
    # Filter to requested date range
    filtered_df = trip_df[
        (trip_df['window_start'] >= start_date) &
        (trip_df['window_start'] <= end_date)
    ]
    
    print(f"Filtered to {len(filtered_df)} records in date range {start_date.date()} to {end_date.date()}")
    
    if filtered_df.empty:
        print(f"\n⚠️  No trip data found for date range {start_date.date()} to {end_date.date()}")
        print(f"Available trip dates: {trip_df['window_start'].dt.date.unique()}")
        print("\nTip: Update the date range in main() to match available trip data, or run backfill-transportation-data.py first.")
        return
    
    trip_df = filtered_df
    
    # one spatial point per minute
    trip_df = trip_df.sort_values("window_start")
    trip_df = trip_df.groupby("window_start", as_index=False).first()

    # Sample unique geohash cells
    sample_points = (
        trip_df
        .dropna(subset=["lat_mean", "lon_mean"])
        .assign(geohash=lambda df: df.apply(
            lambda r: geohash.encode(r.lat_mean, r.lon_mean, precision=6),
            axis=1
        ))
        .drop_duplicates("geohash")
    )
    
    print(f"Found {len(sample_points)} unique geohash cells for event lookup")
    
    if sample_points.empty:
        print("No valid sample points found for event ingestion.")
        return
    
    events_df = ingest_raw_events(sample_points, fs)
    
    # If no events were found, skip aggregation
    if events_df is None:
        print("Skipping event aggregation due to no events.")
        return
    
    event_agg_df = aggregate_events_for_trip_windows(
        trip_df=trip_df,
        events_df=events_df,
        radius_km=EVENT_RADIUS_KM
    )
    
    if event_agg_df.empty:
        print("No event-trip associations found.")
        return
    
    trip_event_fg = fs.get_or_create_feature_group(
        name="trip_event_agg_fg",
        version=1,
        primary_key=["trip_id", "window_start"],
        event_time="window_start",
        online_enabled=False,
        description="Count of nearby events per trip-minute window within temporal window [window_start - 60min, window_start + 60min]"
    )

    trip_event_fg.insert(clean_column_names(event_agg_df))
    
    print(f"Event backfill completed successfully for {start_date.date()} to {end_date.date()}.")
    print(f"Stored {len(event_agg_df)} trip-event aggregations in trip_event_agg_fg")
    gc.collect()  # memory cleanup
    
    
if __name__=='__main__':
    main()

