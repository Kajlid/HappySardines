import hopsworks
import os
import pandas as pd
import requests
import tempfile
import zipfile
import py7zr
import time
import great_expectations as ge
from haversine import haversine, Unit
from datetime import datetime, timedelta
import sys
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2
import xml.etree.ElementTree as ET
import numpy as np
import gc

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from util import clean_column_names, distance_m, validate_with_expectations

load_dotenv()

# Configuration
KODA_RT_API_BASE_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-rt/otraf/VehiclePositions"
KODA_STATIC_API_BASE_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-static/otraf"
TRAFIKVERKET_API_URL = "https://api.trafikinfo.trafikverket.se/v2/data.xml"

TRAFIKVERKET_API_KEY = os.environ["TRAFIKVERKET_API_KEY"]
KODA_KEY = os.environ.get("KODA_API_KEY")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 2  # seconds
MAX_BACKOFF = 30  # seconds

def parse_vehiclepositions(content: bytes, date: str, hour: int):
    """Parse GTFS-RT VehiclePositions protobuf content into a list of dict rows.

    Uses gtfs_realtime_bindings when available, otherwise attempts a generic protobuf parse.
    """
    rows = []

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(content)

    for entity in feed.entity:
        # skip entities without 'vehicle'
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle

        trip = getattr(v, 'trip', None)
        pos = getattr(v, 'position', None)

        ts = None
        if getattr(v, 'timestamp', None):
            try:
                ts = int(v.timestamp)
            except Exception:
                ts = None

        ts_dt = pd.to_datetime(ts, unit='s') if ts else None

        vehicle_obj = getattr(v, 'vehicle', None) or v    # v.vehicle

        vehicle_id = getattr(vehicle_obj, 'id', None)

        row = {
            'feed_entity_id': getattr(entity, 'id', None),
            'date': date,
            'hour': hour,
            'timestamp': ts_dt,
            'vehicle_id': vehicle_id,
            'trip_id': getattr(trip, 'trip_id', None),
            'position_lat': getattr(pos, 'latitude', None),
            'position_lon': getattr(pos, 'longitude', None),
            'bearing': getattr(pos, 'bearing', None),
            'speed': getattr(pos, 'speed', None),
            'occupancy_status': getattr(v, 'occupancy_status', None),
        }
        rows.append(row)

    return rows

# Every hour, we fetch new data from the real-time API
def fetch_koda_rt_data(date, hour):
    url = f"{KODA_RT_API_BASE_URL}?date={date}&hour={hour}&key={os.environ['KODA_API_KEY']}"
    
    print(f"Checking RT data for date: {date}, hour: {hour}")

    rows = []
    try:
        response = requests.get(url, stream=True)
        print(f"HTTP {response.status_code} response received.")
        
        if response.status_code == 200 and response.content:
            
            # Save 7z to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".7z") as tmp_7z:
                tmp_7z.write(response.content)
                tmp_7z_path = tmp_7z.name
                
                # Extract protobuf(s) from 7z
                with tempfile.TemporaryDirectory() as tmp_dir:
                    with py7zr.SevenZipFile(tmp_7z.name, mode='r') as archive:
                        archive.extractall(path=tmp_dir)
                        all_files = archive.getnames()  # lists all files including paths
                        # print("Files in archive:", all_files)
                        for rel_path in all_files:
                            if rel_path.endswith(".pb"):
                                file_path = os.path.join(tmp_dir, rel_path)
                                # print("Processing:", file_path, "Size:", os.path.getsize(file_path))
                                with open(file_path, "rb") as f:
                                    content = f.read()  # raw protobuf binary
                                    try:
                                        rows.extend(parse_vehiclepositions(content, date, hour))
                                    except Exception as e:
                                        print(f"Failed to parse protobuf {rel_path}: {e}")
                                    
                # os.remove(tmp_7z_path)
                print(f"Hour {hour}: {len(rows)} vehicle positions parsed")
                return rows
                
        
        else:
            print("No data to save.")
            return []

        
    except Exception as e:
        print(f"Error fetching RT data: {e}")

def fetch_koda_static_data(date_str: str, tmp_dir: str) -> str:
    """Download GTFS static ZIP file from KODA API."""
    url = f"{KODA_STATIC_API_BASE_URL}?date={date_str}&key={KODA_KEY}"
    zip_path = os.path.join(tmp_dir, f"gtfs_static_{date_str.replace('-', '_')}.zip")
    
    if os.path.exists(zip_path):
        return zip_path
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 200 and "application/zip" in response.headers.get("Content-Type", ""):
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return zip_path
        else:
            print(f"  Warning: Could not download static data for {date_str}")
            return None
    except Exception as e:
        print(f"  Error downloading static data: {e}")
        return None

def ingest_koda_rt_data(dates, fs):
    print(dates)
    
    # Data expectations
    agg_expectation_suite = ge.core.ExpectationSuite(
        expectation_suite_name="agg_expectation_suite"
    )

    for col in ["trip_id", "window_start"]:
        agg_expectation_suite.add_expectation(
            ge.core.ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": col}
            )
        )
    
    # For model to predict occupancy on aggregated features
    vehicle_trip_agg_fg = fs.get_or_create_feature_group(
        name="vehicle_trip_agg_fg",  
        version=2,
        primary_key=["trip_id", "window_start"],
        event_time="window_start",
        description="1 minute windows of vehicle-trip, speed, and occupancy features",
        expectation_suite=agg_expectation_suite,
        online_enabled=False
    )
    
    for date in dates:
        date_string = datetime.strftime(date, "%Y-%m-%d")
        dfs = []
        for hour in range(24):
            print(f"Processing hour: {hour}")
            rows = fetch_koda_rt_data(date_string, hour)
            if not rows:
                continue
            
            print(f"Fetched {len(rows)} rows")
            rt_df = pd.DataFrame(rows)
            rt_df = rt_df.loc[:, ~rt_df.columns.duplicated()]

            dfs.append(clean_column_names(rt_df))
        
        if not dfs:
            continue
        
        daily_rt_df = pd.concat(dfs, ignore_index=True)
        
        # Remove any duplicated columns (by name)
        daily_rt_df = daily_rt_df.loc[:, ~daily_rt_df.columns.duplicated()]
        print("big_df columns:", daily_rt_df.columns)
        
        # Now create window_start once, for the whole day
        daily_rt_df['timestamp'] = pd.to_datetime(daily_rt_df['timestamp'], unit='s') # to_datetime assumes nanoseconds by default

        # Remove rows without trip_id or vehicle_id
        daily_rt_df = daily_rt_df[daily_rt_df['trip_id'].notna() & daily_rt_df['vehicle_id'].notna()]

        daily_aggregated_df = get_aggregated_temporal_trip_features(daily_rt_df)
        # Drop duplicates in the key columns
        daily_aggregated_df = daily_aggregated_df.drop_duplicates(subset=["trip_id", "window_start"], keep="last")
        
        # Data validation before ingestion
        validate_with_expectations(daily_aggregated_df, agg_expectation_suite, name="Aggregated vehicle features")
        
        vehicle_trip_agg_fg.insert(
            clean_column_names(daily_aggregated_df),
            write_options={"wait_for_job": False}
        )
            
        print(f"Completed ingestion for date: {date_string}\n")
        return daily_aggregated_df
    
    
def get_aggregated_temporal_trip_features(rt_df):
    rt_df = rt_df.sort_values(["timestamp", "trip_id", "vehicle_id"])  
    rt_df['window_start'] = rt_df['timestamp'].dt.floor('1min')    # create windows of a minute
    
    print("rt_df columns: ", rt_df.columns)
    
    vehicle_trip_1min_df = (
        rt_df
        .groupby(["trip_id", "window_start"], as_index=False)
        .agg(
            vehicle_id=("vehicle_id",              # the trip might have switched vehicle
                            lambda x: x.mode().iloc[0] if not x.mode().empty else None),
            avg_speed=("speed", "mean"),
            max_speed=("speed", "max"),
            n_positions=("speed", "count"),
            speed_std=("speed", "std"),
            lat_min=("position_lat", "min"),
            lat_max=("position_lat", "max"),
            lat_mean=("position_lat", "mean"),
            lon_min=("position_lon", "min"),
            lon_max=("position_lon", "max"),
            lon_mean=("position_lon", "mean"),
            bearing_min=("bearing", "min"),
            bearing_max=("bearing", "max"),
            occupancy_mode=("occupancy_status",
                            lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        )
    )
    
    vehicle_trip_1min_df["date"] = vehicle_trip_1min_df["window_start"].dt.date
    vehicle_trip_1min_df["hour"] = vehicle_trip_1min_df["window_start"].dt.hour
    vehicle_trip_1min_df["day_of_week"] = vehicle_trip_1min_df["window_start"].dt.weekday
    vehicle_trip_1min_df["window_start"] = pd.to_datetime(vehicle_trip_1min_df["window_start"])

    return vehicle_trip_1min_df

# Trafikverket Traffic Data Fetching & Parsing
def build_situation_query(day_start: str, day_end: str) -> str:
    return f"""
    <REQUEST>
      <LOGIN authenticationkey="{TRAFIKVERKET_API_KEY}" />
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

# Fetch Trafikverket traffic situations
def fetch_situations(day_start: str, day_end: str) -> ET.Element:
    xml_query = build_situation_query(day_start, day_end)

    response = requests.post(
        TRAFIKVERKET_API_URL,
        data=xml_query.encode("utf-8"),
        headers={"Content-Type": "application/xml"},
        timeout=60
    )
    
    if response.status_code != 200:
        print(f"API Error {response.status_code}: {response.text}")
        response.raise_for_status()
    
    return ET.fromstring(response.content)


def parse_wkt_centroid(wkt: str):
    if not wkt:
        return None, None

    if wkt.startswith("POINT"):
        lon, lat = map(
            float,
            wkt.replace("POINT (", "").replace(")", "").split()
        )
        return lat, lon

    if wkt.startswith("LINESTRING"):
        coords = [
            tuple(map(float, p.split()))
            for p in wkt.replace("LINESTRING (", "").replace(")", "").split(",")
        ]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return np.mean(lats), np.mean(lons)

    return None, None

def extract_geometry_wkt(geometry_el):
    """Extract WKT geometry from Geometry element, preferring Point > Line > WGS84."""
    if geometry_el is None:
        return None
    
    wgs84_point = geometry_el.find("Point/WGS84")
    if wgs84_point is not None:
        return wgs84_point.text
    
    wgs84_line = geometry_el.find("Line/WGS84")
    if wgs84_line is not None:
        return wgs84_line.text
    
    wgs84_simple = geometry_el.find("WGS84")
    return wgs84_simple.text if wgs84_simple is not None else None


def extract_deviation_info(dev):
    """Extract deviation information from a Deviation element."""
    if dev is None:
        return {
            "road_number": None,
            "road_number_numeric": None,
            "start_time": None,
            "end_time": None,
            "header": None,
            "message_code": None,
            "message_type": None,
            "severity_code": None,
            "severity_text": None,
            "location_description": None,
            "valid_until_further_notice": False,
            "geometry_wkt": None,
            "lat": None,
            "lon": None,
            "deviation_id": None,
        }
    
    road_number = dev.findtext("RoadNumber")
    road_number_numeric = dev.findtext("RoadNumberNumeric")
    start_time = dev.findtext("StartTime")
    end_time = dev.findtext("EndTime")
    header = dev.findtext("Header")
    location_description = dev.findtext("LocationDescriptor")
    valid_until_further_notice = dev.findtext("ValidUntilFurtherNotice") == "true"
    deviation_id = dev.findtext("Id")
    message_code = dev.findtext("MessageCode")
    message_type = dev.findtext("MessageType")
    severity_code = dev.findtext("SeverityCode")
    severity_text = dev.findtext("SeverityText")
    
    geometry_el = dev.find("Geometry")
    geometry_wkt = extract_geometry_wkt(geometry_el)
    lat, lon = parse_wkt_centroid(geometry_wkt)
    
    return {
        "road_number": road_number,
        "road_number_numeric": road_number_numeric,
        "start_time": start_time,
        "end_time": end_time,
        "header": header,
        "message_code": message_code,
        "message_type": message_type,
        "severity_code": severity_code,
        "severity_text": severity_text,
        "location_description": location_description,
        "valid_until_further_notice": valid_until_further_notice,
        "geometry_wkt": geometry_wkt,
        "lat": lat,
        "lon": lon,
        "deviation_id": deviation_id,
    }


def parse_situations(root: ET.Element, ingestion_date: datetime.date) -> pd.DataFrame:
    rows = []
    
    for situation in root.findall(".//Situation"):
        situation_id = situation.findtext("Id")
        deleted = situation.findtext("Deleted") == "true"

        deviations = situation.findall("Deviation")
        if not deviations:
            deviations = [None]

        for dev in deviations:
            dev_info = extract_deviation_info(dev)

            rows.append({
                "event_id": situation_id,
                "deviation_id": dev_info["deviation_id"],
                "start_time": dev_info["start_time"],
                "end_time": dev_info["end_time"],
                "latitude": dev_info["lat"],
                "longitude": dev_info["lon"],
                "geometry_wkt": dev_info["geometry_wkt"],
                "affected_road": dev_info["road_number"],
                "road_number_numeric": dev_info["road_number_numeric"],
                "location_description": dev_info["location_description"],
                "message_code": dev_info["message_code"],
                "message_type": dev_info["message_type"],
                "severity_code": dev_info["severity_code"],
                "severity_text": dev_info["severity_text"],
                "header": dev_info["header"],
                "valid_until_further_notice": dev_info["valid_until_further_notice"],
                "deleted": deleted,
                "source": "trafikverket",
                "ingestion_date": ingestion_date
            })

    return pd.DataFrame(rows)

def ingest_trafikverket_events(dates, project):
    fs = project.get_feature_store()

    traffic_fg = fs.get_or_create_feature_group(
        name="trafikverket_traffic_event_fg",
        version=2,
        primary_key=["event_id", "deviation_id"],
        event_time="start_time",
        online_enabled=False,
        description="Raw traffic events (Situations/Deviations) from Trafikverket"
    )

    for date in dates:
        day_start = date.strftime("%Y-%m-%dT00:00:00")
        day_end = date.strftime("%Y-%m-%dT23:59:59")

        root = fetch_situations(day_start, day_end)
        df = parse_situations(root, ingestion_date=date.date())

        if df.empty:
            print(f"No situations found for {date.date()}")
            continue

        # Convert to datetime and ensure timezone-aware UTC
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce", utc=True)

        # Remove rows with null start_time (required field)
        df = df.dropna(subset=["start_time"])
        
        if df.empty:
            print(f"No valid situations after filtering for {date.date()}")
            continue
        
        # For missing end_time, use start_time + 1 day (ongoing event)
        # This ensures we have a valid timestamp instead of NaT
        missing_end = df["end_time"].isna()
        df.loc[missing_end, "end_time"] = df.loc[missing_end, "start_time"] + timedelta(days=1)

        # Ensure numeric fields are proper types
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["road_number_numeric"] = pd.to_numeric(df["road_number_numeric"], errors="coerce")
        
        # Convert booleans to proper type
        df["valid_until_further_notice"] = df["valid_until_further_notice"].astype(bool)
        df["deleted"] = df["deleted"].astype(bool)

        print(f"Ingesting {len(df)} traffic events for {date.date()}")
        traffic_fg.insert(df, write_options={"wait_for_job": False})
        
        return df


# Daily feature ingestion
def main():
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()
    
    # Calculate yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    yesterday_ts = pd.Timestamp(yesterday.date(), tz='UTC')
    
    print(f"\n{'='*70}")
    print(f"Running daily feature pipeline for {yesterday.date()}")
    print(f"{'='*70}\n")
    
    tmp_dir = tempfile.mkdtemp()
    rt_rows_all = []
    
    try:
        # Fetch KODA RT vehicle data for all 24 hours
        print(f"Fetching KODA real-time vehicle positions...")
        dates = [yesterday]
        trip_agg_df = ingest_koda_rt_data(dates, fs)
        
        # Fetch Trafikverket traffic events
        print(f"\nFetching Trafikverket traffic events...")
        dates = [yesterday_ts]
        
        traffic_df = ingest_trafikverket_events(dates, project)
        
        # Create enriched features combining trip + traffic
        print(f"\nCreating enriched features...")
        
        if traffic_df.empty:
            enriched_df = trip_agg_df.copy()
            enriched_df["has_nearby_event"] = 0
            enriched_df["num_nearby_traffic_events"] = 0
            enriched_df["nearby_event_severity"] = None
            enriched_df["affected_roads"] = None
        else:
            # For each trip, find nearby traffic events
            enriched_data = []
            for idx, trip_row in trip_agg_df.iterrows():
                window_start = trip_row["window_start"]
                trip_lat, trip_lon = trip_row["lat_mean"], trip_row["lon_mean"]
                
                # Find active traffic events within 500m
                if pd.notna(trip_lat) and pd.notna(trip_lon):
                    nearby = traffic_df[
                        (traffic_df["start_time"] <= window_start) &
                        (traffic_df["end_time"] >= window_start)
                    ].copy()
                    
                    nearby["distance_m"] = nearby.apply(
                        lambda r: distance_m(trip_lat, trip_lon, r["latitude"], r["longitude"]),
                        axis=1
                    )
                    nearby = nearby[nearby["distance_m"] <= 500]
                else:
                    nearby = pd.DataFrame()
                
                if nearby.empty:
                    enriched_data.append({
                        "has_nearby_event": 0,
                        "num_nearby_traffic_events": 0,
                        "nearby_event_severity": None,
                        "affected_roads": None,
                    })
                else:
                    severity_codes = nearby["severity_code"].dropna()
                    max_severity = int(severity_codes.max()) if len(severity_codes) > 0 else None
                    affected_roads = nearby["affected_road"].dropna().unique()
                    roads_str = ",".join(str(r) for r in affected_roads) if len(affected_roads) > 0 else None
                    
                    enriched_data.append({
                        "has_nearby_event": int(len(nearby) > 0),
                        "num_nearby_traffic_events": len(nearby),
                        "nearby_event_severity": max_severity,
                        "affected_roads": roads_str,
                    })
            
            enriched_features = pd.DataFrame(enriched_data)
            enriched_df = pd.concat([trip_agg_df.reset_index(drop=True), enriched_features], axis=1)
        
        # Ingest to combined feature group
        print(f"  Creating/updating trip_occupancy_features_daily...")
        enriched_fg = fs.get_or_create_feature_group(
            name="trip_occupancy_features_daily",
            version=1,
            primary_key=["trip_id", "window_start"],
            event_time="window_start",
            online_enabled=False,
            description="Daily enriched trip features with vehicle metrics and nearby traffic events"
        )
        
        enriched_df.columns = (
            enriched_df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        
        enriched_fg.insert(clean_column_names(enriched_df), write_options={"wait_for_job": False})
        print(f"  Ingested {len(enriched_df)} enriched records to trip_occupancy_features_daily")
        
        print(f"\n{'='*70}")
        print(f"Daily feature pipeline completed successfully for {yesterday.date()}")
        print(f"{'='*70}\n")
        
    finally:
        # Cleanup
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        gc.collect()


if __name__ == "__main__":
    main()
