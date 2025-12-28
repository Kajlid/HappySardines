import hopsworks
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import json
import time
import gc
import numpy as np
from dotenv import load_dotenv
import great_expectations as ge
import zipfile
import py7zr
import tempfile
import pyarrow as pa
import base64
from google.transit import gtfs_realtime_pb2

# Configuration
load_dotenv()
MAX_RETRIES = 3
WAIT_SECONDS = 5  # wait between retries
KODA_RT_API_BASE_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-rt/otraf/VehiclePositions"   # real-time data of the vehicles
KODA_STATIC_API_BASE_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-static/otraf"       # static data of the vehicles
CHECK_INTERVAL = 60  # Interval in seconds

hopsworks_key = os.getenv("HOPSWORKS_API_KEY")
hopsworks_project = os.getenv("HOPSWORKS_PROJECT")

out_dir = os.environ["out_dir"]
# Create the output folder for data
os.makedirs(out_dir, exist_ok=True)

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
        # vehicle_label = getattr(vehicle_obj, 'label', None)

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

def download_static_file(url, out_path):
    try:
        if os.path.exists(out_path):
            print(f"File already exists: {out_path}.")
            return
        
        # Send a GET request to download the file
        response = requests.get(url, stream=True)

        if response.status_code == 200 and "application/zip" in response.headers.get("Content-Type", ""):
            with open(out_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Zip file downloaded successfully: {out_path}")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
            print(response.headers.get("Content-Type", ""))
    except Exception as e:
        print(f"Error while downloading the file: {e}")


# Every hour, we fetch new data from the real-time API
def fetch_rt_data(date, hour):
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

        
def fetch_static_data(date):
    url = f"{KODA_STATIC_API_BASE_URL}?date={date}&key={os.environ['KODA_API_KEY']}"
    
    print(f"Checking static data for date: {date}")
    
    file_name_date = date.replace("-", "_")
    static_file_path = os.path.join(out_dir, f"gtfs_static_{file_name_date}.zip")

    while True:
        try:
            # Send HTTP HEAD request
            response = requests.head(url)
            
            if response.status_code == 202:
                print(f"[{datetime.now()}] HTTP 202 Accepted. Retrying in {CHECK_INTERVAL} seconds...")
                time.sleep(CHECK_INTERVAL)
            elif response.status_code == 200:
                print(f"[{datetime.now()}] HTTP 200 OK. Attempting to download the file...")
                download_static_file(url, out_path=static_file_path)
                break
            else:
                print(f"[{datetime.now()}] Unexpected response: {response.status_code}. Exiting.")
                break
        except Exception as e:
            print(f"[{datetime.now()}] Error: {e}")
            time.sleep(CHECK_INTERVAL)
            
            
    # After writing data to zip, we create dataframes:
    with zipfile.ZipFile(static_file_path, 'r') as z:
        # print("Files inside ZIP:", z.namelist())
        
        with z.open('stop_times.txt') as f:
            stop_times_df = pd.read_csv(f)
        
        with z.open('stops.txt') as f:
            stops_df = pd.read_csv(f)
            
        with z.open('calendar.txt') as f:
            calendar_df = pd.read_csv(f)
            
        calendar_df["start_date"] = pd.to_datetime(calendar_df["start_date"], format="%Y%m%d")
        calendar_df["end_date"]   = pd.to_datetime(calendar_df["end_date"], format="%Y%m%d")
        
        with z.open('calendar_dates.txt') as f:
            calendar_dates_df = pd.read_csv(f)   # overides "is_active" for special days
            
        calendar_dates_df["date"] = pd.to_datetime(calendar_dates_df["date"], format="%Y%m%d")
        
        with z.open('routes.txt') as f:
            routes_df = pd.read_csv(f)
            
        with z.open('agency.txt') as f:
            agency_df = pd.read_csv(f)
            
        with z.open('trips.txt') as f:
            trips_df = pd.read_csv(f)
        
        # Geometric information about the planned path of a trip   
        with z.open('shapes.txt') as f:
            shapes_df = pd.read_csv(f)
            
    # dimension table with route attributes, agency attributes and trip identifiers
    trip_dim = (trips_df
    .merge(routes_df, on="route_id", how="left", validate="many_to_one")
    .merge(agency_df, on="agency_id", how="left", validate="many_to_one")
    )
    
    # fact table with stops
    trip_stops_fact = (stop_times_df
    .merge(stops_df, on="stop_id", how="left", validate="many_to_one")
    .sort_values(["trip_id", "stop_sequence"])
    )
    
    records = []

    for _, row in calendar_df.iterrows():
        dates = pd.date_range(row.start_date, row.end_date)
        for d in dates:
            weekday = d.strftime("%A").lower()
            if row[weekday] == 1:
                records.append({
                    "service_id": row.service_id,
                    "date": d,
                    "is_active": 1
                })

    calendar_fact = pd.DataFrame(records)
    
    if calendar_fact.empty:
        calendar_fact = pd.DataFrame(columns=['service_id', 'date', 'is_active'])

    
    # Override activity of service on special days when service does not follow schedule
    adds = calendar_dates_df[calendar_dates_df.exception_type == 1]
    removes = calendar_dates_df[calendar_dates_df.exception_type == 2]
    
    print(calendar_df.columns)
    print(calendar_dates_df.columns)

    calendar_fact = (
        calendar_fact
        .merge(adds.assign(is_active=1), on=["service_id", "date"], how="outer")
        .merge(removes.assign(is_active=0), on=["service_id", "date"], how="left", suffixes=("", "_remove"))
    )
    
    # Sort and normalize geographic dimension table
    shapes_df = (
    shapes_df
    .assign(
        shape_pt_sequence=lambda df: df["shape_pt_sequence"].astype(int),
        shape_pt_lat=lambda df: df["shape_pt_lat"].astype(float),
        shape_pt_lon=lambda df: df["shape_pt_lon"].astype(float),
        shape_dist_traveled=lambda df: pd.to_numeric(df["shape_dist_traveled"], errors="coerce")
    )
    .sort_values(["shape_id", "shape_pt_sequence"])
    )
    
    return trip_dim, trip_stops_fact, calendar_fact, shapes_df
            
            
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
        
        
def ingest_static_data(dates, project):
    fs = project.get_feature_store()
    print(dates)
    
    trip_info_fg = fs.get_or_create_feature_group(
        name="static_trip_info_fg",
        version=1,
        primary_key=["trip_id", "feed_date"],
        description="Static information about a planned trip",
        online_enabled=False
    )
    
    trip_and_stops_info_fg = fs.get_or_create_feature_group(
        name="static_trip_and_stops_info_fg",
        version=1,
        primary_key=["trip_id", "stop_sequence", "feed_date"],
        description="Static information about a planned trip",
        online_enabled=False
    ) 
    for date in dates:
        date_string = datetime.strftime(date, "%Y-%m-%d")
        file_name_date = date_string.replace("-", "_")
        static_file_path = os.path.join(out_dir, f"gtfs_static_{file_name_date}.zip")
        if not os.path.exists(static_file_path):
            print(f"Processing date: {date_string}")
            trip_static_df, trip_stops_fact_df, calendar_fact_df, shapes_df = fetch_static_data(date_string)
            
            trip_static_df["feed_date"] = date

            static_trip_stops_df = (trip_stops_fact_df
                .merge(trip_static_df, on="trip_id", how="left", validate="many_to_one")
            )
            
            static_trip_stops_df["feed_date"] = date

            trip_static_df.dropna(
                subset=["trip_id"],
                inplace=True
            )
            
            static_trip_stops_df = static_trip_stops_df.drop_duplicates(subset=["trip_id", "stop_id", "stop_sequence"])
            
            # Convert NaNs to nullable strings
            for col, dtype in trip_static_df.dtypes.items():
                if dtype == "object":
                    trip_static_df[col] = trip_static_df[col].where(
                        trip_static_df[col].notna(), None
                    )
                    
            for col, dtype in static_trip_stops_df.dtypes.items():
                if dtype == "object":
                    static_trip_stops_df[col] = static_trip_stops_df[col].where(static_trip_stops_df[col].notna(), None)
            
            trip_info_fg.insert(clean_column_names(trip_static_df),
                write_options={"wait_for_job": True})
            
            trip_and_stops_info_fg.insert(clean_column_names(static_trip_stops_df),
                write_options={"wait_for_job": True})
            
    
            
def ingest_rt_data(dates, project):
    fs = project.get_feature_store()
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
    
    # vehicle_trip_rt_fg = fs.get_or_create_feature_group(
    #     name="vehicle_trip_rt_fg",  
    #     version=1,
    #     primary_key=["vehicle_id", "trip_id", "timestamp"],
    #     event_time="timestamp",
    #     description="Real-time vehicle-trip, speed, and occupancy features",
    #     online_enabled=True
    # )
    
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
            rows = fetch_rt_data(date_string, hour)
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

        # vehicle_trip_rt_fg.insert(       # daily inserts
        #     clean_column_names(daily_rt_df),
        #     write_options={"wait_for_job": False}
        # )
        
        daily_aggregated_df = get_aggregated_temporal_trip_features(daily_rt_df)
        # Drop duplicates in the key columns
        daily_aggregated_df = daily_aggregated_df.drop_duplicates(subset=["trip_id", "window_start"], keep="last")
        
        # Data validation before ingestion
        validate_with_expectations(daily_aggregated_df, agg_expectation_suite, name="Aggregated vehicle features")
        
        vehicle_trip_agg_fg.insert(
            clean_column_names(daily_aggregated_df),
            write_options={"wait_for_job": False}
        )
        
        
        # Discard to free memory
        del dfs, daily_rt_df, daily_aggregated_df
        gc.collect()
            
        print(f"Completed ingestion for date: {date_string}\n")
    
    
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


def store_shapes_dataset(project, shapes_df, feed_date: str):
    """
    Store GTFS shapes in Hopsworks object storage (Datasets).
    """
    shapes_df = clean_column_names(shapes_df)

    dataset_base = "gtfs/shapes"
    dataset_path = f"{dataset_base}/feed_date={feed_date}"

    # Create dataset if it does not exist
    try:
        project.get_dataset(dataset_base)
    except Exception:
        project.create_dataset(
            name=dataset_base,
            description="GTFS shapes stored per feed version/date"
        )

    local_tmp = f"/tmp/shapes_{feed_date}.parquet"
    shapes_df.to_parquet(local_tmp, index=False)

    dataset = project.get_dataset(dataset_base)
    dataset.upload(local_tmp, dataset_path, overwrite=True)

    print(f"Stored shapes dataset at: {dataset_path}")

    
def main():
    # project = hopsworks.login(host=hopsworks_host, project=hopsworks_project, api_key_value=hopsworks_key, engine="python")
    project = hopsworks.login(project=hopsworks_project, api_key_value=hopsworks_key)
        
    # Ingest data over the last month
    start_date = datetime.now() - timedelta(days=30)  # 1 month
    end_date = datetime.now() - timedelta(days=1)     # yesterday
    
    # Keep track of months we have ingested static data for
    ingested_months = set()
    
    # Iterate week by week
    for week_start in pd.date_range(start=start_date, end=end_date, freq="W-MON"):  # weeks starting on Monday
        week_end = week_start + pd.Timedelta(days=6)
        month_key = week_start.strftime("%Y-%m")
        if week_end > end_date:
            week_end = end_date
        
        print(f"Ingesting week: {week_start.date()} - {week_end.date()}")
        
        dates = pd.date_range(week_start, week_end, normalize=True) # start at midnight each date
        
        if month_key not in ingested_months:
            # Use first date of the month to fetch static GTFS
            first_of_month = week_start.replace(day=1)
            ingest_static_data([first_of_month], project)
            ingested_months.add(month_key)
            
        ingest_rt_data(dates, project=project)
        print(f"Completed ingestion for week starting {week_start.date()}")

        gc.collect()    # memory cleanup

    

if __name__=='__main__':
    main()
        
        