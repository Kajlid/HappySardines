import hopsworks
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import json
import time
import numpy as np
from dotenv import load_dotenv
import great_expectations as ge
import zipfile
import py7zr
import base64
try:
    from gtfs_realtime_bindings import gtfs_realtime_pb2
except Exception:
    gtfs_realtime_pb2 = None

# Configuration
load_dotenv()
MAX_RETRIES = 3
WAIT_SECONDS = 5  # wait between retries
KODA_RT_API_BASE_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-rt/sl/VehiclePositions"   # real-time data of the vehicles
KODA_STATIC_API_BASE_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-static/sl"       # static data of the vehicles
CHECK_INTERVAL = 60  # Interval in seconds

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

    if gtfs_realtime_pb2 is None:
        raise RuntimeError("gtfs_realtime_bindings is not available.")

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(content)

    for entity in feed.entity:
        # skip entities without 'vehicle'
        if not getattr(entity, 'vehicle', None):
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
        vehicle_label = getattr(vehicle_obj, 'label', None)

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

    try:
        response = requests.get(url, stream=True)
        print(f"HTTP {response.status_code} response received.")
        
        try:
            text_preview = response.text[:100]
            print("Response body preview:", text_preview)
        except Exception as e:
            print("Could not read response text:", e)

        # Save file locally if content exists
        if response.status_code == 200 and response.content:
            # file_path = os.path.join(out_dir, f"vehiclepositions-{date}-{hour}.bin")
            # with open(file_path, "wb") as f:
            #     f.write(response.content)
            # print(f"File saved: {file_path}")
            
            # Parse and return rows for this hour
            try:
                parsed = parse_vehiclepositions(response.content, date, hour)
                return parsed
            except Exception as e:
                print(f"Failed to parse GTFS-RT content: {e}")
                return []
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
        print("Files inside ZIP:", z.namelist())
        
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
    
    # Override activity of service on special days when service does not follow schedule
    adds = calendar_dates_df[calendar_dates_df.exception_type == 1]
    removes = calendar_dates_df[calendar_dates_df.exception_type == 2]

    calendar_fact = (
        calendar_fact
        .merge(adds.assign(is_active=1), on=["service_id", "date"], how="outer")
        .merge(removes.assign(is_active=0), on=["service_id", "date"], how="left", suffixes=("", "_remove"))
    )

    calendar_fact["is_active"] = (
        calendar_fact["is_active"]
        .fillna(calendar_fact["is_active_remove"])
        .fillna(1)
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
    

def extract_rt_data():
    for hour in range(24):
        rt_file = os.path.join(out_dir, f"otraf-vehiclepositions-2025-12-10-{hour}.bin")  

        print("File size:", os.path.getsize(rt_file))
            
        extract_dir = os.path.join(out_dir, "extracted_rt")
        os.makedirs(extract_dir, exist_ok=True)
            
        with py7zr.SevenZipFile(rt_file, mode='r') as z:
            z.extractall(path=extract_dir)     
            
def make_requests(dates):
    print(dates)
    all_rows = []
    for date in dates:
        date_string = datetime.strftime(date, "%Y-%m-%d")
        print(f"Processing date: {date_string}")
        trip_dim, trip_stops_fact, calendar_fact, shapes_df = fetch_static_data(date_string)
        for hour in range(24):
            print(f"Processing hour: {hour}")
            rows = fetch_rt_data(date_string, hour)
            if rows:
                all_rows.extend(rows)
        print(f"Completed processing for date: {date_string}\n")
    # Build dataframe from collected rows and save
    if all_rows:
        df = pd.DataFrame(all_rows)
        
        df_enriched = df.merge(
            trip_dim,
            on="trip_id",
            how="left",
            validate="many_to_one"
        )
        
        # We want only valid trips:
        df_enriched["date"] = pd.to_datetime(df_enriched["date"])

        df_enriched = df_enriched.merge(
            calendar_fact[calendar_fact.is_active == 1],
            on=["service_id", "date"],
            how="inner"
        )
        
        df_enriched = clean_column_names(df_enriched)
        
        return df_enriched
    
    else:
        return []
    
        
        