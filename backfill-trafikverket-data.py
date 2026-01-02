import hopsworks
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import numpy as np
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import hashlib

load_dotenv()

TRAFIKVERKET_API_URL = "https://api.trafikinfo.trafikverket.se/v2/data.xml"
TRAFIKVERKET_API_KEY = os.environ["TRAFIKVERKET_API_KEY"]

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


def main():
    project = hopsworks.login(
        project=os.environ["HOPSWORKS_PROJECT"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"]
    )
    
    # Parse date range
    start_date = datetime.strptime("2025-11-01", "%Y-%m-%d")
    end_date = datetime.strptime("2025-12-31", "%Y-%m-%d")
    
    # Convert to UTC-aware timestamps for comparison with trip data
    start_date = pd.Timestamp(start_date, tz='UTC')
    end_date = pd.Timestamp(end_date, tz='UTC')

    dates = pd.date_range(start_date, end_date, normalize=True)
    ingest_trafikverket_events(dates, project)


if __name__ == "__main__":
    main()



