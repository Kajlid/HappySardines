import requests
import time
import xml.etree.ElementTree as ET
import pandas as pd
from haversine import haversine, Unit
import great_expectations as ge

MAX_RETRIES = 3
WAIT_SECONDS = 5  # wait between retries

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