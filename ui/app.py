"""
HappySardines - Bus Occupancy Predictor UI (Streamlit version)

A Streamlit app with clickable map and heat map overlay for predicting
bus crowding in Östergötland.
"""

import streamlit as st
import json

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="HappySardines",
    page_icon="🐟",
    layout="wide"
)

import os
import folium
from streamlit_folium import st_folium
import numpy as np
from datetime import datetime, timedelta

import hopsworks

# Import prediction and data fetching modules
from predictor import predict_occupancy, load_model, OCCUPANCY_LABELS
from weather import get_weather_for_prediction
from holidays import get_holiday_features
from trip_info import load_static_trip_info, find_nearest_trip, load_static_stops_info, find_closest_stop
from contours import load_contours_from_file, grid_to_cells_geojson

# Constants
DEFAULT_LAT = 58.4108
DEFAULT_LON = 15.6214
DEFAULT_ZOOM = 9  # Slightly zoomed out to show more of the region

# Bounds derived from actual GTFS stop locations (3119 stops)
# Run ui/get_boundaries.py to recalculate if needed
BOUNDS = {
    "min_lat": 56.6414,
    "max_lat": 58.8654,
    "min_lon": 14.6144,
    "max_lon": 16.9578,
}

# Color scheme for occupancy levels (must match contours.py CLASS_COLORS)
OCCUPANCY_COLORS = {
    0: "#22c55e",  # Empty - green
    1: "#84cc16",  # Many seats - lime
    2: "#eab308",  # Few seats - yellow
    3: "#f97316",  # Standing - orange
    4: "#ef4444",  # Crushed - red
    5: "#ef4444",  # Full - red
    6: "#6b7280",  # Not accepting - gray
}

# Lazy-load static data (deferred to avoid blocking app startup)
@st.cache_resource
def get_static_trip_df():
    """Load static trip info from Hopsworks (cached)."""
    try:
        return load_static_trip_info()
    except Exception as e:
        print(f"Warning: Could not load static trip info: {e}")
        return None

@st.cache_resource
def get_static_stops_df():
    """Load static stops info from Hopsworks (cached)."""
    try:
        return load_static_stops_info()
    except Exception as e:
        print(f"Warning: Could not load static stops info: {e}")
        return None


@st.cache_resource
def get_model():
    """Load model once and cache it."""
    try:
        return load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_data
def cached_predict_occupancy(lat, lon, hour, day_of_week, weather, holidays):
    return predict_occupancy(lat, lon, hour, day_of_week, weather, holidays)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_heatmaps_from_hopsworks():
    """
    Fetch all precomputed heatmaps from Hopsworks Feature Store.

    Returns dict mapping (hour, weekday) -> GeoJSON FeatureCollection
    """
    try:
        print("Fetching heatmaps from Hopsworks...")
        project = hopsworks.login()
        fs = project.get_feature_store()

        # Get the feature group
        heatmap_fg = fs.get_feature_group("heatmap_geojson_fg", version=1)

        # Read all data
        df = heatmap_fg.read()

        if df is None or df.empty:
            print("No heatmap data found in Hopsworks")
            return {}

        # Convert to dict with tuple keys
        heatmaps = {}
        for _, row in df.iterrows():
            key = (int(row["hour"]), int(row["weekday"]))
            geojson = json.loads(row["geojson"])
            heatmaps[key] = geojson

        print(f"Loaded {len(heatmaps)} heatmaps from Hopsworks")
        return heatmaps

    except Exception as e:
        print(f"Error fetching heatmaps from Hopsworks: {e}")
        return {}


def load_precomputed_contours():
    """Load precomputed contour GeoJSON from file (not cached to pick up new files)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    contours_path = os.path.join(script_dir, "precomputed_contours.json")

    if os.path.exists(contours_path):
        try:
            contours = load_contours_from_file(contours_path)
            print(f"Loaded {len(contours)} precomputed time slots from {contours_path}")
            return contours
        except Exception as e:
            print(f"Error loading contours: {e}")
            return {}
    print(f"Contours file not found: {contours_path}")
    return {}


def generate_contours_on_demand(hour, day_of_week, weather, holidays):
    """
    Generate grid cell GeoJSON on-demand if precomputed data is not available.
    This is slower but provides a fallback.
    """
    model = get_model()
    if model is None:
        return None

    # Grid for on-demand generation (smaller for speed)
    lat_steps = 15
    lon_steps = 20
    lats = np.linspace(BOUNDS["min_lat"], BOUNDS["max_lat"], lat_steps)
    lons = np.linspace(BOUNDS["min_lon"], BOUNDS["max_lon"], lon_steps)

    lat_step = (BOUNDS["max_lat"] - BOUNDS["min_lat"]) / (lat_steps - 1)
    lon_step = (BOUNDS["max_lon"] - BOUNDS["min_lon"]) / (lon_steps - 1)

    prediction_data = []
    for lat in lats:
        for lon in lons:
            try:
                pred_class, confidence, _ = cached_predict_occupancy(
                    lat=lat, lon=lon, hour=hour, day_of_week=day_of_week,
                    weather=weather, holidays=holidays
                )
                prediction_data.append([lat, lon, pred_class])
            except Exception:
                prediction_data.append([lat, lon, 0])

    # Convert to GeoJSON grid cells
    return grid_to_cells_geojson(prediction_data, lat_step, lon_step)


def get_test_contour_geojson():
    """
    Return a simple hardcoded test GeoJSON to verify rendering works.
    Creates a small grid of cells with different colors.
    """
    # Create a 3x3 grid of test cells around Linköping
    center_lat = 58.41
    center_lon = 15.62
    cell_size = 0.15

    # Test predictions: mix of classes
    test_data = [
        (center_lat - cell_size, center_lon - cell_size, 0),  # green
        (center_lat - cell_size, center_lon, 0),              # green
        (center_lat - cell_size, center_lon + cell_size, 1),  # green
        (center_lat, center_lon - cell_size, 0),              # green
        (center_lat, center_lon, 2),                          # yellow
        (center_lat, center_lon + cell_size, 2),              # yellow
        (center_lat + cell_size, center_lon - cell_size, 0),  # green
        (center_lat + cell_size, center_lon, 3),              # orange
        (center_lat + cell_size, center_lon + cell_size, 0),  # green
    ]

    return grid_to_cells_geojson(test_data, cell_size, cell_size)


def get_contour_geojson(hour, day_of_week, weather=None, holidays=None):
    """
    Get contour GeoJSON for the given hour and day of week.

    Tries sources in order:
    1. Hopsworks Feature Store (primary)
    2. Local JSON file (fallback)
    3. Test contours (last resort)
    """
    key = (hour, day_of_week)

    # Try Hopsworks first
    hopsworks_heatmaps = fetch_heatmaps_from_hopsworks()
    if key in hopsworks_heatmaps:
        geojson = hopsworks_heatmaps[key]
        n_features = len(geojson.get("features", []))
        print(f"Found heatmap in Hopsworks for {key}: {n_features} features")
        return geojson

    # Fall back to local JSON file
    precomputed = load_precomputed_contours()
    if key in precomputed:
        geojson = precomputed[key]
        n_features = len(geojson.get("features", []))
        print(f"Found heatmap in local file for {key}: {n_features} features")
        return geojson

    # Last resort: test contours
    print(f"No heatmap for {key}, using test contours")
    return get_test_contour_geojson()


def create_map(selected_lat=None, selected_lon=None, show_heatmap=False,
               contour_geojson=None):
    """Create a Folium map with optional marker and contour overlay."""
    center_lat = selected_lat if selected_lat else DEFAULT_LAT
    center_lon = selected_lon if selected_lon else DEFAULT_LON

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=DEFAULT_ZOOM,
        tiles="CartoDB positron"
    )

    # Add contour overlay if enabled
    if show_heatmap and contour_geojson and contour_geojson.get("features"):
        # Add each contour level as a separate GeoJSON layer
        folium.GeoJson(
            contour_geojson,
            style_function=lambda feature: {
                'fillColor': feature['properties']['color'],
                'fillOpacity': feature['properties'].get('fillOpacity', 0.35),
                'color': 'none',  # No border
                'weight': 0
            },
            name="Crowding Forecast"
        ).add_to(m)

    # Add coverage area rectangle (subtle border)
    folium.Rectangle(
        bounds=[[BOUNDS["min_lat"], BOUNDS["min_lon"]],
                [BOUNDS["max_lat"], BOUNDS["max_lon"]]],
        color="#6b7280",
        fill=False,
        weight=1,
        opacity=0.3,
        dash_array="5, 5",
    ).add_to(m)

    # Add marker if location selected
    if selected_lat and selected_lon:
        folium.Marker(
            [selected_lat, selected_lon],
            tooltip=f"Selected: {selected_lat:.4f}, {selected_lon:.4f}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    return m


def make_prediction(lat, lon, selected_datetime):
    """Make prediction and return formatted result."""
    if lat is None or lon is None:
        return None, None, None

    # Check bounds
    if not (BOUNDS["min_lat"] <= lat <= BOUNDS["max_lat"] and
            BOUNDS["min_lon"] <= lon <= BOUNDS["max_lon"]):
        return None, None, "Location outside coverage area"

    try:
        weather = get_weather_for_prediction(lat, lon, selected_datetime)
        holidays = get_holiday_features(selected_datetime)

        pred_class, confidence, probs = cached_predict_occupancy(
            lat=lat, lon=lon,
            hour=selected_datetime.hour,
            day_of_week=selected_datetime.weekday(),
            weather=weather,
            holidays=holidays
        )

        trip_info = None
        static_trip_df = get_static_trip_df()
        if static_trip_df is not None:
            trip_info = find_nearest_trip(lat, lon, selected_datetime, static_trip_df)

        return pred_class, confidence, {
            "weather": weather,
            "holidays": holidays,
            "datetime": selected_datetime,
            "trip_info": trip_info
        }
    except Exception as e:
        return None, None, str(e)


# Initialize session state
if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = DEFAULT_LAT
if "selected_lon" not in st.session_state:
    st.session_state.selected_lon = DEFAULT_LON

# Header
st.title("HappySardines")
st.markdown("*Predicted bus crowding in Östergötland*")

# Check if model is available
model = get_model()
if model is None:
    st.error("Could not load prediction model. Please check the configuration.")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Settings")

    # Date/time selection
    st.subheader("When?")
    date_option = st.radio("Date", ["Today", "Tomorrow"], horizontal=True)
    hour = st.slider("Hour", 5, 23, 8)

    today = datetime.now().date()
    selected_date = today if date_option == "Today" else today + timedelta(days=1)
    selected_datetime = datetime.combine(selected_date, datetime.min.time().replace(hour=hour))

    st.markdown(f"**{selected_datetime.strftime('%A, %B %d at %H:00')}**")

    st.divider()

    # View mode
    st.subheader("View Mode")
    show_heatmap = st.toggle("Show Crowding Forecast", value=True,
                              help="Display predicted crowding across the region")

    if show_heatmap:
        st.markdown("""
        **Legend:**
        <div style="display: flex; flex-direction: column; gap: 4px; font-size: 14px;">
            <div><span style="display: inline-block; width: 16px; height: 16px; background: #22c55e; border-radius: 3px; vertical-align: middle;"></span> Empty</div>
            <div><span style="display: inline-block; width: 16px; height: 16px; background: #84cc16; border-radius: 3px; vertical-align: middle;"></span> Many seats</div>
            <div><span style="display: inline-block; width: 16px; height: 16px; background: #eab308; border-radius: 3px; vertical-align: middle;"></span> Few seats</div>
            <div><span style="display: inline-block; width: 16px; height: 16px; background: #f97316; border-radius: 3px; vertical-align: middle;"></span> Standing room</div>
            <div><span style="display: inline-block; width: 16px; height: 16px; background: #ef4444; border-radius: 3px; vertical-align: middle;"></span> Crowded</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # About
    with st.expander("About this tool"):
        st.markdown("""
        **How it works:**

        This tool predicts bus crowding levels based on:
        - 📍 Location
        - 🕐 Time of day
        - 📅 Day of week
        - 🌡️ Weather conditions
        - 🇸🇪 Holidays

        **Data sources:**
        - Bus occupancy data from Östgötatrafiken (KODA API)
        - Weather from Open-Meteo
        - Holidays from Svenska Dagar API

        **Built for KTH ID2223**
        """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Click on the map to select a location")

    # Get weather/holidays for on-demand generation fallback
    weather = get_weather_for_prediction(DEFAULT_LAT, DEFAULT_LON, selected_datetime)
    holidays = get_holiday_features(selected_datetime)

    # Get contour GeoJSON for current hour/day
    contour_geojson = None
    if show_heatmap:
        contour_geojson = get_contour_geojson(
            hour, selected_date.weekday(),
            weather=weather, holidays=holidays
        )

    # Create and display map
    m = create_map(
        selected_lat=st.session_state.selected_lat,
        selected_lon=st.session_state.selected_lon,
        show_heatmap=show_heatmap,
        contour_geojson=contour_geojson
    )

    # Render the map - key includes hour and day to force re-render on time change
    map_data = st_folium(
        m,
        height=500,
        use_container_width=True,
        key=f"map_{hour}_{selected_date.weekday()}"
    )

    # Handle map clicks
    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        st.session_state.selected_lat = clicked["lat"]
        st.session_state.selected_lon = clicked["lng"]
        st.rerun()

with col2:
    st.subheader("Prediction")

    # Show selected coordinates
    st.markdown(f"**Location:** {st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f}")

    # Make prediction
    with st.spinner("Fetching prediction..."):
        pred_class, confidence, result = make_prediction(
            st.session_state.selected_lat,
            st.session_state.selected_lon,
            selected_datetime
        )

    if pred_class is not None:
        label_info = OCCUPANCY_LABELS[pred_class]
        color = OCCUPANCY_COLORS[pred_class]

        # Result card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border-left: 4px solid {color};
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
        ">
            <div style="font-size: 1.3em; font-weight: 600; color: {color};">
                {label_info['icon']} {label_info['label']}
            </div>
            <div style="margin-top: 8px; color: #374151;">
                {label_info['message']}
            </div>
            <div style="margin-top: 12px; font-size: 0.9em; opacity: 0.8;">
                Confidence: {confidence:.0%}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Context info
        if isinstance(result, dict):
            weather = result["weather"]
            holidays = result["holidays"]
            trip_info = result.get("trip_info")

            day_type = "Holiday" if holidays.get("is_red_day") else (
                "Work-free day" if holidays.get("is_work_free") else "Regular day"
            )

            if trip_info:
                info_lines = []

                # Route number or name
                route_number = trip_info.get("route_short_name")
                route_long_name = trip_info.get("route_long_name")
                if route_number and route_long_name:
                    info_lines.append(f"{route_number} - {route_long_name}")
                elif route_number:
                    info_lines.append(f"Route: {route_number}")
                elif route_long_name:
                    info_lines.append(f"Route: {route_long_name}")

                # Bus type / description
                route_desc = trip_info.get("route_desc")
                if route_desc:
                    info_lines.append(f"Type: {route_desc}")

                # Trip ID
                trip_id = trip_info.get("trip_id")
                if trip_id is not None:
                    # Closest stop
                    static_stops_df = get_static_stops_df()
                    closest_stop = find_closest_stop(
                        st.session_state.selected_lat,
                        st.session_state.selected_lon,
                        trip_id,
                        static_stops_df
                    )
                    if closest_stop:
                        info_lines.append(f"Nearest stop: {closest_stop}")

                if info_lines:
                    st.markdown("**Bus Info:**\n- " + "\n- ".join(info_lines))

            # Weather conditions
            conditions = []
            temp = weather.get('temperature_2m')
            if temp is not None:
                conditions.append(f"{temp:.0f}°C")

            if weather.get('snowfall', 0) > 0:
                conditions.append("Snow")
            if weather.get('rain', 0) > 0:
                conditions.append("Rain")

            conditions.append(day_type)
            conditions.append(selected_datetime.strftime('%A'))

            st.markdown("**Conditions:** " + " | ".join(conditions))

    elif isinstance(result, str):
        st.error(result)
    else:
        st.info("Click on the map to select a location")

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; opacity: 0.6;'>Built for KTH ID2223 - Scalable Machine Learning</div>",
    unsafe_allow_html=True
)
