"""
HappySardines - Bus Occupancy Predictor UI (Streamlit version)

A Streamlit app with clickable map and heat map overlay for predicting
bus crowding in Östergötland.
"""

import streamlit as st
import pickle

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="HappySardines",
    page_icon="🐟",
    layout="wide"
)

import os
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
from datetime import datetime, timedelta

# Import prediction and data fetching modules
from predictor import predict_occupancy, load_model, OCCUPANCY_LABELS
from weather import get_weather_for_prediction
from holidays import get_holiday_features
from trip_info import load_static_trip_info, find_nearest_trip, load_static_stops_info, find_closest_stop

# Constants
DEFAULT_LAT = 58.4108
DEFAULT_LON = 15.6214
DEFAULT_ZOOM = 10

BOUNDS = {
    "min_lat": 57.8,
    "max_lat": 58.9,
    "min_lon": 14.5,
    "max_lon": 16.8
}

# Color scheme for occupancy levels
OCCUPANCY_COLORS = {
    0: "#22c55e",  # Empty - green
    1: "#22c55e",  # Many seats - green
    2: "#eab308",  # Few seats - yellow
    3: "#f97316",  # Standing - orange
    4: "#ef4444",  # Crushed - red
    5: "#ef4444",  # Full - red
    6: "#6b7280",  # Not accepting - gray
}

static_trip_df = load_static_trip_info()
static_trip_and_stops_df = load_static_stops_info()

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

@st.cache_data
def load_precomputed_heatmaps():
    with open("ui/precomputed_heatmaps.pkl", "rb") as f:
        return pickle.load(f)

precomputed_heatmaps = load_precomputed_heatmaps()

def generate_heatmap_data(hour, day_of_week, weather, holidays):
    """Generate heat map data by predicting crowding across a grid."""
    model = get_model()
    if model is None:
        return []

    # Create grid of points across Östergötland
    # lat_steps = 15
    # lon_steps = 20
    lat_steps = 10
    lon_steps = 15
    lats = np.linspace(BOUNDS["min_lat"], BOUNDS["max_lat"], lat_steps)
    lons = np.linspace(BOUNDS["min_lon"], BOUNDS["max_lon"], lon_steps)

    heatmap_data = []
    total_points = len(lats) * len(lons)
    progress_bar = st.progress(0)
    counter = 0

    for lat in lats:
        for lon in lons:
            try:
                pred_class, confidence, _ = cached_predict_occupancy(
                    lat=lat, lon=lon, hour=hour, day_of_week=day_of_week,
                    weather=weather, holidays=holidays
                )
                # Weight by occupancy level (higher = more crowded = more intense)
                intensity = pred_class / 5.0  # Normalize to 0-1
                if intensity > 0.1:  # Only show if there's some crowding
                    heatmap_data.append([lat, lon, intensity])
            except Exception:
                pass
            
            counter += 1
            progress_bar.progress(counter / total_points)

    progress_bar.empty()
    return heatmap_data

def get_top_crowded_points(n=5, hour=None, day_of_week=None, weather=None, holidays=None):
    """Return top N crowded points with associated trip/stop info."""
    heatmap_data = precomputed_heatmaps.get((hour, day_of_week), [])

    if not heatmap_data:
        # Fallback: small grid prediction
        heatmap_data = generate_heatmap_data(hour, day_of_week, weather, holidays)

    # Sort by intensity descending
    top_points = sorted(heatmap_data, key=lambda x: x[2], reverse=True)[:n]

    top_info = []
    for lat, lon, intensity in top_points:
        trip_info = find_nearest_trip(lat, lon, selected_datetime, static_trip_df)
        closest_stop = None
        if trip_info:
            trip_id = trip_info.get("trip_id")
            closest_stop = find_closest_stop(lat, lon, trip_id, static_trip_and_stops_df)
        top_info.append({
            "lat": lat,
            "lon": lon,
            "intensity": intensity,
            "trip_info": trip_info,
            "closest_stop": closest_stop
        })
    return top_info


def create_map(selected_lat=None, selected_lon=None, show_heatmap=False,
               heatmap_data=None):
    """Create a Folium map with optional marker and heatmap."""
    center_lat = selected_lat if selected_lat else DEFAULT_LAT
    center_lon = selected_lon if selected_lon else DEFAULT_LON

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=DEFAULT_ZOOM,
        tiles="CartoDB positron"
    )

    # Add coverage area rectangle
    folium.Rectangle(
        bounds=[[BOUNDS["min_lat"], BOUNDS["min_lon"]],
                [BOUNDS["max_lat"], BOUNDS["max_lon"]]],
        color="#3388ff",
        fill=False,
        weight=2,
        opacity=0.5,
    ).add_to(m)

    # Add heatmap if enabled
    if show_heatmap and heatmap_data and len(heatmap_data) > 0:
        HeatMap(
            data=heatmap_data,
            min_opacity=0.3,
            radius=25,
            blur=15,
        ).add_to(m)

    # Add marker if location selected
    if selected_lat and selected_lon:
        folium.Marker(
            [selected_lat, selected_lon],
            tooltip=f"Selected: {selected_lat:.4f}, {selected_lon:.4f}",
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
st.markdown("*How packed are buses in Östergötland?*")

# Check if model is available
model = get_model()
if model is None:
    st.error("⚠️ Could not load prediction model. Please check the configuration.")
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
    show_heatmap = st.toggle("Show Crowding Forecast", value=False,
                              help="Display predicted crowding across the region")

    if show_heatmap:
        st.info("🔥 Heat map shows predicted crowding levels. Red = busy, Green = quiet.")

        with st.spinner("Generating predictions across region..."):
            weather = get_weather_for_prediction(DEFAULT_LAT, DEFAULT_LON, selected_datetime)
            holidays = get_holiday_features(selected_datetime)
            st.session_state.heatmap_data = generate_heatmap_data(
                hour, selected_date.weekday(), weather, holidays
            )

    st.divider()

    # About
    with st.expander("About this tool"):
        st.markdown("""
        **How it works:**

        This tool predicts bus crowding levels based on:
        - 📍 Location
        - 🚌 Bus vehicle data
        - 🕐 Time of day
        - 📅 Day of week
        - 🌡️ Weather conditions
        - 🇸🇪 Holidays

        **Data sources:**
        - Bus occupancy data from Östgötatrafiken (Koda API)
        - Weather from Open-Meteo
        - Holidays from Svenska Dagar API

        **Built for KTH ID2223**
        """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Click on the map to select a location")

    # Get heatmap data if available
    # heatmap_data = st.session_state.get("heatmap_data", [])
    heatmap_data = precomputed_heatmaps.get((hour, selected_date.weekday()), []) if show_heatmap else []

    # Create and display map
    m = create_map(
        selected_lat=st.session_state.selected_lat,
        selected_lon=st.session_state.selected_lon,
        show_heatmap=show_heatmap,
        heatmap_data=heatmap_data
    )
    
    weather = get_weather_for_prediction(DEFAULT_LAT, DEFAULT_LON, selected_datetime)
    holidays = get_holiday_features(selected_datetime)
    
    # top_crowded = get_top_crowded_points(n=3)

    # # Add markers on the map
    # for point in top_crowded:
    #     folium.Marker(
    #         [point["lat"], point["lon"]],
    #         tooltip=f"Crowding: {int(point['intensity']*100)}%",
    #         icon=folium.Icon(color="red", icon="bus", prefix="fa")
    #     ).add_to(m)

    # Render the map
    map_data = st_folium(
        m,
        height=500,
        use_container_width=True,
        key="map"
    )

    # Handle map clicks
    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        st.session_state.selected_lat = clicked["lat"]
        st.session_state.selected_lon = clicked["lng"]
        st.rerun()

    # Display list below map
    # for idx, point in enumerate(top_crowded, 1):
    #     trip_info = point["trip_info"]
    #     closest_stop = point["closest_stop"]
    #     intensity_pct = int(point["intensity"] * 100)
    #     st.markdown(f"**{idx}. Location:** {point['lat']:.4f}, {point['lon']:.4f} (Crowding: {intensity_pct}%)")
    #     if trip_info:
    #         route = trip_info.get("route_short_name") or trip_info.get("route_long_name", "Unknown route")
    #         st.markdown(f"- Route: {route}")
    #         if trip_info.get("route_desc"):
    #             st.markdown(f"- Bus type: {trip_info['route_desc']}")
    #         if trip_info.get("trip_id"):
    #             st.markdown(f"- Trip ID: {trip_info['trip_id']}")
    #     if closest_stop:
    #         st.markdown(f"- Closest stop: {closest_stop}")

with col2:
    st.subheader("Prediction")

    # Show selected coordinates
    st.markdown(f"**Location:** {st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f}")

    # Make prediction
    with st.spinner("Fetching occupancy prediction..."):
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
                Prediction confidence: {confidence:.0%}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Context info
        if isinstance(result, dict):
            weather = result["weather"]
            holidays = result["holidays"]
            trip_info = result.get("trip_info")

            day_type = "📅🇸🇪 Holiday" if holidays.get("is_red_day") else (
                "📅🇸🇪 Work-free day" if holidays.get("is_work_free") else "📅 Regular day"
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
                    info_lines.append(f"Bus type: {route_desc}")

                # Trip ID (always show)
                trip_id = trip_info.get("trip_id")
                if trip_id is not None:
                    info_lines.append(f"Trip ID: {trip_id}")
                    
                    # Closest stop
                    closest_stop = find_closest_stop(
                        st.session_state.selected_lat,
                        st.session_state.selected_lon,
                        trip_id,
                        static_trip_and_stops_df  # assume this is loaded once globally
                    )
                    if closest_stop:
                        info_lines.append(f"Closest stop: {closest_stop}")

                # Display
                st.markdown("**Bus Info:**\n- " + "\n- ".join(info_lines))

            conditions_lines = []
            conditions_lines.append(f"Temperature: {weather.get('temperature_2m', '?'):.0f}°C")
            conditions_lines.append(f"Weekday: {day_type}")
            conditions_lines.append(f"{selected_datetime.strftime('%A')}")
            
            if weather.get('snowfall') != 0:
                conditions_lines.append("Risk of snow")
                
            if weather.get('rain') != 0:
                conditions_lines.append("Risk of rain")
                
            st.markdown("**Conditions:**\n- " + "\n- ".join(conditions_lines))

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
