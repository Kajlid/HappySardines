"""
HappySardines - Bus Occupancy Predictor UI (Streamlit version)

A Streamlit app with clickable map and heat map overlay for predicting
bus crowding in Östergötland.
"""

import streamlit as st

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


@st.cache_resource
def get_model():
    """Load model once and cache it."""
    try:
        return load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def generate_heatmap_data(hour, day_of_week, weather, holidays):
    """Generate heat map data by predicting crowding across a grid."""
    model = get_model()
    if model is None:
        return []

    # Create grid of points across Östergötland
    lat_steps = 15
    lon_steps = 20
    lats = np.linspace(BOUNDS["min_lat"], BOUNDS["max_lat"], lat_steps)
    lons = np.linspace(BOUNDS["min_lon"], BOUNDS["max_lon"], lon_steps)

    heatmap_data = []

    for lat in lats:
        for lon in lons:
            try:
                pred_class, confidence, _ = predict_occupancy(
                    lat=lat, lon=lon, hour=hour, day_of_week=day_of_week,
                    weather=weather, holidays=holidays
                )
                # Weight by occupancy level (higher = more crowded = more intense)
                intensity = pred_class / 5.0  # Normalize to 0-1
                if intensity > 0.1:  # Only show if there's some crowding
                    heatmap_data.append([lat, lon, intensity])
            except Exception:
                pass

    return heatmap_data


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

        pred_class, confidence, probs = predict_occupancy(
            lat=lat, lon=lon,
            hour=selected_datetime.hour,
            day_of_week=selected_datetime.weekday(),
            weather=weather,
            holidays=holidays
        )

        return pred_class, confidence, {
            "weather": weather,
            "holidays": holidays,
            "datetime": selected_datetime
        }
    except Exception as e:
        return None, None, str(e)


# Initialize session state
if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = DEFAULT_LAT
if "selected_lon" not in st.session_state:
    st.session_state.selected_lon = DEFAULT_LON

# Header
st.title("🐟 HappySardines")
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

        if st.button("Generate Heat Map", type="primary"):
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
        - 🕐 Time of day
        - 📅 Day of week
        - 🌡️ Weather conditions
        - 🎉 Holidays

        **Data sources:**
        - Bus occupancy data from Östgötatrafiken
        - Weather from Open-Meteo
        - Holidays from Svenska Dagar API

        **Built for KTH ID2223**
        """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Click on the map to select a location")

    # Get heatmap data if available
    heatmap_data = st.session_state.get("heatmap_data", [])

    # Create and display map
    m = create_map(
        selected_lat=st.session_state.selected_lat,
        selected_lon=st.session_state.selected_lon,
        show_heatmap=show_heatmap,
        heatmap_data=heatmap_data
    )

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

with col2:
    st.subheader("🔮 Prediction")

    # Show selected coordinates
    st.markdown(f"**Location:** {st.session_state.selected_lat:.4f}, {st.session_state.selected_lon:.4f}")

    # Make prediction
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

            day_type = "🎉 Holiday" if holidays.get("is_red_day") else (
                "🏖️ Work-free day" if holidays.get("is_work_free") else "📅 Regular day"
            )

            st.markdown(f"""
            **Conditions:**
            - 🌡️ {weather.get('temperature_2m', '?'):.0f}°C
            - {day_type}
            - {selected_datetime.strftime('%A')}
            """)

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
