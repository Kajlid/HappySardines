"""
HappySardines - Bus Occupancy Predictor UI

A Gradio app that predicts how crowded buses are in Östergötland based on
location, time, weather, and holidays.
"""

import os
import gradio as gr
import folium
from datetime import datetime, timedelta

# Import prediction and data fetching modules
from predictor import predict_occupancy, predict_occupancy_mock, OCCUPANCY_LABELS
from weather import get_weather_for_prediction
from holidays import get_holiday_features

# Try to load model on startup, fall back to mock
USE_MOCK = os.environ.get("USE_MOCK", "false").lower() == "true"

if not USE_MOCK:
    try:
        from predictor import load_model
        load_model()
        print("Model loaded successfully - using real predictions")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Using mock predictions for testing")
        USE_MOCK = True

# Select predictor function
_predict_fn = predict_occupancy_mock if USE_MOCK else predict_occupancy

# Default map center: Linköping
DEFAULT_LAT = 58.4108
DEFAULT_LON = 15.6214
DEFAULT_ZOOM = 11

# Östergötland bounds (roughly)
BOUNDS = {
    "min_lat": 57.8,
    "max_lat": 58.9,
    "min_lon": 14.5,
    "max_lon": 16.8
}

# Preset locations for quick selection
PRESET_LOCATIONS = {
    "Linköping Central": (58.4158, 15.6253),
    "Norrköping Central": (58.5942, 16.1826),
    "Linköping University": (58.3980, 15.5762),
    "Mjärdevi Science Park": (58.4027, 15.5672),
    "Motala": (58.5375, 15.0364),
    "Finspång": (58.7050, 15.7700),
}


def create_map(lat=DEFAULT_LAT, lon=DEFAULT_LON):
    """Create a Folium map centered on location with marker."""
    m = folium.Map(
        location=[lat, lon],
        zoom_start=DEFAULT_ZOOM,
        tiles="CartoDB positron"
    )

    # Add marker at selected location
    folium.Marker(
        [lat, lon],
        popup=f"Selected: {lat:.4f}, {lon:.4f}",
        icon=folium.Icon(color="blue", icon="bus", prefix="fa")
    ).add_to(m)

    # Add a rectangle showing the coverage area
    folium.Rectangle(
        bounds=[[BOUNDS["min_lat"], BOUNDS["min_lon"]],
                [BOUNDS["max_lat"], BOUNDS["max_lon"]]],
        color="#3388ff",
        fill=False,
        weight=1,
        opacity=0.3,
        popup="Coverage area"
    ).add_to(m)

    return m._repr_html_()


def make_prediction(lat, lon, date_choice, hour):
    """
    Make occupancy prediction for given inputs.

    Returns formatted result HTML.
    """
    if lat is None or lon is None:
        return create_result_card(
            "Please select a location",
            "Use the preset buttons or enter coordinates.",
            "gray",
            None
        )

    # Validate coordinates are in Östergötland
    if not (BOUNDS["min_lat"] <= lat <= BOUNDS["max_lat"] and
            BOUNDS["min_lon"] <= lon <= BOUNDS["max_lon"]):
        return create_result_card(
            "Location outside coverage area",
            f"Please select a location within Östergötland. Selected: {lat:.4f}, {lon:.4f}",
            "gray",
            None
        )

    # Determine date
    today = datetime.now().date()
    if date_choice == "Today":
        selected_date = today
    else:  # Tomorrow
        selected_date = today + timedelta(days=1)

    selected_datetime = datetime.combine(selected_date, datetime.min.time().replace(hour=int(hour)))

    try:
        # Get weather forecast
        weather = get_weather_for_prediction(lat, lon, selected_datetime)

        # Get holiday features
        holidays = get_holiday_features(selected_datetime)

        # Make prediction
        prediction, confidence, probabilities = _predict_fn(
            lat=lat,
            lon=lon,
            hour=int(hour),
            day_of_week=selected_date.weekday(),
            weather=weather,
            holidays=holidays
        )

        # Format result
        label_info = OCCUPANCY_LABELS[prediction]

        # Build context string
        day_name = selected_date.strftime("%A")
        day_type = "Holiday" if holidays.get("is_red_day") else ("Work-free day" if holidays.get("is_work_free") else "Regular day")
        temp = weather.get("temperature_2m", "?")

        context = f"{temp:.0f}°C  •  {day_name}  •  {day_type}"

        return create_result_card(
            label_info["label"],
            label_info["message"],
            label_info["color"],
            context,
            confidence
        )

    except Exception as e:
        return create_result_card(
            "Prediction failed",
            f"Error: {str(e)}",
            "gray",
            None
        )


def create_result_card(title, message, color, context, confidence=None):
    """Create HTML result card."""
    color_map = {
        "green": "#22c55e",
        "yellow": "#eab308",
        "orange": "#f97316",
        "red": "#ef4444",
        "gray": "#6b7280"
    }
    bg_color = color_map.get(color, "#6b7280")

    confidence_html = ""
    if confidence is not None:
        confidence_html = f'<div style="font-size: 0.9em; opacity: 0.8;">Confidence: {confidence:.0%}</div>'

    context_html = ""
    if context:
        context_html = f'<div style="margin-top: 15px; font-size: 0.9em; opacity: 0.7;">{context}</div>'

    return f"""
    <div style="
        background: linear-gradient(135deg, {bg_color}22, {bg_color}11);
        border-left: 4px solid {bg_color};
        border-radius: 12px;
        padding: 24px;
        margin: 10px 0;
    ">
        <div style="
            font-size: 1.4em;
            font-weight: 600;
            color: {bg_color};
            margin-bottom: 8px;
        ">{title}</div>
        <div style="
            font-size: 1.1em;
            color: #374151;
            line-height: 1.5;
        ">{message}</div>
        {confidence_html}
        {context_html}
    </div>
    """


# Custom CSS
CUSTOM_CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    margin-top: 5px;
    margin-bottom: 20px;
}
.location-btn {
    margin: 2px !important;
}
"""

# Build Gradio interface
with gr.Blocks(
    title="HappySardines",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
    css=CUSTOM_CSS
) as app:

    # Header
    gr.Markdown("# 🐟 HappySardines", elem_classes=["main-title"])
    gr.Markdown("*How packed are buses in Östergötland?*", elem_classes=["subtitle"])

    with gr.Row():
        # Left column: Map and location
        with gr.Column(scale=2):
            gr.Markdown("### Select Location")

            # Quick location buttons
            gr.Markdown("**Quick select:**")
            with gr.Row():
                location_buttons = []
                for name in list(PRESET_LOCATIONS.keys())[:3]:
                    btn = gr.Button(name, size="sm", elem_classes=["location-btn"])
                    location_buttons.append((name, btn))
            with gr.Row():
                for name in list(PRESET_LOCATIONS.keys())[3:]:
                    btn = gr.Button(name, size="sm", elem_classes=["location-btn"])
                    location_buttons.append((name, btn))

            # Coordinate inputs
            with gr.Row():
                lat_input = gr.Number(
                    label="Latitude",
                    value=DEFAULT_LAT,
                    precision=4,
                    minimum=BOUNDS["min_lat"],
                    maximum=BOUNDS["max_lat"]
                )
                lon_input = gr.Number(
                    label="Longitude",
                    value=DEFAULT_LON,
                    precision=4,
                    minimum=BOUNDS["min_lon"],
                    maximum=BOUNDS["max_lon"]
                )

            # Map display
            map_display = gr.HTML(value=create_map())

        # Right column: Time and predict
        with gr.Column(scale=1):
            gr.Markdown("### When?")

            date_choice = gr.Radio(
                choices=["Today", "Tomorrow"],
                value="Today",
                label="Date"
            )

            hour_slider = gr.Slider(
                minimum=5,
                maximum=23,
                value=8,
                step=1,
                label="Hour",
                info="Select time of day (24h format)"
            )

            time_display = gr.Markdown("**Selected: 08:00**")

            predict_btn = gr.Button("Predict Crowding", variant="primary", size="lg")

    # Result section
    gr.Markdown("### Prediction")
    result_display = gr.HTML(
        value=create_result_card(
            "Select location and time",
            "Then click 'Predict Crowding' to see the forecast.",
            "gray",
            None
        )
    )

    # About section
    with gr.Accordion("About this tool", open=False):
        gr.Markdown("""
        **How it works:**

        This tool predicts typical bus crowding levels based on:
        - **Location** - Different areas have different ridership patterns
        - **Time** - Rush hours vs. off-peak
        - **Day of week** - Weekdays vs. weekends
        - **Weather** - Temperature, precipitation, etc.
        - **Holidays** - Swedish red days and work-free days

        **Data sources:**
        - Historical bus occupancy data from Östgötatrafiken (GTFS-RT, Nov-Dec 2025)
        - Weather forecasts from Open-Meteo
        - Swedish holiday calendar from Svenska Dagar API

        **Limitations:**
        - Accuracy varies by location and time
        - The model predicts general area crowding, not specific bus lines

        **Built for KTH ID2223 - Scalable Machine Learning and Deep Learning**
        """)

    # Event handlers
    def update_time_display(hour):
        return f"**Selected: {int(hour):02d}:00**"

    def update_location(name):
        lat, lon = PRESET_LOCATIONS[name]
        return lat, lon, create_map(lat, lon)

    def update_map_from_coords(lat, lon):
        if lat is not None and lon is not None:
            return create_map(lat, lon)
        return create_map()

    hour_slider.change(
        fn=update_time_display,
        inputs=[hour_slider],
        outputs=[time_display]
    )

    # Connect location buttons
    for name, btn in location_buttons:
        btn.click(
            fn=lambda n=name: update_location(n),
            outputs=[lat_input, lon_input, map_display]
        )

    # Update map when coordinates change
    lat_input.change(
        fn=update_map_from_coords,
        inputs=[lat_input, lon_input],
        outputs=[map_display]
    )
    lon_input.change(
        fn=update_map_from_coords,
        inputs=[lat_input, lon_input],
        outputs=[map_display]
    )

    predict_btn.click(
        fn=make_prediction,
        inputs=[lat_input, lon_input, date_choice, hour_slider],
        outputs=[result_display]
    )


# For local testing
if __name__ == "__main__":
    app.launch()
