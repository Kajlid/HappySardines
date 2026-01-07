"""
Minimal test to verify Folium GeoJSON rendering with Streamlit.
"""
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Map Test", layout="wide")
st.title("Folium GeoJSON Test")

# Create a simple map centered on Linköping
center_lat = 58.41
center_lon = 15.62

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=9,
    tiles="CartoDB positron"
)

# Add test GeoJSON - three concentric rectangles
test_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"color": "#22c55e", "fillOpacity": 0.35},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [center_lon - 0.8, center_lat - 0.5],
                    [center_lon + 0.8, center_lat - 0.5],
                    [center_lon + 0.8, center_lat + 0.5],
                    [center_lon - 0.8, center_lat + 0.5],
                    [center_lon - 0.8, center_lat - 0.5],
                ]]
            }
        },
        {
            "type": "Feature",
            "properties": {"color": "#eab308", "fillOpacity": 0.35},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [center_lon - 0.4, center_lat - 0.25],
                    [center_lon + 0.4, center_lat - 0.25],
                    [center_lon + 0.4, center_lat + 0.25],
                    [center_lon - 0.4, center_lat + 0.25],
                    [center_lon - 0.4, center_lat - 0.25],
                ]]
            }
        },
        {
            "type": "Feature",
            "properties": {"color": "#ef4444", "fillOpacity": 0.35},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [center_lon - 0.15, center_lat - 0.1],
                    [center_lon + 0.15, center_lat - 0.1],
                    [center_lon + 0.15, center_lat + 0.1],
                    [center_lon - 0.15, center_lat + 0.1],
                    [center_lon - 0.15, center_lat - 0.1],
                ]]
            }
        }
    ]
}

# Add GeoJSON layer
folium.GeoJson(
    test_geojson,
    style_function=lambda feature: {
        'fillColor': feature['properties']['color'],
        'fillOpacity': feature['properties'].get('fillOpacity', 0.35),
        'color': 'none',
        'weight': 0
    },
    name="Test Overlay"
).add_to(m)

# Display the map
st.write("You should see a map with three overlapping colored rectangles (green, yellow, red)")
map_data = st_folium(m, height=500, use_container_width=True)
st.write("Map rendered successfully!" if map_data else "Waiting for map...")
