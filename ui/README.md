---
title: HappySardines
emoji: 🐟
colorFrom: blue
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
short_description: Predict bus crowding levels in Östergötland, Sweden
---

# 🐟 HappySardines

**How packed are buses in Östergötland?**

Click on the map to select a location, pick a time, and see predicted crowding levels. Toggle the heat map to see crowding patterns across the entire region.

## Features

- 🗺️ **Interactive map** - Click to select any location
- 🔥 **Heat map overlay** - See predicted crowding across the region
- 🌡️ **Real-time weather** - Forecasts from Open-Meteo
- 📅 **Holiday awareness** - Swedish red days and work-free days

## How it works

This tool predicts bus crowding levels based on:
- **Location** - Different areas have different ridership patterns
- **Time** - Rush hours vs. off-peak
- **Day of week** - Weekdays vs. weekends
- **Weather** - Temperature, precipitation, etc.
- **Holidays** - Swedish red days and work-free days

## Data sources

- Bus occupancy data from Östgötatrafiken (GTFS-RT)
- Weather forecasts from [Open-Meteo](https://open-meteo.com/)
- Swedish holiday calendar from [Svenska Dagar API](https://sholiday.faboul.se/)

## Technical details

- **Model**: XGBoost Classifier
- **Features**: Location, time, weather, holidays
- **Feature Store**: Hopsworks
- **Framework**: Streamlit + Folium

## Credits

Built for **KTH ID2223 - Scalable Machine Learning and Deep Learning**

By: Axel & Kajsa
