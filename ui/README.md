---
title: HappySardines
emoji: 🐟
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Predict bus crowding levels in Östergötland, Sweden
---

# 🐟 HappySardines

**How packed are buses in Östergötland?**

Drop a pin on the map, pick a time, and find out how crowded buses typically are in that area. Built with ML using historical transit data from Östgötatrafiken.

## How it works

This tool predicts typical bus crowding levels based on:
- **Location** - Different areas have different ridership patterns
- **Time** - Rush hours vs. off-peak
- **Day of week** - Weekdays vs. weekends
- **Weather** - Temperature, precipitation, etc.
- **Holidays** - Swedish red days and work-free days

## Data sources

- Historical bus occupancy data from Östgötatrafiken (GTFS-RT, Nov-Dec 2025)
- Weather forecasts from [Open-Meteo](https://open-meteo.com/)
- Swedish holiday calendar from [Svenska Dagar API](https://sholiday.faboul.se/)

## Limitations

- Predictions are based on historical patterns, not real-time data
- Accuracy varies by location and time
- The model predicts general area crowding, not specific bus lines

## Technical details

- **Model**: XGBoost Classifier trained on ~6M trip records
- **Features**: Location, time, weather, holidays
- **Feature Store**: Hopsworks
- **Framework**: Gradio

## Credits

Built for **KTH ID2223 - Scalable Machine Learning and Deep Learning**

By: Axel & Kajsa
