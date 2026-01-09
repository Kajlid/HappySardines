# HappySardines

We have built a service for public transportation planners and the general public in Östergötlands län to get hourly predictions of the congestion level of all buses and trams in areas of Östergötlands län, using serverless machine learning and automated workflows.

You can interact with the service here: [Link](link). OBS LÄGG TILL!!!!!


## Motivation

Public transportation plays an important role in society's green transition, how people move and interact, and in making communities less car dependent. However, crowded public transportations lead to increased spread of infections and dissatisfaction. Furthermore, city planners benefit from knowing which bus or tram trips that have a higher or lower utilization rate, which could act as an indicator for how to set timetables or where to make infrastructure efforts. This could also ensure that timetables are created with predicted occupancy in mind from the start, since changing timetables could be a costly venture.  

Since December 2022, The Trafiklab initiative has provided occupancy data for Östgötatrafiken's vehicles in their realtime GTFS (General Transit Feed Specification) Regional API. Since then, there has been BI initiatives to visualize and forecast congestion trends, but mostly at a Proof of Concept level. This project complements that initiative with a scalable machine learning based forecasting service that could act as an example application.

This project was built by Kajsa Lidin and Axel Barck-Holst as a final project within the course ID2223 Scalable Machine Learning and Deep Learning at KTH.


## Table of Contents

- [HappySardines](#happysardines)
  - [Motivation](#motivation)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
  - [Dependencies and Setup](#dependencies-and-setup)
  - [Data and Features](#data-and-features)
    - [Koda API - bus and tram transportation and trip data](#koda-api---bus-and-tram-transportation-and-trip-data)
    - [Trafikverket - nearby traffic situations](#trafikverket---nearby-traffic-situations)
    - [Open-Meteo - Weather](#open-meteo---weather)
    - [Svenska Dagar API - Holidays](#svenska-dagar-api---holidays)
    - [Selected features and considerations](#selected-features-and-considerations)
  - [Pipelines](#pipelines)
    - [Feature Pipeline](#feature-pipeline)
    - [Training Pipeline](#training-pipeline)
    - [Inference Pipeline](#inference-pipeline)
  - [Monitoring](#monitoring)
  - [Contributing](#contributing)
  - [Contact](#contact)


## Architecture

This project relies on [Hopsworks](https://www.hopsworks.ai/) for feature storage and model registry. Data from all used APIs were ingested for historical dates stretching from 2025-11-01 and forward, but where real-time public trip features were aggregated per minute to reduce the data size in storage.

All workflow scrips (scheduled and manually run) can be found in ```.github/workflows``` and are automated with the help of GitHub Actions. The workflow scripts each run pipeline files, that can be found in ```/pipelines```. 

The machine learning model used for prediction in this project is XGBClassifer from [XGBoost](https://xgboost.readthedocs.io/en/stable/).


## Dependencies and Setup:

Dependencies used for this project can be found in requirements.txt, and can be installed via a uv or conda environment.

In order to use the uv environment, run ```uv sync``` to update the environment with the correct dependencies. 

To run this project, you need a [Hopsworks](https://www.hopsworks.ai/) account with an API key. You also need a [TrafikLab](https://developer.trafiklab.se/) API key, as well as a [Trafikverket](https://data.trafikverket.se/) API key. 

To be able to run the scripts locally you need to:

1. Create a .env file:
```
cp .env.example .env
```
2. Create the required API keys at the developer portals of each website and paste these into your .env.

 
## Data and Features

### Koda API - bus and tram transportation and trip data

From the [Trafiklab GTFS Regional API](https://www.trafiklab.se/api/our-apis/koda/koda-api-specification/#/GTFS%20historic%20data/fetchRawData), backfilled features were aggregated minute by minute before ingestion. The features from this API consist of:
- trip_id
- vehicle_id
- speed 
- n_positions 
- bearing 
- date
- day_of_week
- hour
- lat
- lon
- occupancy_status
- ... and more.

For aggregation, we aggregated timestamp to windows with start time window_start and kept min, max and mean position values (latitude and longitude). For the occupancy status, we chose the mode value (the most common label), in order to keep the discrete nature. 

For explanatory power, we also ingested data from the Koda API's static trip and stop endpoints, including:
- route_short_name
- route_long_name
- route_desc
- trip_headsign
- departure_time
- arrival_time
- location_type
- agency_name
- stop_id
- stop_headsign
- stop_name
- ... and so on.

The real-time vehicle positions follow the [General Transit Feed Specification (GTFS)](https://gtfs.org/documentation/realtime/feed-entities/vehicle-positions/). It is their convention and labels we have used when interpreting our target variable occupancy_status with the following encoding:
```
0: Empty
1: Many seats available
2: Few seats available
3: Standing room only
4: Crushed standing room only
5: Full
6: Not accepting passengers
```
However, it is important to note that the class distribution in the ingested dataset was highly imbalanced, with the majority of samples concentrated in the lowest occupancy classes (e.g., “Empty” and “Many seats available”). Moreover, labels 4–6 were entirely absent from the training data at the time of model development, but we made the decision not to remove them in case they will later appear in the ingested data. As a result of the class imbalance, despite applying class weighting to upweight underrepresented labels during training, the model remains biased toward predicting the majority classes and shows significantly higher confidence in its prediction of the these classes. 

We decided to make the tradeoff to allow a slightly lower overall accuracy of the model in favor of not completely missing the underrepresanted classes.

### Trafikverket - nearby traffic situations

From the [Trafikverket Situation API](https://data.trafikverket.se/documentation/datacache/testbench?queryTemplate=Situation%20(namespace%20road.trafficinfo.new)), some of the ingested features using the "Situation" query template with nearby traffic information were:
- deviation_id
- start_time (of the nearby traffic situation such as accident, event, weather warning etc). 
- end_time (None if situation is ongoing).
- latitude
- longitude
- affected_road
- severity_text
- location_description
- valid_until_further_notice (boolean flag)
- deleted (boolean flag)
- source

These features were later joined with the Koda transportation data to find traffic events close to current bus locations, but these cross-source features were time-intensive to compute and did not contribute much to the prediction performance. For that reason, they have been excluded from the training features and inference at the moment.

### Open-Meteo - Weather

From the [Open-Meteo API](https://open-meteo.com/en/docs/historical-weather-api), we retrieved weather features based on location information (longitude and latitude) and timestamp, both historial and forecast weather data. Here we retrieved features such as:
- temperature_2m
- relative_humidity_2m
- precipitation
- snowfall
- rain
- cloud_cover
- wind_speed_10m
- wind_gusts_10m

### Svenska Dagar API - Holidays

From the [Svenska Dagar API](https://sholiday.faboul.se/), we backfilled Swedish calendar feature data such as:
- week
- day_of_week
- is_work_free
- is_red_day
- is_day_before_holiday

### Selected features and considerations

This wide range of features with different temporal resolutions made the prediction problem and creation of the service tricky. Data from the Koda API was slow to fetch, even with aggregation, as the number of trip postions and vehicles in motion within an entire region is very large. It was also our aspiration to make this a real-time service, but the other features used did not have the same granularity as the Koda API, which made us decide to create a UI with hourly predictions. 

In this project, we mainly had to focus on data engineering and seeing how different features contributed to predicted performance. We dropped some features that did not contribute with enough predictive power, and also decided to exclude the features from Trafikverket as they were to intensive to compute despite not contributing with enough predictive power. 

![Feature importance plot from XGBoost model](./docs/feature_importance_example.png)

The final features selected for training the model was:
- **Aggregated trip/vehicle features** ("trip_id", "vehicle_id","max_speed", "n_positions", "lat_mean" "lon_mean", "hour", "day_of_week")
- **Hourly weather features** ("temperature_2m", "precipitation", "cloud_cover", "wind_speed_10m", "snowfall", "rain")
- **Holiday features** ("is_work_free", "is_red_day", "is_day_before_holiday")

With occupancy_mode being the target variable.


## Pipelines

### Feature Pipeline

The daily feature pipeline updates Feature Groups with the latest data and is run daily on schedule.

The script: 
1. Fetches real-time vehicle position data (GTFS-RT) from KODA for each hour of the previous day and parses protobuf payloads.

2. Aggregates vehicle features into 1-minute windows (speed, positions, occupancy, bearings, etc.) and validates data with Great Expectations.

3. Creates/updates the vehicle_trip_agg_fg Feature Group in Hopsworks with aggregated trip features.

4. Fetches traffic situation data from Trafikverket (Situations/Deviations API) for the previous day and parses XML geometry and metadata.

5. Creates/updates the trafikverket_traffic_event_fg Feature Group with normalized traffic event records.

6. Fetches hourly weather data and ingests it into the weather_hourly_fg Feature Group.

7. Retrieves holiday indicators for the day (work-free, red day, holiday name, etc.).

8. Builds enriched features by joining:
- trip aggregates
- nearby traffic events
- hourly weather attributes
- holiday indicators

9. Computes proximity-based traffic features (e.g., number of nearby events, severity, affected roads) using distance filters.

	
### Training Pipeline

This pipeline trains an XGBClassifier (XGBoost) to predict occupancy status (occupancy_mode) and register the model with Hopsworks.

The script:
1. Selects features for training the data and create a Feature View.
2. Splits the training data into train/test data sets based on a time-series split.
3. Trains an XGBClassifer model to predict occupancy_mode, with extra weight given to underrepresentated classes.
4. Saves the trained model and performance metrics in a Hopsworks model registry.

The performance of the model is monitored by the monitoring UI, which acts as an indicator of whether the model should be retrained (if performance is too bad).
	
### Inference Pipeline

This script downloads the trained model from Hopsworks and fills a monitoring feature group for creating a dashboard UI, including model performance metrics and predictions. This script is automatically run daily by a GitHub Actions workflow.  

## Monitoring

A monitoring UI keeping track of the model performance over time can be found [here](link). 


## Contributing

1. Clone the repository:
```
git clone https://github.com/Kajlid/HappySardines.git
cd HappySardines
```  

2. Create a branch for your changes.

3. Make a contribution, preferably using the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/) convention.

4. Push your changes to the feature branch and create a Pull Request to main.



## Contact

If you have any questions, feel free to create a new issue to this repository, or write to kajsalid@kth.se or axelbh@kth.se.


