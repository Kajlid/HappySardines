# Traffic Events Feature Analysis

## Summary

We integrated Trafikverket traffic event data into the HappySardines occupancy prediction model to test the hypothesis that road incidents (accidents, roadwork, closures) might correlate with bus crowding. After proper testing, **the feature provides no predictive value** and can be safely excluded.

## What Are Traffic Events?

### Data Source
Traffic events come from Trafikverket's public API (`api.trafikinfo.trafikverket.se`), which provides real-time and historical information about road situations in Sweden.

### Event Types

The API returns "Situations" containing "Deviations". Our dataset contains 1,237 events broken down as follows:

#### By Message Type (message_type)
| Type | Count | Description |
|------|-------|-------------|
| Trafikmeddelande | 690 | Traffic announcements (speed limits, lane closures) |
| Vägarbete | 442 | Roadwork |
| Hinder | 62 | Obstacles (frost damage, objects on road) |
| Färjor | 41 | Ferry information |
| Viktig trafikinformation | 2 | Important traffic info (severe weather warnings) |

#### By Message Code (message_code)
| Code | Count | Examples |
|------|-------|----------|
| Vägarbete | 440 | "Vägarbete, Solnavägen vid Centralvägen" |
| Hastighetsbegränsning gäller | 342 | Speed restrictions on E6, various vägar |
| Körfältsavstängningar | 219 | Lane closures |
| Viktbegränsning gäller | 64 | Weight restrictions |
| Tjälskada | 59 | Frost damage to roads |
| Vägen avstängd | 42 | Road closures |
| Färja | 41 | Ferry services (Oxdjupsleden, Hamburgsundsleden, etc.) |
| Djur på vägen | 9 | Animals on road |
| Other | ~15 | Vehicle breakdowns, ice warnings, blasting, etc. |

#### By Severity (severity_text)
| Severity | Count | Percentage |
|----------|-------|------------|
| Liten påverkan (Minor impact) | 451 | 63% |
| Stor påverkan (Major impact) | 203 | 28% |
| Mycket stor påverkan (Very major impact) | 45 | 6% |
| Ingen påverkan (No impact) | 14 | 2% |

#### Example Events
```
Vägarbete (Roadwork):
  Location: "E45 från Skarvsjöby till Fianberg i riktning mot Östersund"
  Road: E45

Trafikmeddelande (Lane closure):
  Code: "Körfältsavstängningar"
  Location: "Väg 509 från Hästbo till Trafikplats Gävle S"

Hinder (Obstacle):
  Code: "Tjälskada" (Frost damage)
  Location: "Väg 993 i Strömsunds kommun"

Viktig trafikinformation (Weather warning):
  Header: "Varning för extrema snömängder"
  Header: "Kraftigt snöfall i östra Svealand, Gävleborg och på Gotland"
```

### Data Schema (trafikverket_traffic_event_fg)
```
event_id (string, PK)
deviation_id (string, PK)
start_time (timestamp, event_time)
end_time (timestamp)
latitude (double)
longitude (double)
affected_road (string)
message_type (string)
severity_code (int)
header (string)
location_description (string)
```

## How We Used It

### Hypothesis
Road incidents near bus routes might cause:
1. Traffic congestion → buses delayed → passengers pile up at stops → crowded buses
2. Road closures → fewer cars → people take the bus instead → crowded buses

### Implementation

#### 1. Data Collection (backfill-trafikverket-data.py)
- Fetched all traffic events from 2025-11-01 to 2025-12-31
- Parsed XML responses and extracted coordinates
- Stored in Hopsworks feature group `trafikverket_traffic_event_fg` (v2)
- **Result**: 1,237 traffic events with valid coordinates

#### 2. Feature Engineering (training_pipeline.py)
For each of the 6.1 million bus trips, we calculated:
- `has_nearby_event`: Binary flag (1 if any traffic event within 500m)
- `num_nearby_traffic_events`: Count of events within 500m

#### 3. Matching Algorithm
```python
# For each trip:
1. Get trip's time (window_start) and location (lat_mean, lon_mean)
2. Filter traffic events active at that time (start_time <= trip_time <= end_time)
3. Calculate haversine distance to each active event
4. Count events within 500 meters
```

#### 4. Optimization
The naive O(n×m) algorithm (6.1M trips × 1,237 events) was too slow. We optimized by:
- Grouping trips by date (42 unique dates)
- Pre-filtering events to those active on each date
- Using vectorized numpy haversine calculation

## Results

### Match Statistics
```
Total trips:           6,151,601
Trips with nearby event:   1,460  (0.02%)
Traffic events:            1,237
```

Only **0.02%** of trips had a traffic event within 500 meters during the trip time.

### Feature Importance
```
Feature                      Importance (gain)
─────────────────────────────────────────────
is_day_before_holiday        0.1356
lat_mean                     0.1281
lon_mean                     0.1010
is_red_day                   0.0888
day_of_week                  0.0868
wind_speed_10m               0.0712
temperature_2m               0.0664
is_work_free                 0.0635
hour                         0.0633
cloud_cover                  0.0595
precipitation                0.0524
n_positions                  0.0301
max_speed                    0.0260
avg_speed                    0.0151
speed_std                    0.0120
has_nearby_event             0.0000  ← Zero
num_nearby_traffic_events    0.0000  ← Zero
```

### Model Comparison

| Model | Traffic Events | Accuracy | Class 2 Recall | Class 3 Recall |
|-------|---------------|----------|----------------|----------------|
| Without traffic | Disabled | 51.1% | 37.3% | 64.6% |
| With traffic | Enabled | 51.1% | 37.3% | 64.6% |

**Identical results.** The traffic events feature adds no predictive power.

## Why It Didn't Work

### 1. Geographic Mismatch
Trafikverket reports incidents on **roads**, not on bus routes. The bus network in Östergötland uses many streets and paths that aren't major roads tracked by Trafikverket. A traffic jam on E4 doesn't affect a city bus on Storgatan.

### 2. Sparse Coverage
With only 1,237 events across 42 days in the entire region, and buses operating on specific routes at specific times, the overlap was minimal. Only 1,460 out of 6.1 million trip records had any match.

### 3. Wrong Causality Direction
Our hypothesis assumed: road incident → traffic → crowded bus. But:
- Bus passengers don't usually switch from cars in real-time based on incidents
- Bus routes are often separated from car traffic
- Crowding is driven more by time-of-day and calendar patterns

### 4. Temporal Mismatch
Traffic events often have long durations (roadwork lasting weeks). This makes the temporal matching less meaningful—a trip "near" a 2-week roadwork event isn't necessarily affected by it.

## What Actually Predicts Crowding

The features that matter (in order of importance):
1. **Calendar features**: is_day_before_holiday, is_red_day, day_of_week, is_work_free
2. **Location**: lat_mean, lon_mean (certain routes are busier)
3. **Weather**: wind_speed, temperature, cloud_cover, precipitation
4. **Time**: hour of day
5. **Trip characteristics**: n_positions, speed metrics

## Recommendation

**Remove traffic events from the training pipeline.** The feature:
- Adds zero predictive value
- Increases training time from ~5 minutes to ~80 minutes
- Adds complexity to the codebase

The 500m radius could be adjusted, but given the fundamental geographic and temporal mismatches, it's unlikely to help.

## Lessons Learned

1. **Test hypotheses properly before optimizing.** We spent significant effort optimizing the traffic event calculation before confirming the feature was useful.

2. **Data sparsity kills features.** With only 0.02% of samples having any signal, the feature cannot contribute meaningfully to a model.

3. **Domain knowledge matters.** Looking at the actual event types reveals the mismatch:
   - **Roadwork and lane closures** affect highways (E4, E6, E45) where buses rarely operate
   - **Speed restrictions** are irrelevant to buses on fixed routes with their own schedules
   - **Ferry information** only matters for a handful of rural routes
   - **Frost damage (tjälskada)** occurs on rural roads, not urban bus networks
   - **Weight restrictions** affect trucks, not buses

   Trafikverket's data is designed for **car drivers planning road trips**, not for urban public transit. The events are concentrated on highways and rural roads, while our bus data is from Östgötatrafiken's urban/regional network. A transit-specific incident feed (delays, vehicle breakdowns, route diversions) would be far more relevant, but no such open data source exists in Sweden.

4. **Calendar and location are king.** For predicting regular patterns like bus crowding, simple temporal and spatial features dominate over external factors.

## Technical Details

### Runtime Comparison
| Configuration | Data Fetch | Traffic Calc | Training | Total |
|--------------|------------|--------------|----------|-------|
| Without traffic | 44s | 0s | ~3min | ~5min |
| With traffic | 46s | ~75min | ~3min | ~80min |

### Code Location
- Backfill script: `backfill-trafikverket-data.py`
- Feature calculation: `pipelines/training_pipeline.py` → `calculate_nearby_traffic_events()`
- Feature group: `trafikverket_traffic_event_fg` (v2)
