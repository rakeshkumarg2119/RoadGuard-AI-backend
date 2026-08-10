# Road Guard AI — Backend

Pothole detection + kinematic alert system for bikes.  
Built for 48-hour hackathon. Python 3.12 · FastAPI · MongoDB · Roboflow · OpenWeatherMap

---

## What this backend does

1. **Upload flow** — rider photos a pothole → app sends image + GPS → Roboflow detects it → physics engine calculates impact → simulation table stored in MongoDB
2. **Alert flow** — while riding, Flutter polls every 1 second → backend geo-queries MongoDB → if pothole is within calculated alert distance → returns severity + sound/flashlight trigger

No map. No login. No complex UI. Just upload and alert.

---

## Project structure

```
backend/
├── core/
│   ├── physics_engine.py       # kinematic formula: d_alert = d_stop + d_react
│   ├── camera_calibration.py   # converts YOLO bbox pixels → real-world metres
│   └── pothole_detector.py     # Roboflow cloud API → calibration → physics pipeline
├── services/
│   ├── weather_service.py      # OpenWeatherMap poll every 30 min → μ dry/wet/gravel
│   └── simulation_store.py     # pre-computes speed table (10–100 km/h), MongoDB save/query
├── tests/
│   └── test_pipeline.py        # 28 tests — physics, calibration, full pipeline
├── requirements.txt
└── .env.example                # copy to .env and fill keys
```

---

## Physics — what makes this different

Most pothole apps detect and show on a map. This one calculates **when exactly to warn the rider** using real kinematics:

```
d_alert = d_stop + d_react

d_react = v × t_react          (t_react = 3 seconds)
d_stop  = v² / (2 × μ × g)

μ = 0.70  (dry road)
μ = 0.35  (wet road / rain)    ← weather service switches this automatically
μ = 0.45  (gravel)

g = 9.81 m/s²
```

Example — same 8 cm pothole, different conditions:

| Speed   | Dry road  | Wet road (rain) |
|---------|-----------|-----------------|
| 30 km/h | 30.1 m    | 35.1 m          |
| 50 km/h | 55.7 m    | 69.8 m          |
| 70 km/h | 85.9 m    | 113.4 m         |

Alert fires at the mathematically correct distance — not a fixed 50m guess.

---

## Setup

```bash
# 1. Clone and enter
cd backend

# 2. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Fill in your keys (see Keys section below)

# 5. Run tests to verify everything works
python -m pytest tests/test_pipeline.py -v
```

---

## Keys needed in .env

```env
ROBOFLOW_API_KEY=        # roboflow.com → avatar → API Keys
OPENWEATHER_API_KEY=     # openweathermap.org → API Keys (free tier)
MONGODB_URI=             # MongoDB Atlas connection string
DEFAULT_LAT=9.9252       # Madurai — change to your test location
DEFAULT_LON=78.1198
```

---

## API routes (FastAPI — to be implemented)

### POST /upload
Receives pothole photo + GPS from Flutter.

**Request (multipart/form-data):**
```
image   : file      (JPG/PNG from camera)
lat     : float     (current GPS latitude)
lon     : float     (current GPS longitude)
speed   : float     (km/h at time of upload, default 0)
```

**Response:**
```json
{
  "status": "stored",
  "pothole_id": "mongo_object_id",
  "severity": "high",
  "d_alert_m": 85.9,
  "simulation_table": {
    "30": { "d_alert_m": 30.1, "severity": "high" },
    "50": { "d_alert_m": 55.7, "severity": "critical" }
  }
}
```

---

### GET /alert
Called by Flutter every 1 second while riding.

**Request params:**
```
lat    : float   (current GPS latitude)
lon    : float   (current GPS longitude)
speed  : float   (current speed in km/h)
```

**Response — pothole found:**
```json
{
  "alert": true,
  "pothole_id": "mongo_object_id",
  "severity": "high",
  "distance_m": 72.3,
  "d_alert_m": 85.9,
  "sound": "alert",
  "flashlight": true,
  "message": "DANGER: Deep pothole ahead! Brake immediately."
}
```

**Response — no pothole nearby:**
```json
{
  "alert": false
}
```

---

## MongoDB setup (one-time)

```javascript
// Run in MongoDB Atlas shell or Compass
// Creates the 2dsphere index required for geo-queries

db.potholes.createIndex({ "gps": "2dsphere" })
```

Without this index the `/alert` geo-query will not work.

---

## Model details

- File       : `model/road_guard_pothole_best.pt`
- Framework  : Ultralytics YOLO, loaded locally (no cloud API)
- Classes    : `0 = Manhole`, `1 = Pothole` (only class 1 used)
- Confidence threshold : `0.25`

Inference runs entirely on the local `.pt` file via `core/pothole_detector.py`.

---

## Deployment (Railway — recommended)

```bash
# 1. Push backend/ to a GitHub repo
# 2. Go to railway.app → New Project → Deploy from GitHub
# 3. Add environment variables from .env in Railway dashboard
# 4. Railway auto-detects FastAPI and deploys
# 5. Get your live URL: https://roadguard-api.railway.app
```

Flutter release APK points to this URL.

---

## What is done / what is pending

### Done ✅
- Physics engine (`d_alert` formula, fall classification, severity bands)
- Camera calibration (bbox pixels → real-world metres)
- Local `.pt` detector (`road_guard_pothole_best.pt` → calibration → physics → alert)
- Alert payload carries `danger: true` for high/critical severity, for Flutter to trigger sound + flashlight
- Weather service (OpenWeatherMap, 30-min poll, μ switching)
- Simulation store (speed table pre-computation, MongoDB save, geo-query, deduplication)

### Pending — Dharunish (tonight) 🔧
- MongoDB connection setup (motor async client, Atlas URI)
- GPS alert simulation module with **real-time GPS logic commented out** for safety
- Simulate the 48.2m → 38.5m → 28.1m demo walk (stage presentation)
- Wire `SimulationStore` to actual MongoDB collection
- Wire `WeatherService` startup into app lifecycle

### Pending — tomorrow (Rakesh + Dharunish) 🔧
- FastAPI routes (`POST /upload`, `GET /alert`)
- Wire JSON response to Flutter
- End-to-end test: upload photo → alert fires on phone

---

## Demo mode (stage presentation)

For the hackathon stage demo, real GPS movement is simulated in steps:

```
Step 1: distance = 48.2m  → no alert
Step 2: distance = 38.5m  → no alert  
Step 3: distance = 28.1m  → ALERT FIRES (sound + flashlight)
```

Upload a pothole photo on stage first to seed the coordinate.
Then run the step simulation — alert fires at step 3.
Real GPS polling is commented out and replaced with this simulation for the demo.

---
