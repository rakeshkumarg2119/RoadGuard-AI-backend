<div align="center">
  <img src="road-guard-ai.png" alt="Road Guard AI Logo" width="500" height="500"/>
  <h1>Road Guard AI - Backend 🧠</h1>
  <p><em>Kinematic Alert System & AI Pothole Detection Engine</em></p>
</div>

---

## 🌟 Overview

The **Road Guard AI Backend** powers the logic behind the app. It processes user-uploaded photos through a YOLOv8 AI model, calculates real-world pothole dimensions, and uses a custom physics engine to determine the exact distance a rider needs to brake safely, adjusting for current weather conditions.

*Note: We are currently using **ngrok** to tunnel the local server and connect it with the frontend.*

---

## ✨ Features

- **🤖 AI Detection (YOLOv8):** Analyzes uploaded images to detect potholes and manholes.
- **📏 Camera Calibration:** Converts bounding box pixels from the AI into real-world meters.
- **🧮 Kinematic Physics Engine:** Calculates the alert distance (`d_alert = d_stop + d_react`) based on the rider's speed and road friction (dry, wet, gravel).
- **⛅ Weather Integration:** Polls OpenWeatherMap to automatically adjust the road friction coefficient.
- **🗺️ Geospatial Queries:** Uses MongoDB's `2dsphere` indexes to instantly find hazards within the rider's path.
- **📡 Live Dashboard:** WebSocket-powered live feed for monitoring system health and alerts in real-time.

---

## 🛠️ Tech Stack

- **Core Framework:** Python 3.12, FastAPI, Uvicorn
- **AI & Vision:** Ultralytics (YOLOv8), OpenCV, Pillow
- **Database:** MongoDB (Motor/PyMongo) for geospatial storage
- **Integrations:** OpenWeatherMap API, `aiohttp`
- **Connectivity:** **ngrok** (for local-to-frontend tunneling)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12
- MongoDB Atlas (or local MongoDB with 2dsphere support)
- OpenWeatherMap API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rakeshkumarg2119/RoadGuard-AI-backend.git
   cd RoadGuard-AI-backend
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file based on `.env.example`:
   ```env
   OPENWEATHER_API_KEY=your_key_here
   MONGODB_URI=your_mongodb_connection_string
   DEFAULT_LAT=9.9252
   DEFAULT_LON=78.1198
   ```

5. **MongoDB Setup (Crucial):**
   Run this in your MongoDB shell to enable geospatial queries:
   ```javascript
   db.potholes.createIndex({ "gps": "2dsphere" })
   ```

6. **Start the server:**
   ```bash
   python main.py
   # Or using uvicorn directly:
   # uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

7. **Connect Frontend (ngrok):**
   In a separate terminal, expose port 8000 using ngrok:
   ```bash
   ngrok http 8000
   ```
   *Copy the generated forwarding URL and use it in the Flutter app.*

---

## 🧮 Physics Breakdown

Unlike apps that use a fixed 50m guess, we calculate exactly when to warn the rider:

```text
d_alert = d_stop + d_react

d_react = v × t_react          (t_react = 3 seconds)
d_stop  = v² / (2 × μ × g)

μ = 0.70  (dry road)
μ = 0.35  (wet road / rain)    
μ = 0.45  (gravel)

g = 9.81 m/s²
```

**Example:** For an 8cm deep pothole:
| Speed   | Dry road  | Wet road (rain) |
|---------|-----------|-----------------|
| 30 km/h | 30.1 m    | 35.1 m          |
| 50 km/h | 55.7 m    | 69.8 m          |

---

## 🧪 Testing & Simulation

We include a robust suite of tests and a simulation mode for demonstrations without real GPS movement. 
To run the test suite:
```bash
python -m pytest tests/test_pipeline.py -v
```

---
<p align="center">Powering safer rides with AI and Physics. ⚙️</p>
