# RoadGuard AI Backend — Task Accomplished

## Core Logic

- **physics_engine.py** — Calculates braking/reaction distance, impact energy, fall type, injury risk, and severity for a detected pothole. Fully working.

- **camera_calibration.py** — Converts YOLO pixel bounding box → real-world pothole size (metres) using perspective math. Working, but uses assumed camera values (needs real calibration on the actual bike mount).

- **pothole_detector.py** — Runs the local YOLO model, filters only the "Pothole" class, sends result through calibration → physics. Correctly wired to `road_guard_pothole_best.pt`.

## Data Layer

- **db.py** — Connects to MongoDB Atlas, sets up the geo index needed for location-based pothole search. Working.

- **simulation_store.py** — Saves detected potholes, avoids duplicates nearby, pre-computes alert distance for 10–100 km/h speed bands, finds nearest pothole for alerts. Most complete file in the project.

- **weather_service.py** — Polls OpenWeatherMap every 30 min, maps weather → road friction (dry/wet/gravel) for the physics engine. Working.

## App Wiring

- **main.py** — FastAPI entry point. Only has `/` and `/health` routes right now — the actual upload/alert endpoints described in the README are **not built yet**.

- **app_state.py** — Starts/stops MongoDB, weather service, and detector on app startup/shutdown. Working, but nothing calls it from an actual API route yet.

## Demo / Testing

- **gps_alert_simulator.py** — Scripted stage-demo (fake GPS steps: 48.2m → 38.5m → 28.1m) to safely show the alert firing without needing live GPS. Working standalone script.

- **test_pipeline.py** — 4 tests: physics engine, calibration, synthetic image, full pipeline. (README says "28 tests" — that's outdated/wrong.)

## Model Files

- **road_guard_pothole_best.pt** — The real trained model. 2 classes: Manhole, Pothole. This is what the app actually uses.


## Training

- **RoadGuardAI_YOLOv8_Train.ipynb** — Colab notebook to train the pothole model from scratch and export `road_guard_pothole_best.pt`. Complete and correct.

---

## Not Done Yet (biggest gaps)

1. No actual `/upload` or `/alert` API routes — the core product flow isn't exposed yet.
2. Folder structure doesn't match the imports (`core/`, `services/`, `demo/`, `model/` folders don't exist in the zip — code will fail to import as-is).
