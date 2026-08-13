"""
RoadGuard AI — /simulate/step route

Stateless demo endpoint. Flutter passes pothole_id + step index + condition.
Backend computes that step's GPS position, speed, distance, and alert.

Steps (6 total):
  0: 80m, 10 km/h  → approaching, no alert
  1: 65m, 20 km/h  → closing in, no alert
  2: 50m, 30 km/h  → low alert zone
  3: 35m, 40 km/h  → medium alert
  4: 22m, 40 km/h  → HIGH alert
  5: 12m, 40 km/h  → CRITICAL

condition: "dry" | "rain"
"""

import math
import logging

from bson import ObjectId
from fastapi import APIRouter, Query, HTTPException

from core.app_state import get_store
from core.physics_engine import (
    PotholePhysicsEngine,
    PotholeGeometry,
    RoadCondition,
)

router = APIRouter()
logger = logging.getLogger("roadguard.routes.simulate")

engine = PotholePhysicsEngine()

# ── Step table: (distance_m, speed_kmh) ──────────────────────────────────────
STEPS = [
    (80.0, 10.0),
    (65.0, 20.0),
    (50.0, 30.0),
    (35.0, 40.0),
    (22.0, 40.0),
    (12.0, 40.0),
]

TOTAL_STEPS = len(STEPS)

CONDITION_MAP = {
    "dry":  RoadCondition.DRY,
    "rain": RoadCondition.WET,
}

SEVERITY_ORDER = ["low", "medium", "high", "critical"]


def _offset_coords(lat: float, lon: float, distance_m: float) -> tuple:
    """
    Offset lat/lon by distance_m due north.
    Simple flat-earth approximation — fine for <200m.
    """
    delta_lat = distance_m / 111_320.0
    return round(lat + delta_lat, 7), round(lon, 7)


def _alert_color(severity: str) -> str:
    return {
        "low":      "#4CAF50",
        "medium":   "#FF9800",
        "high":     "#F44336",
        "critical": "#B71C1C",
    }.get(severity, "#FF9800")


def _vibration(severity: str) -> str:
    return {
        "low":      "short",
        "medium":   "double",
        "high":     "long",
        "critical": "continuous",
    }.get(severity, "short")


def _sound(severity: str) -> str:
    return {
        "low":      "beep",
        "medium":   "warn",
        "high":     "alert",
        "critical": "siren",
    }.get(severity, "beep")


def _message(severity: str) -> str:
    return {
        "low":      "Minor pothole ahead. Reduce speed slightly.",
        "medium":   "Pothole detected! Slow down now.",
        "high":     "DANGER: Deep pothole ahead! Brake immediately.",
        "critical": "CRITICAL HAZARD! Stop if safe. Severe pothole.",
    }.get(severity, "Pothole detected. Slow down.")


@router.get("/simulate/step")
async def simulate_step(
    pothole_id: str = Query(..., description="MongoDB _id of the pothole"),
    step: int = Query(..., ge=0, lt=TOTAL_STEPS, description=f"Step index 0–{TOTAL_STEPS - 1}"),
    condition: str = Query("dry", description="Road condition: dry | rain"),
):
    """
    Returns physics + alert for one simulated approach step.
    Flutter calls this repeatedly, incrementing step each time.
    GPS coords are offset from the real pothole location — masking phone GPS.
    """

    # ── Validate condition ────────────────────────────────────────────────
    condition = condition.lower()
    if condition not in CONDITION_MAP:
        raise HTTPException(status_code=400, detail="condition must be 'dry' or 'rain'")

    road_condition = CONDITION_MAP[condition]

    # ── Fetch pothole from MongoDB ────────────────────────────────────────
    store = get_store()
    try:
        oid = ObjectId(pothole_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid pothole_id format")

    doc = await store.collection.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Pothole not found")

    # ── Pothole real coords ───────────────────────────────────────────────
    pothole_lat = doc["gps_readable"]["lat"]
    pothole_lon = doc["gps_readable"]["lon"]

    pothole_data = doc.get("pothole", {})
    pothole_geom = PotholeGeometry(
        width_m=float(pothole_data.get("width_m", 0.4)),
        depth_m=float(pothole_data.get("depth_m", 0.08)),
        confidence=float(pothole_data.get("confidence", 1.0)),
    )

    # ── Step values ───────────────────────────────────────────────────────
    distance_m, speed_kmh = STEPS[step]

    # ── Masked GPS: rider is distance_m north of pothole ─────────────────
    rider_lat, rider_lon = _offset_coords(pothole_lat, pothole_lon, distance_m)

    # ── Physics ───────────────────────────────────────────────────────────
    result = engine.calculate(
        speed_kmh=speed_kmh,
        pothole=pothole_geom,
        road_condition=road_condition,
    )

    alert_fires = distance_m <= result.d_alert_m
    severity    = result.severity if alert_fires else None

    # ── Response ──────────────────────────────────────────────────────────
    return {
        "step":           step,
        "total_steps":    TOTAL_STEPS,
        "is_last_step":   step == TOTAL_STEPS - 1,

        # Simulated rider position (masked GPS)
        "rider_gps": {
            "lat": rider_lat,
            "lon": rider_lon,
        },

        # Real pothole position
        "pothole_gps": {
            "lat": pothole_lat,
            "lon": pothole_lon,
        },

        "distance_m":   distance_m,
        "speed_kmh":    speed_kmh,
        "condition":    condition,

        # Physics
        "d_alert_m":    round(result.d_alert_m, 1),
        "d_stop_m":     round(result.d_stop_m, 1),
        "d_react_m":    round(result.d_react_m, 1),

        # Alert
        "alert": alert_fires,
        "severity":     severity,
        "color_hex":    _alert_color(severity) if alert_fires else None,
        "vibration":    _vibration(severity)   if alert_fires else None,
        "sound":        _sound(severity)       if alert_fires else None,
        "message":      _message(severity)     if alert_fires else None,

        # Pothole info
        "pothole": {
            "id":        pothole_id,
            "width_m":   pothole_geom.width_m,
            "depth_m":   pothole_geom.depth_m,
            "seen_count": doc.get("seen_count", 1),
        },
    }


@router.get("/simulate/info")
async def simulate_info():
    """
    Returns the full step table so Flutter can preview the demo script.
    """
    return {
        "total_steps": TOTAL_STEPS,
        "conditions":  ["dry", "rain"],
        "steps": [
            {
                "step":       i,
                "distance_m": d,
                "speed_kmh":  s,
            }
            for i, (d, s) in enumerate(STEPS)
        ],
    }
