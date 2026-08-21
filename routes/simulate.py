"""
RoadGuard AI — /simulate/step route

Stateless demo endpoint. Flutter passes pothole_id + step index + condition.
Backend computes that step's GPS position, speed, distance, and alert.

Steps (7 total, ~70 sec at 10s/step — matches the Flutter demo timeline):
  0: 90m,  0 km/h  → start, no alert
  1: 70m, 12 km/h  → accelerating, no alert
  2: 55m, 25 km/h  → ramping up, low alert
  3: 40m, 40 km/h  → peak speed (mid-route), medium alert
  4: 28m, 28 km/h  → braking begins, medium alert
  5: 16m, 15 km/h  → HIGH alert
  6:  6m,  5 km/h  → CRITICAL, near-stop, siren (closest approach)

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
# Speed ramps up then brakes down toward the hazard — mimics a real
# rider accelerating, cruising, then braking as the pothole gets close.
STEPS = [
    (90.0, 0.0),
    (70.0, 12.0),
    (55.0, 25.0),
    (40.0, 40.0),
    (28.0, 28.0),
    (16.0, 15.0),
    (6.0, 5.0),
]

TOTAL_STEPS = len(STEPS)

# The peak speed baked into STEPS above (step 3 = 40 km/h). User-selected
# speed_kmh is scaled relative to this so the whole curve (accelerate ->
# peak -> brake) stretches/shrinks proportionally instead of being ignored.
PEAK_SPEED_IN_TABLE = 40.0

# Severity is forced by step index rather than derived from the physics
# engine's d_alert_m threshold. The physics threshold scales with speed,
# so a fast mid-route step (e.g. step 3 at 40 km/h) could otherwise fire
# CRITICAL well before the rider is actually close — out of sync with
# the Flutter app's stage1/stage2/stage3 curve. Forcing it by step keeps
# backend severity and client alert stage locked together.
SEVERITY_BY_STEP = {
    0: None,
    1: None,
    2: "low",
    3: "medium",
    4: "medium",
    5: "high",
    6: "critical",
}

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
    speed_kmh: float = Query(
        PEAK_SPEED_IN_TABLE, gt=0,
        description="User-selected peak speed (km/h) — scales the step curve",
    ),
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
    # distance_m and the table's base speed stay tied to `step` (this is
    # what keeps severity/alert-stage timing locked to the Flutter curve —
    # see SEVERITY_BY_STEP note above). Only the speed value itself is
    # scaled to reflect what the user actually picked.
    distance_m, base_speed_kmh = STEPS[step]

    scale = speed_kmh / PEAK_SPEED_IN_TABLE
    speed_kmh_actual = round(base_speed_kmh * scale, 1)

    # ── Masked GPS: rider is distance_m north of pothole ─────────────────
    rider_lat, rider_lon = _offset_coords(pothole_lat, pothole_lon, distance_m)

    # ── Physics (kept for reference / display only — see note above) ─────
    result = engine.calculate(
        speed_kmh=speed_kmh_actual,
        pothole=pothole_geom,
        road_condition=road_condition,
    )

    # ── Alert: forced by step index, not physics threshold ───────────────
    severity    = SEVERITY_BY_STEP.get(step)
    alert_fires = severity is not None

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
        "speed_kmh":    speed_kmh_actual,
        "condition":    condition,

        # Physics (reference values only — not used to trigger alert_fires)
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
async def simulate_info(
    speed_kmh: float = Query(
        PEAK_SPEED_IN_TABLE, gt=0,
        description="User-selected peak speed (km/h) — scales the preview curve",
    ),
):
    """
    Returns the full step table so Flutter can preview the demo script.
    Scaled by speed_kmh so the preview matches what /simulate/step will
    actually return for that speed (see PEAK_SPEED_IN_TABLE note above).
    """
    scale = speed_kmh / PEAK_SPEED_IN_TABLE
    return {
        "total_steps": TOTAL_STEPS,
        "conditions":  ["dry", "rain"],
        "steps": [
            {
                "step":       i,
                "distance_m": d,
                "speed_kmh":  round(s * scale, 1),
                "severity":   SEVERITY_BY_STEP.get(i),
            }
            for i, (d, s) in enumerate(STEPS)
        ],
    }