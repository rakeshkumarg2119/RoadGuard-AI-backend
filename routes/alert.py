"""
RoadGuard AI — /alert route

Called by Flutter background service every ~2 seconds while riding.

Query params:
  lat:       float  — current GPS latitude
  lon:       float  — current GPS longitude
  speed_kmh: float  — current speed from phone

Returns:
  alert payload from SimulationStore (pre-computed physics)
  or {"alert": false} if no pothole within 200m
"""

import logging
from fastapi import APIRouter, Query, HTTPException
from core.app_state import get_store

router = APIRouter()
logger = logging.getLogger("roadguard.routes.alert")


@router.get("/alert")
async def get_alert(
    lat: float = Query(..., description="Current GPS latitude"),
    lon: float = Query(..., description="Current GPS longitude"),
    speed_kmh: float = Query(30.0, description="Current speed in km/h"),
):
    """
    Check if rider is approaching a known pothole.
    Flutter background service polls this every ~2s.
    """

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid GPS coordinates")

    if speed_kmh < 0 or speed_kmh > 300:
        raise HTTPException(status_code=400, detail="Invalid speed value")

    store = get_store()

    try:
        result = await store.get_alert_for_speed(
            lat=lat,
            lon=lon,
            speed_kmh=speed_kmh,
        )
    except Exception as exc:
        logger.error("Alert lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Alert lookup error: {exc}")

    if result is None:
        return {
            "alert":   False,
            "message": "No potholes nearby",
            "gps":     {"lat": lat, "lon": lon},
        }

    return {
        "alert":            True,
        "gps":              {"lat": lat, "lon": lon},
        "speed_kmh":        speed_kmh,
        "pothole_id":       result.get("pothole_id"),
        "distance_m":       result.get("distance_m"),
        "severity":         result.get("severity"),
        "color_hex":        result.get("color_hex"),
        "vibration":        result.get("vibration"),
        "sound":            result.get("sound"),
        "message":          result.get("message"),
        "alert_distance_m": result.get("alert_distance_m"),
        "d_stop_m":         result.get("d_stop_m"),
        "fall_type":        result.get("fall_type"),
        "seen_count":       result.get("seen_count", 1),
    }
