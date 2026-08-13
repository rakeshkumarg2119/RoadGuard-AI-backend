"""
RoadGuard AI — /upload route

Accepts: multipart/form-data
  - image: UploadFile  (JPEG/PNG from phone camera)
  - lat:   float
  - lon:   float
  - speed_kmh: float (optional, default 30)

Pipeline:
  image bytes → numpy array → PotholeDetector
  → SimulationStore.save() with current weather
  → response with detections + pothole_id
"""

import io
import logging
import numpy as np

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from PIL import Image

from core.app_state import get_detector, get_store, get_weather

router = APIRouter()
logger = logging.getLogger("roadguard.routes.upload")


@router.post("/upload")
async def upload_image(
    image: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    speed_kmh: float = Form(30.0),
):
    """
    Receive a photo + GPS from Flutter, run YOLO, store result.
    Returns detections list and pothole_id (or null if no pothole).
    """

    # ── Read image bytes → numpy BGR array ──────────────────
    try:
        raw = await image.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        # YOLO expects BGR (cv2 format)
        frame = np.array(pil_img)[:, :, ::-1].copy()
    except Exception as exc:
        logger.error("Image decode failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    # ── Get services ─────────────────────────────────────────
    detector = get_detector()
    store    = get_store()
    weather  = get_weather()

    current_weather = weather.get_current_sync()  # WeatherCondition or None

    # ── Road condition from weather ──────────────────────────
    from core.physics_engine import RoadCondition
    road_condition = RoadCondition.DRY
    if current_weather is not None:
        road_condition = current_weather.road_condition

    # ── Run detection ────────────────────────────────────────
    try:
        detections = detector.process_frame(
            frame=frame,
            speed_kmh=speed_kmh,
            road_condition=road_condition,
            gps_coords=(lat, lon),
        )
    except Exception as exc:
        logger.error("Detection failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Detection error: {exc}")

    # ── Save each detection to MongoDB ───────────────────────
    pothole_ids = []
    for detection in detections:
        try:
            pid = await store.save(detection, weather_condition=current_weather)
            if pid:
                pothole_ids.append(pid)
        except Exception as exc:
            logger.warning("Save failed for detection: %s", exc)

    # ── Response ─────────────────────────────────────────────
    return {
        "status":       "ok",
        "gps":          {"lat": lat, "lon": lon},
        "speed_kmh":    speed_kmh,
        "detections":   len(detections),
        "pothole_ids":  pothole_ids,
        "results": [
            {
                "pothole_id":   pothole_ids[i] if i < len(pothole_ids) else None,
                "confidence":   d["detection"]["confidence"],
                "distance_m":   d["detection"]["distance_m"],
                "severity":     d["physics"]["impact"]["severity"],
                "fall_type":    d["physics"]["impact"]["fall_type"],
                "d_alert_m":    d["physics"]["kinematics"]["d_alert_m"],
                "alert":        d["alert"],
                "inference_ms": d["inference_ms"],
            }
            for i, d in enumerate(detections)
            ],
        "weather":      current_weather.to_dict() if current_weather else None,
    }
