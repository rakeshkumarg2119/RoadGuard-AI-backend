"""
Road Guard AI — Pothole Detector
LOCAL MODEL ONLY — road_guard_pothole_best.pt (ultralytics YOLO)

Model classes (verified from .pt):
  0 = Manhole
  1 = Pothole   ← only this class is used

Pipeline: frame -> local YOLO -> calibration (px -> metres) -> physics engine
          -> alert payload (severity + danger flag for Flutter)
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from .camera_calibration import CameraCalibrator, BoundingBox
from .physics_engine import (
    PotholePhysicsEngine,
    PotholeGeometry,
    BikeConfig,
    PhysicsResult,
    RoadCondition,
)

logger = logging.getLogger(__name__)

# ── Local model config ─────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "model" / "road_guard_pothole_best.pt"
POTHOLE_CLASS_ID = 1        # "Pothole" (class 0 = "Manhole", ignored)
MIN_CONFIDENCE = 0.25       # matches sample_yolo.py working threshold

# Severities that must trigger a Flutter alert regardless of distance gate
DANGER_SEVERITIES = {"high", "critical"}


class PotholeDetector:
    """
    Full pipeline: image -> local YOLO (.pt) -> calibration -> physics -> alert payload.
    """

    def __init__(
        self,
        model_path:         Optional[str] = None,
        bike_config:        Optional[BikeConfig] = None,
        camera_calibrator:  Optional[CameraCalibrator] = None,
        min_confidence:     float = MIN_CONFIDENCE,
    ):
        from ultralytics import YOLO

        self.min_confidence = min_confidence
        self.physics        = PotholePhysicsEngine(bike_config or BikeConfig())
        self.calibrator     = camera_calibrator or CameraCalibrator()

        path = model_path or str(MODEL_PATH)
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")

        self.model = YOLO(path)
        logger.info(f"Loaded local model: {path} | classes: {self.model.names}")

    # ── Public API ───────────────────────────────────────────────────────

    def process_frame(
        self,
        frame:          np.ndarray,
        speed_kmh:      float,
        road_condition: RoadCondition = RoadCondition.DRY,
        gps_coords:     Optional[tuple] = None,
    ) -> list[dict]:
        """
        Process one frame (BGR numpy array from cv2).
        Returns list of detection+physics+alert payloads, one per pothole found.
        """
        t_start = time.perf_counter()

        img_h, img_w = frame.shape[:2]
        self.calibrator.img_w = img_w
        self.calibrator.img_h = img_h

        raw_predictions = self._infer(frame)

        detections = []
        for pred in raw_predictions:
            if pred["class_id"] != POTHOLE_CLASS_ID:
                continue
            if pred["confidence"] < self.min_confidence:
                continue

            bbox = self._to_bbox(pred)
            detection = self._build_detection(
                bbox, pred["confidence"], speed_kmh, road_condition,
                gps_coords, t_start,
            )
            detections.append(detection)

        return detections

    def process_image_file(
        self,
        image_path:     str,
        speed_kmh:      float,
        road_condition: RoadCondition = RoadCondition.DRY,
        gps_coords:     Optional[tuple] = None,
    ) -> list[dict]:
        """Run on a saved image file. Used in demo/stage mode."""
        import cv2
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        return self.process_frame(frame, speed_kmh, road_condition, gps_coords)

    # ── Inference ────────────────────────────────────────────────────────

    def _infer(self, frame: np.ndarray) -> list[dict]:
        """Runs local .pt model, returns flat prediction dicts."""
        results = self.model(frame, verbose=False, conf=self.min_confidence)
        preds = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                preds.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": float(box.conf[0]),
                    "class_id":   int(box.cls[0]),
                })
        return preds

    # ── Helpers ──────────────────────────────────────────────────────────

    def _to_bbox(self, pred: dict) -> BoundingBox:
        return BoundingBox(
            x1=pred["x1"], y1=pred["y1"],
            x2=pred["x2"], y2=pred["y2"],
            confidence=pred["confidence"],
        )

    def _build_detection(
        self,
        bbox:           BoundingBox,
        conf:           float,
        speed_kmh:      float,
        road_condition: RoadCondition,
        gps_coords:     Optional[tuple],
        t_start:        float,
    ) -> dict:
        """Calibrate -> physics -> assemble MongoDB payload."""
        width_m, depth_m, distance_m = self.calibrator.bbox_to_real_world(bbox)

        pothole = PotholeGeometry(
            width_m=width_m,
            depth_m=depth_m,
            confidence=conf,
        )

        physics_result: PhysicsResult = self.physics.calculate(
            speed_kmh=speed_kmh,
            pothole=pothole,
            road_condition=road_condition,
        )

        return {
            "timestamp":    time.time(),
            "gps":          {"lat": gps_coords[0], "lon": gps_coords[1]}
                            if gps_coords else None,
            "detection": {
                "bbox_px":    [round(bbox.x1), round(bbox.y1),
                               round(bbox.x2), round(bbox.y2)],
                "confidence": round(conf, 3),
                "distance_m": round(distance_m, 2),
                "source":     "local_yolo_pt",
            },
            "physics":      physics_result.to_dict(),
            "alert":        self._build_alert(physics_result, distance_m),
            "inference_ms": round((time.perf_counter() - t_start) * 1000, 1),
        }

    def _build_alert(self, result: PhysicsResult, distance_m: float) -> dict:
        cfg = {
            "low":      {"color": "#4CAF50", "vibration": "short",      "sound": "beep"},
            "medium":   {"color": "#FF9800", "vibration": "double",     "sound": "warn"},
            "high":     {"color": "#F44336", "vibration": "long",       "sound": "alert"},
            "critical": {"color": "#B71C1C", "vibration": "continuous", "sound": "siren"},
        }[result.severity]

        # trigger: rider is inside (or past) the calculated braking+reaction window
        within_alert_window = distance_m <= result.d_alert_m * 1.2
        # danger: always push to Flutter for high/critical severity, even outside window,
        # so the app can pre-warn / log it even if the rider is currently far off
        is_danger = result.severity in DANGER_SEVERITIES

        return {
            "trigger":          within_alert_window or is_danger,
            "danger":           is_danger,
            "severity":         result.severity,
            "color_hex":        cfg["color"],
            "vibration_type":   cfg["vibration"],
            "sound_type":       cfg["sound"],
            "message":          self._alert_message(result),
            "alert_distance_m": round(result.d_alert_m, 1),
            "pothole_ahead_m":  round(distance_m, 1),
        }

    def _alert_message(self, result: PhysicsResult) -> str:
        return {
            "low":      "Minor pothole ahead. Reduce speed slightly.",
            "medium":   "Pothole detected! Slow down now.",
            "high":     "DANGER: Deep pothole ahead! Brake immediately.",
            "critical": "CRITICAL HAZARD! Stop if safe. Severe pothole.",
        }[result.severity]