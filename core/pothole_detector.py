"""
Road Guard AI — Pothole Detector
Local-only mode:
  Uses local YOLO weights (best.pt if available, else yolov8n.pt fallback).
  Roboflow cloud code is kept below but disabled by default (use_cloud=False).

Your model output from screenshot:
  class_id: 1  ("pothole detection")
  x, y, width, height — center-based format from Roboflow API
"""

import os
import time
import base64
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

# ── Your Roboflow config (from screenshot) ────────────────────────────────────
ROBOFLOW_API_KEY       = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE     = "rakeshkumars-workspace"
ROBOFLOW_WORKFLOW_ID   = "road-guard-pothole-vroad-guard-pothole-1-yolov8s-t1-logic"
ROBOFLOW_API_URL       = "https://serverless.roboflow.com"

# class_id of pothole in YOUR model (from screenshot: class_id = 1)
POTHOLE_CLASS_ID       = 1
MIN_CONFIDENCE         = 0.38   # your model's current working confidence


class PotholeDetector:
    """
    Full pipeline: image → local YOLO → calibration → physics → alert payload.
    Cloud (Roboflow) is available but disabled unless use_cloud=True is passed explicitly.
    """

    def __init__(
        self,
        api_key:            Optional[str] = None,
        bike_config:        Optional[BikeConfig] = None,
        camera_calibrator:  Optional[CameraCalibrator] = None,
        min_confidence:     float = MIN_CONFIDENCE,
        use_cloud:          bool  = False,
    ):
        self.min_confidence = min_confidence
        self.physics        = PotholePhysicsEngine(bike_config or BikeConfig())
        self.calibrator     = camera_calibrator or CameraCalibrator()

        self.api_key        = api_key or ROBOFLOW_API_KEY
        self.use_cloud       = use_cloud and bool(self.api_key)

        if self.use_cloud:
            from inference_sdk import InferenceHTTPClient
            self.client = InferenceHTTPClient(
                api_url=ROBOFLOW_API_URL,
                api_key=self.api_key,
            )
            logger.info("Roboflow Cloud API ready")
        else:
            # Local-only mode
            from ultralytics import YOLO
            backend_root = Path(__file__).parent.parent
            candidates = [
                backend_root / "models" / "best.pt",
                backend_root / "best.pt",
                backend_root / "yolov8n.pt",
            ]
            model_path = next((p for p in candidates if p.exists()), None)
            model_name = str(model_path) if model_path else "yolov8n.pt"
            self.local_model = YOLO(model_name)
            logger.warning(f"Using local model: {model_name}")

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame:          np.ndarray,
        speed_kmh:      float,
        road_condition: RoadCondition = RoadCondition.DRY,
        gps_coords:     Optional[tuple] = None,
    ) -> list[dict]:
        """
        Process one video frame (BGR numpy array from cv2).
        Returns list of detection+physics payloads ready for MongoDB.
        """
        t_start = time.perf_counter()

        img_h, img_w = frame.shape[:2]
        self.calibrator.img_w = img_w
        self.calibrator.img_h = img_h

        if self.use_cloud:
            raw_predictions = self._infer_cloud(frame)
        else:
            raw_predictions = self._infer_local(frame)

        detections = []
        for pred in raw_predictions:
            conf = pred.get("confidence", 0)
            if conf < self.min_confidence:
                continue

            bbox = self._to_bbox(pred, img_w, img_h)
            detection = self._build_detection(
                bbox, conf, speed_kmh, road_condition,
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

    # ── Inference backends ────────────────────────────────────────────────────

    def _infer_cloud(self, frame: np.ndarray) -> list[dict]:
        """
        Call Roboflow Serverless API.
        Roboflow returns predictions in center-based xywh format.
        """
        import cv2
        try:
            # Encode frame as JPEG bytes → base64
            _, buf   = cv2.imencode(".jpg", frame)
            b64_img  = base64.b64encode(buf.tobytes()).decode("utf-8")

            result = self.client.run_workflow(
                workspace_name=ROBOFLOW_WORKSPACE,
                workflow_id=ROBOFLOW_WORKFLOW_ID,
                images={"image": b64_img},
                use_cache=True,
            )

            # Roboflow workflow result structure:
            # result[0]["predictions"]["predictions"] → list of prediction dicts
            preds = (
                result[0]
                .get("predictions", {})
                .get("predictions", [])
            )
            # Filter to pothole class only (class_id = 1 in your model)
            return [p for p in preds if p.get("class_id") == POTHOLE_CLASS_ID]

        except Exception as e:
            logger.error(f"Roboflow API error: {e}")
            return []

    def _infer_local(self, frame: np.ndarray) -> list[dict]:
        """Local YOLO fallback — converts ultralytics output to same dict format."""
        results = self.local_model(frame, verbose=False, conf=self.min_confidence)
        preds   = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w  = x2 - x1
                h  = y2 - y1
                preds.append({
                    "x":          x1 + w / 2,   # center x
                    "y":          y1 + h / 2,   # center y
                    "width":      w,
                    "height":     h,
                    "confidence": float(box.conf[0]),
                    "class_id":   int(box.cls[0]),
                })
        return preds

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_bbox(self, pred: dict, img_w: int, img_h: int) -> BoundingBox:
        """
        Roboflow predictions are center-based (x, y = centre of box).
        Convert to corner-based for our calibrator.
        """
        cx = pred["x"];     w = pred["width"]
        cy = pred["y"];     h = pred["height"]
        return BoundingBox(
            x1=cx - w / 2,
            y1=cy - h / 2,
            x2=cx + w / 2,
            y2=cy + h / 2,
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
        """Calibrate → physics → assemble MongoDB payload."""
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
                "source":     "roboflow_cloud" if self.use_cloud else "local_yolo",
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

        return {
            "trigger":          distance_m <= result.d_alert_m * 1.2,
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