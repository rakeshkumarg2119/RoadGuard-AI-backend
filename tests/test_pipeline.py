"""
RoadGuard AI - Pipeline Test Suite

Tests:
1. Physics engine
2. Camera calibration
3. Synthetic road image
4. YOLO pothole detector
5. Full pipeline integration
"""

import sys
from pathlib import Path

import numpy as np


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORTS
# =========================================================

from core.physics_engine import (
    PotholePhysicsEngine,
    PotholeGeometry,
    RoadCondition,
)

from core.camera_calibration import (
    CameraCalibrator,
    BoundingBox,
)

from core.pothole_detector import (
    PotholeDetector,
)


# =========================================================
# HELPER
# =========================================================

def create_synthetic_road_image():
    """Create a synthetic road image containing a dark pothole."""

    frame = np.full(
        (720, 1280, 3),
        fill_value=80,
        dtype=np.uint8,
    )

    # Synthetic pothole
    frame[560:620, 540:740] = 20

    return frame


# =========================================================
# 1. PHYSICS ENGINE
# =========================================================

def test_physics_engine():
    """Test basic pothole physics calculation."""

    engine = PotholePhysicsEngine()

    geometry = PotholeGeometry(
        width_m=0.40,
        depth_m=0.08,
    )

    result = engine.calculate(
        speed_kmh=30.0,
        pothole=geometry,
        road_condition=RoadCondition.DRY,
    )

    assert result is not None

    print()
    print("Physics result:")
    print(result)


# =========================================================
# 2. CAMERA CALIBRATION
# =========================================================

def test_camera_calibration():
    """Test camera calibration using a known reference object."""

    calibrator = CameraCalibrator()

    reference_bbox = BoundingBox(
        x1=500,
        y1=500,
        x2=700,
        y2=600,
        confidence=1.0,
    )

    scale = calibrator.calibrate_from_reference(
        ref_bbox=reference_bbox,
        known_width_m=2.0,
    )

    assert scale is not None
    assert scale > 0

    print()
    print(f"Calibration scale: {scale}")


# =========================================================
# 3. SYNTHETIC ROAD IMAGE
# =========================================================

def test_synthetic_road_image():
    """Test synthetic road image creation."""

    frame = create_synthetic_road_image()

    assert frame.shape == (720, 1280, 3)
    assert frame.dtype == np.uint8

    print()
    print("Synthetic road frame created successfully.")


# =========================================================
# 4. YOLO POTHOLE DETECTOR
# =========================================================

def test_pothole_detector():
    """Test the YOLO pothole detector."""

    detector = PotholeDetector(
        min_confidence=0.10,
    )

    assert detector is not None

    frame = create_synthetic_road_image()

    detections = detector.process_frame(
        frame=frame,
        speed_kmh=30.0,
        road_condition=RoadCondition.DRY,
        gps_coords=(9.9252, 78.1198),
    )

    assert detections is not None

    print()
    print(
        f"Detection result type: "
        f"{type(detections).__name__}"
    )

    if isinstance(detections, list):

        print(
            f"Number of detections: "
            f"{len(detections)}"
        )

        for index, detection in enumerate(
            detections,
            start=1,
        ):
            print()
            print(f"Detection #{index}:")
            print(detection)


# =========================================================
# 5. FULL PIPELINE INTEGRATION
# =========================================================

def test_full_pipeline():
    """Test the complete RoadGuard detection pipeline."""

    frame = create_synthetic_road_image()

    detector = PotholeDetector(
        min_confidence=0.10,
    )

    detections = detector.process_frame(
        frame=frame,
        speed_kmh=30.0,
        road_condition=RoadCondition.DRY,
        gps_coords=(9.9252, 78.1198),
    )

    assert detections is not None

    print()
    print(
        "Full RoadGuard pipeline "
        "executed successfully."
    )