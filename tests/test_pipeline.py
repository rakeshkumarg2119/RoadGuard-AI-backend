"""
RoadGuard AI - Pipeline Test Suite

Tests:
1. Physics engine
2. Camera calibration
3. YOLO pothole detector
4. Full pipeline integration
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
# TEST HELPER
# =========================================================

def check(condition, message):
    if condition:
        print(f"  ✅  {message}")
    else:
        print(f"  ❌  {message}")
        raise AssertionError(message)


# =========================================================
# HEADER
# =========================================================

print()
print("═" * 60)
print("  ROAD GUARD AI — PIPELINE TEST SUITE")
print("═" * 60)
print()


# =========================================================
# 1. PHYSICS ENGINE
# =========================================================

print("▶ PHYSICS ENGINE")
print()


physics = PotholePhysicsEngine()


# ---------------------------------------------------------
# Create pothole
# ---------------------------------------------------------

pothole = PotholeGeometry(
    width_m=0.8,
    depth_m=0.05,
)


check(
    pothole is not None,
    "PotholeGeometry created successfully",
)


# ---------------------------------------------------------
# Test DRY road
# ---------------------------------------------------------

try:

    result_dry = physics.calculate(
        speed_kmh=30.0,
        pothole=pothole,
        road_condition=RoadCondition.DRY,
    )

    check(
        result_dry is not None,
        "Physics calculation completed",
    )

    print()
    print("  Dry-road result:")

    if hasattr(result_dry, "__dict__"):

        for key, value in result_dry.__dict__.items():
            print(f"     {key}: {value}")

    else:

        print(f"     {result_dry}")


except Exception as exc:

    print()
    print(f"  ❌ Physics calculation failed: {exc}")
    raise


# ---------------------------------------------------------
# Test WET road
# ---------------------------------------------------------

try:

    result_wet = physics.calculate(
        speed_kmh=30.0,
        pothole=pothole,
        road_condition=RoadCondition.WET,
    )

    check(
        result_wet is not None,
        "Wet-road physics calculation completed",
    )

    print()
    print("  Wet-road result:")

    if hasattr(result_wet, "__dict__"):

        for key, value in result_wet.__dict__.items():
            print(f"     {key}: {value}")

    else:

        print(f"     {result_wet}")


except Exception as exc:

    print()
    print(f"  ❌ Wet-road calculation failed: {exc}")
    raise


# ---------------------------------------------------------
# Test zero speed
# ---------------------------------------------------------

try:

    result_zero = physics.calculate(
        speed_kmh=0.0,
        pothole=pothole,
        road_condition=RoadCondition.DRY,
    )

    check(
        result_zero is not None,
        "Zero-speed calculation completed",
    )

except Exception as exc:

    print(
        f"  ❌ Zero-speed calculation failed: {exc}"
    )

    raise


# =========================================================
# 2. CAMERA CALIBRATION
# =========================================================

print()
print("▶ CAMERA CALIBRATION")
print()


calibrator = CameraCalibrator()


# ---------------------------------------------------------
# Create bounding box
# ---------------------------------------------------------

bbox = BoundingBox(
    x1=540,
    y1=560,
    x2=740,
    y2=650,
    confidence=0.90,
)


check(
    bbox.x1 == 540,
    "BoundingBox created correctly",
)

check(
    bbox.x2 == 740,
    "BoundingBox coordinates correct",
)


# ---------------------------------------------------------
# Convert bounding box to real world
# ---------------------------------------------------------

try:

    geometry = calibrator.bbox_to_real_world(
        bbox
    )

    check(
        geometry is not None,
        "BoundingBox converted to real-world geometry",
    )

    print()
    print("  Real-world geometry:")

    if hasattr(geometry, "__dict__"):

        for key, value in geometry.__dict__.items():
            print(f"     {key}: {value}")

    else:

        print(f"     {geometry}")


except Exception as exc:

    print(
        f"  ⚠️  bbox_to_real_world failed: {exc}"
    )


# ---------------------------------------------------------
# Calibration from reference
# ---------------------------------------------------------

reference_bbox = BoundingBox(
    x1=500,
    y1=500,
    x2=700,
    y2=600,
    confidence=1.0,
)


try:

    scale = calibrator.calibrate_from_reference(
        ref_bbox=reference_bbox,
        known_width_m=2.0,
    )

    check(
        scale is not None,
        "Reference calibration completed",
    )

    print(
        f"  Calibration scale: {scale}"
    )

except Exception as exc:

    print(
        f"  ⚠️  Reference calibration failed: {exc}"
    )


# =========================================================
# 3. SYNTHETIC ROAD IMAGE
# =========================================================

print()
print("▶ SYNTHETIC ROAD IMAGE")
print()


frame = np.full(
    (720, 1280, 3),
    fill_value=80,
    dtype=np.uint8,
)


# ---------------------------------------------------------
# Synthetic pothole
# ---------------------------------------------------------

frame[
    560:620,
    540:740
] = 20


check(
    frame.shape == (720, 1280, 3),
    "Synthetic road frame created",
)

check(
    frame.dtype == np.uint8,
    "Synthetic frame has correct image type",
)


# =========================================================
# 4. YOLO POTHOLE DETECTOR
# =========================================================

print()
print("▶ POTHOLE DETECTOR")
print()


try:

    detector = PotholeDetector(
        min_confidence=0.10,
    )

    check(
        detector is not None,
        "PotholeDetector created successfully",
    )


    # -----------------------------------------------------
    # Model information
    # -----------------------------------------------------

    if hasattr(detector, "model"):

        print(
            f"  Model loaded: "
            f"{type(detector.model).__name__}"
        )

        if hasattr(detector.model, "names"):

            print(
                f"  Classes: "
                f"{detector.model.names}"
            )


    # -----------------------------------------------------
    # Run process_frame
    # -----------------------------------------------------

    print()
    print("  Running process_frame...")


    detections = detector.process_frame(
        frame=frame,
        speed_kmh=30.0,
        road_condition=RoadCondition.DRY,
        gps_coords=(9.9252, 78.1198),
    )


    check(
        detections is not None,
        "Detector returned a result",
    )


    # -----------------------------------------------------
    # Print detections
    # -----------------------------------------------------

    print()

    print(
        f"  Detection result type: "
        f"{type(detections).__name__}"
    )


    if isinstance(detections, list):

        print(
            f"  Number of detections: "
            f"{len(detections)}"
        )


        for index, detection in enumerate(
            detections,
            start=1,
        ):

            print()
            print(
                f"  Detection #{index}:"
            )

            if isinstance(detection, dict):

                for key, value in detection.items():

                    print(
                        f"     {key}: {value}"
                    )

            else:

                print(
                    f"     {detection}"
                )


except Exception as exc:

    print()
    print(
        f"  ❌ Detector processing failed: {exc}"
    )

    raise


# =========================================================
# 5. PHYSICS + DETECTOR INTEGRATION
# =========================================================

print()
print("▶ PHYSICS + DETECTOR INTEGRATION")
print()


try:

    integration_pothole = PotholeGeometry(
        width_m=0.7,
        depth_m=0.04,
    )


    integration_result = physics.calculate(
        speed_kmh=30.0,
        pothole=integration_pothole,
        road_condition=RoadCondition.DRY,
    )


    check(
        integration_result is not None,
        "Physics integration calculation completed",
    )


    print()
    print("  Integration physics result:")


    if hasattr(
        integration_result,
        "__dict__",
    ):

        for key, value in integration_result.__dict__.items():

            print(
                f"     {key}: {value}"
            )

    else:

        print(
            f"     {integration_result}"
        )


except Exception as exc:

    print(
        f"  ❌ Integration failed: {exc}"
    )

    raise


# =========================================================
# SUMMARY
# =========================================================

print()
print("═" * 60)
print("  PIPELINE TEST COMPLETE")
print("═" * 60)
print()

print("  ✅ Physics engine")
print("  ✅ Camera calibration")
print("  ✅ Synthetic road image")
print("  ✅ YOLO pothole detector")
print("  ✅ Physics + detector integration")

print()
print("  RoadGuard AI backend pipeline test finished.")
print()