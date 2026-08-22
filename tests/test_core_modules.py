"""
RoadGuard AI - Core Modules Unit Test Suite

Tests core logic in:
- physics_engine.py
- camera_calibration.py
- pothole_detector.py
- simulation_store.py
- weather_service.py
"""

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

from core.physics_engine import (
    PotholePhysicsEngine,
    PotholeGeometry,
    BikeConfig,
    RoadCondition,
    FallType,
)
from core.camera_calibration import CameraCalibrator, BoundingBox
from core.pothole_detector import PotholeDetector
from services.simulation_store import SimulationStore, SPEED_BANDS
from services.weather_service import WeatherService, WeatherCondition


# =========================================================
# 1. PHYSICS ENGINE TESTS
# =========================================================

def test_physics_calculations_dry():
    """Test physics calculations for dry road condition."""
    engine = PotholePhysicsEngine(BikeConfig(rider_mass_kg=75.0, bike_mass_kg=110.0))
    pothole = PotholeGeometry(width_m=0.50, depth_m=0.10)

    result = engine.calculate(
        speed_kmh=50.0,
        pothole=pothole,
        road_condition=RoadCondition.DRY,
    )

    # Validate kinematics
    # speed_ms = 50 / 3.6 = 13.888 m/s
    # d_react = 13.888 * 3.0 = 41.666 m
    # d_stop = (13.888^2) / (2 * 0.7 * 9.81) = 192.9 / 13.734 = 14.04 m
    # d_alert = 41.666 + 14.04 = 55.71 m
    assert pytest.approx(result.d_react_m, 0.1) == 41.67
    assert pytest.approx(result.d_stop_m, 0.1) == 14.04
    assert pytest.approx(result.d_alert_m, 0.1) == 55.71

    # Validate fall type
    assert result.fall_type in [FallType.SAFE, FallType.CONTROLLED, FallType.SIDE_SLIDE, FallType.OVER_BARS]


def test_physics_calculations_wet_and_gravel():
    """Test braking distance increases on wet/gravel roads."""
    engine = PotholePhysicsEngine()
    pothole = PotholeGeometry(width_m=0.30, depth_m=0.05)

    dry_res = engine.calculate(30.0, pothole, RoadCondition.DRY)
    wet_res = engine.calculate(30.0, pothole, RoadCondition.WET)
    gravel_res = engine.calculate(30.0, pothole, RoadCondition.GRAVEL)

    # Wet braking distance should be longer than dry
    assert wet_res.d_stop_m > dry_res.d_stop_m
    assert wet_res.d_alert_m > dry_res.d_alert_m

    # Gravel braking distance should be between dry and wet
    assert dry_res.d_stop_m < gravel_res.d_stop_m < wet_res.d_stop_m


def test_physics_pothole_area_and_dict():
    """Test PotholeGeometry properties and PhysicsResult.to_dict()."""
    pothole = PotholeGeometry(width_m=0.40, depth_m=0.08, confidence=0.95)
    expected_area = math.pi * (0.40 / 2) * (0.08 / 2)
    assert pytest.approx(pothole.area_m2, 0.001) == expected_area

    engine = PotholePhysicsEngine()
    res = engine.calculate(40.0, pothole, RoadCondition.DRY)
    res_dict = res.to_dict()

    assert "pothole" in res_dict
    assert "kinematics" in res_dict
    assert "impact" in res_dict
    assert "damage" in res_dict
    assert res_dict["pothole"]["width_m"] == 0.40


# =========================================================
# 2. CAMERA CALIBRATION TESTS
# =========================================================

def test_camera_calibrator_properties():
    """Test CameraCalibrator default pixel-to-meter estimations."""
    calibrator = CameraCalibrator()

    # Bounding box 100x50 at bottom of 720p frame
    bbox = BoundingBox(x1=500, y1=600, x2=600, y2=650, confidence=0.9)
    width_m, depth_m, distance_m = calibrator.bbox_to_real_world(bbox)

    assert width_m > 0
    assert depth_m > 0
    assert distance_m > 0


def test_camera_calibrator_reference_calibration():
    """Test custom reference calibration scale calculation."""
    calibrator = CameraCalibrator()
    bbox = BoundingBox(x1=100, y1=100, x2=300, y2=200, confidence=1.0)
    focal_length = calibrator.calibrate_from_reference(ref_bbox=bbox, known_width_m=2.0)

    # distance_m = 1.1 / tan(...) = ~12.78m
    # focal_length = (200 px * 12.78m) / 2.0m = ~1278 px
    assert focal_length > 0
    assert calibrator.focal_length_px == focal_length


# =========================================================
# 3. POTHOLE DETECTOR TESTS
# =========================================================

def test_pothole_detector_processing():
    """Test processing image frame with PotholeDetector."""
    detector = PotholeDetector(min_confidence=0.10)
    frame = np.full((720, 1280, 3), fill_value=80, dtype=np.uint8)
    frame[500:600, 500:700] = 20  # Synthetic dark spot

    detections = detector.process_frame(
        frame=frame,
        speed_kmh=30.0,
        road_condition=RoadCondition.DRY,
        gps_coords=(9.9252, 78.1198),
    )

    assert isinstance(detections, list)
    for det in detections:
        assert "detection" in det
        assert "physics" in det
        assert "alert" in det


# =========================================================
# 4. SIMULATION STORE TESTS
# =========================================================

@pytest.mark.anyio
async def test_simulation_store_save_and_lookup():
    """Test SimulationStore save and alert lookup with mock collection."""
    mock_collection = MagicMock()

    # Mock no existing nearby pothole found
    mock_collection.find_one = AsyncMock(return_value=None)

    inserted_id = "507f1f77bcf86cd799439011"
    insert_result = MagicMock()
    insert_result.inserted_id = inserted_id
    mock_collection.insert_one = AsyncMock(return_value=insert_result)

    store = SimulationStore(collection=mock_collection)

    payload = {
        "gps": {"lat": 9.9252, "lon": 78.1198},
        "detection": {"bbox_px": [10, 10, 50, 50], "confidence": 0.9, "distance_m": 10.0},
        "physics": {
            "pothole": {"width_m": 0.4, "depth_m": 0.08, "confidence": 0.9},
        },
    }

    # Test save
    res_id = await store.save(payload)
    assert res_id == inserted_id
    assert mock_collection.insert_one.called

    # Test get_alert_for_speed
    mock_doc = {
        "_id": inserted_id,
        "gps_readable": {"lat": 9.9252, "lon": 78.1198},
        "seen_count": 1,
        "speed_simulation": {
            "30": {
                "d_alert_m": 35.0,
                "d_stop_m": 10.0,
                "d_react_m": 25.0,
                "severity": "medium",
                "fall_type": "controlled",
            }
        },
    }

    # Async generator for cursor iteration
    async def async_generator(items):
        """Yield each item from the provided iterable in order.
        
        Parameters:
            items: The iterable whose items are yielded.
        
        Yields:
            Each item from `items`.
        """
        for item in items:
            yield item

    mock_cursor = MagicMock()
    mock_cursor.__aiter__.side_effect = lambda: async_generator([mock_doc])
    mock_cursor.limit.return_value = mock_cursor
    mock_collection.find.return_value = mock_cursor

    # Distance between identical points is 0m, which is <= d_stop_m (10m).
    # get_alert_for_speed upgrades severity by 1 step in stage 3 (distance <= d_stop_m):
    # base "medium" -> "high".
    alert_res = await store.get_alert_for_speed(lat=9.9252, lon=78.1198, speed_kmh=30.0)
    assert alert_res is not None
    assert alert_res["pothole_id"] == inserted_id
    assert alert_res["severity"] == "high"


def test_simulation_store_haversine_and_speed_band():
    """Test haversine distance calculation and speed band matching."""
    # Distance between same points = 0
    dist = SimulationStore._haversine(9.9252, 78.1198, 9.9252, 78.1198)
    assert dist == 0.0

    # Distance between ~111km apart lat points
    dist_1deg = SimulationStore._haversine(0.0, 0.0, 1.0, 0.0)
    assert pytest.approx(dist_1deg, 1000) == 111195

    # Speed band matching
    assert SimulationStore._speed_band(28.0) == 30
    assert SimulationStore._speed_band(12.0) == 10
    assert SimulationStore._speed_band(95.0) == 90


# =========================================================
# 5. WEATHER SERVICE TESTS
# =========================================================

def test_weather_condition_dataclass():
    """Test WeatherCondition properties and serialization."""
    cond = WeatherCondition(
        road_condition=RoadCondition.WET,
        mu=0.35,
        description="Rain",
        temp_celsius=22.0,
        humidity_pct=85,
        wind_kmh=15.0,
        weather_id=500,  # Rain ID
    )

    assert cond.is_rain is True
    assert cond.is_storm is False
    assert "WET ROAD" in cond.alert_suffix
    assert cond.to_dict()["road_condition"] == "wet"


@pytest.mark.anyio
async def test_weather_service_dry_fallback():
    """Test WeatherService fallback behavior when no API key is set."""
    service = WeatherService(lat=9.9252, lon=78.1198, api_key="")
    cond = await service.get_current()

    assert cond is not None
    assert cond.road_condition == RoadCondition.DRY
    assert cond.mu == 0.70
    assert "Fallback" in cond.description
