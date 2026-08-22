"""
RoadGuard AI - API Routes Unit & Integration Tests

Tests all REST and WebSocket endpoints in `main.py` and `routes/`:
- GET /
- GET /health
- GET /api/status
- POST /upload
- GET /alert
- POST /alert/start
- POST /alert/stop
- GET /simulate/step
- GET /simulate/info
- GET /reports
- PATCH /reports/{pothole_id}/fix
- WebSocket /ws/live
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from main import app
from core.physics_engine import RoadCondition
from services.weather_service import WeatherCondition


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def create_dummy_image_bytes():
    """Helper to generate dummy JPEG bytes for testing file upload."""
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# =========================================================
# 1. BASE / HEALTH / STATUS ENDPOINTS
# =========================================================

def test_dashboard_endpoint(client):
    """Test GET / returns HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_api_status_endpoint(client):
    """Test GET /api/status returns service status."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "RoadGuard AI Backend"


def test_health_endpoint(client):
    """Test GET /health returns environment health info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "mongodb_uri_loaded" in data
    assert "openweather_key_loaded" in data


# =========================================================
# 2. UPLOAD ROUTE (/upload)
# =========================================================

@patch("routes.upload.get_weather")
@patch("routes.upload.get_store")
@patch("routes.upload.get_detector")
def test_upload_image_success(mock_get_detector, mock_get_store, mock_get_weather, client):
    """Test POST /upload with valid image and mock services."""
    # Mock detector response
    mock_detector = MagicMock()
    mock_detector.process_frame.return_value = [
        {
            "detection": {
                "bbox_px": [10, 10, 50, 50],
                "confidence": 0.85,
                "distance_m": 12.5,
                "source": "yolo",
            },
            "physics": {
                "pothole": {"width_m": 0.4, "depth_m": 0.08, "area_m2": 0.03, "confidence": 0.85},
                "speed_kmh": 30.0,
                "road_condition": "dry",
                "kinematics": {"d_react_m": 25.0, "d_stop_m": 5.1, "d_alert_m": 30.1},
                "impact": {"energy_j": 120.0, "fall_type": "jarring", "severity": "medium"},
                "damage": {"tire_bulge": 0.4},
                "injury_risk": {"loss_of_control": 0.3},
            },
            "alert": {
                "trigger": True,
                "danger": True,
                "sound_type": "warn",
                "vibration_type": "double",
                "message": "Pothole ahead",
            },
            "inference_ms": 15.0,
        }
    ]
    mock_get_detector.return_value = mock_detector

    # Mock store response
    mock_store = AsyncMock()
    mock_store.save.return_value = "507f1f77bcf86cd799439011"
    mock_get_store.return_value = mock_store

    # Mock weather response
    mock_weather = MagicMock()
    mock_weather.get_current_sync.return_value = WeatherCondition(
        road_condition=RoadCondition.DRY,
        mu=0.70,
        description="Clear",
        temp_celsius=25.0,
        humidity_pct=50,
        wind_kmh=10.0,
        weather_id=800,
    )
    mock_get_weather.return_value = mock_weather

    img_bytes = create_dummy_image_bytes()
    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"lat": "9.9252", "lon": "78.1198", "speed_kmh": "30.0"}

    response = client.post("/upload", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "ok"
    assert res_json["detections"] == 1
    assert "507f1f77bcf86cd799439011" in res_json["pothole_ids"]
    assert res_json["annotated_image_url"] is not None


def test_upload_image_invalid(client):
    """Test POST /upload with corrupt image payload."""
    files = {"image": ("test.txt", b"not an image", "text/plain")}
    data = {"lat": "9.9252", "lon": "78.1198"}

    response = client.post("/upload", files=files, data=data)
    assert response.status_code == 400
    assert "Cannot decode image" in response.json()["detail"]


# =========================================================
# 3. ALERT ROUTE (/alert)
# =========================================================

@patch("routes.alert.get_weather")
@patch("routes.alert.get_store")
def test_get_alert_hit(mock_get_store, mock_get_weather, client):
    """Test GET /alert when a pothole is nearby."""
    mock_store = AsyncMock()
    mock_store.get_alert_for_speed.return_value = {
        "pothole_id": "507f1f77bcf86cd799439011",
        "distance_m": 15.0,
        "d_alert_m": 30.0,
        "d_stop_m": 10.0,
        "zone": "stage2",
        "severity": "medium",
        "fall_type": "jarring",
        "seen_count": 2,
    }
    mock_get_store.return_value = mock_store

    mock_weather = MagicMock()
    mock_weather.get_current_sync.return_value = WeatherCondition(
        road_condition=RoadCondition.DRY,
        mu=0.70,
        description="Clear",
        temp_celsius=25.0,
        humidity_pct=50,
        wind_kmh=10.0,
        weather_id=800,
    )
    mock_get_weather.return_value = mock_weather

    response = client.get("/alert?lat=9.9252&lon=78.1198&speed_kmh=30")
    assert response.status_code == 200
    data = response.json()
    assert data["alert"] is True
    assert data["pothole_id"] == "507f1f77bcf86cd799439011"
    assert data["zone"] == "stage2"


@patch("routes.alert.get_store")
def test_get_alert_miss(mock_get_store, client):
    """Test GET /alert when no pothole is near."""
    mock_store = AsyncMock()
    mock_store.get_alert_for_speed.return_value = None
    mock_get_store.return_value = mock_store

    response = client.get("/alert?lat=9.9252&lon=78.1198&speed_kmh=30")
    assert response.status_code == 200
    data = response.json()
    assert data["alert"] is False
    assert data["message"] == "No potholes nearby"


def test_get_alert_invalid_params(client):
    """Test GET /alert with invalid coordinates or speed."""
    res1 = client.get("/alert?lat=100.0&lon=78.1198&speed_kmh=30")
    assert res1.status_code == 400

    res2 = client.get("/alert?lat=9.9252&lon=78.1198&speed_kmh=-10")
    assert res2.status_code == 400


# =========================================================
# 4. ALERT SESSION ROUTES (/alert/start & /alert/stop)
# =========================================================

@patch("routes.alert_session.get_db")
def test_alert_start_and_stop(mock_get_db, client):
    """Test POST /alert/start and POST /alert/stop."""
    mock_col = AsyncMock()
    mock_col.insert_one.return_value = MagicMock(inserted_id="507f1f77bcf86cd799439011")
    mock_col.find_one.return_value = {
        "_id": "507f1f77bcf86cd799439011",
        "started_unix": 1000.0,
    }
    mock_col.update_one.return_value = MagicMock(modified_count=1)

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    mock_get_db.return_value = mock_db

    # Test /alert/start
    start_payload = {"lat": 9.9252, "lon": 78.1198, "speed_kmh": 30.0, "weather": "dry"}
    start_res = client.post("/alert/start", json=start_payload)
    assert start_res.status_code == 200
    start_data = start_res.json()
    assert start_data["session_id"] == "507f1f77bcf86cd799439011"

    # Test /alert/stop
    stop_payload = {
        "session_id": "507f1f77bcf86cd799439011",
        "end_lat": 9.9260,
        "end_lon": 78.1200,
    }
    stop_res = client.post("/alert/stop", json=stop_payload)
    assert stop_res.status_code == 200
    stop_data = stop_res.json()
    assert stop_data["session_id"] == "507f1f77bcf86cd799439011"


def test_alert_stop_local_session(client):
    """Test POST /alert/stop gracefully handles local session IDs."""
    stop_payload = {
        "session_id": "local_123456789",
        "end_lat": 9.9260,
        "end_lon": 78.1200,
    }
    stop_res = client.post("/alert/stop", json=stop_payload)
    assert stop_res.status_code == 200
    assert stop_res.json()["session_id"] == "local_123456789"


# =========================================================
# 5. SIMULATION ROUTES (/simulate/step & /simulate/info)
# =========================================================

def test_simulate_info(client):
    """Test GET /simulate/info preview endpoint."""
    response = client.get("/simulate/info?speed_kmh=40")
    assert response.status_code == 200
    data = response.json()
    assert data["total_steps"] == 7
    assert len(data["steps"]) == 7


@patch("routes.simulate.get_store")
def test_simulate_step(mock_get_store, client):
    """Test GET /simulate/step for a valid step index."""
    mock_col = AsyncMock()
    mock_col.find_one.return_value = {
        "_id": "507f1f77bcf86cd799439011",
        "gps_readable": {"lat": 9.9252, "lon": 78.1198},
        "pothole": {"width_m": 0.5, "depth_m": 0.10, "confidence": 0.9},
        "seen_count": 3,
    }
    mock_store = MagicMock()
    mock_store.collection = mock_col
    mock_get_store.return_value = mock_store

    response = client.get("/simulate/step?pothole_id=507f1f77bcf86cd799439011&step=3&condition=dry&speed_kmh=40")
    assert response.status_code == 200
    data = response.json()
    assert data["step"] == 3
    assert data["total_steps"] == 7
    assert data["alert"] is True
    assert data["severity"] == "medium"


# =========================================================
# 6. REPORTS ROUTES (/reports & /reports/{id}/fix)
# =========================================================

@patch("routes.reports.get_db")
def test_get_reports(mock_get_db, client):
    """Test GET /reports."""
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor
    mock_cursor.to_list = AsyncMock(return_value=[
        {
            "_id": "507f1f77bcf86cd799439011",
            "location": {"coordinates": [78.1198, 9.9252]},
            "severity": "high",
            "created_at": "2025-01-01T00:00:00Z",
            "fixed": False,
        }
    ])

    mock_col = MagicMock()
    mock_col.find.return_value = mock_cursor

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    mock_get_db.return_value = mock_db

    response = client.get("/reports?include_fixed=true&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["count"] == 1
    assert data["reports"][0]["pothole_id"] == "507f1f77bcf86cd799439011"


@patch("routes.reports.get_db")
def test_mark_report_fixed(mock_get_db, client):
    """Test PATCH /reports/{pothole_id}/fix."""
    mock_col = AsyncMock()
    mock_col.find_one.return_value = {
        "_id": "507f1f77bcf86cd799439011",
        "fixed": False,
    }
    mock_col.update_one.return_value = MagicMock(modified_count=1)

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_col
    mock_get_db.return_value = mock_db

    response = client.patch("/reports/507f1f77bcf86cd799439011/fix")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "fixed"
    assert data["pothole_id"] == "507f1f77bcf86cd799439011"


# =========================================================
# 7. WEBSOCKET ROUTE (/ws/live)
# =========================================================

def test_websocket_live_feed(client):
    """Test WebSocket /ws/live connection and message exchange."""
    with client.websocket_connect("/ws/live") as websocket:
        websocket.send_text("ping")
        # Ensure connection was cleanly established and accepted
