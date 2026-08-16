"""
RoadGuard AI — /alert/start  and  /alert/stop  routes

/alert/start  — called when rider taps START.
    Creates a session document in MongoDB, returns session_id.
    Allows the backend to log ride duration, start/end coords,
    and alert counts for analytics / history.

/alert/stop   — called when rider taps STOP.
    Marks the session closed. Fire-and-forget from Flutter side.

MongoDB collection: "monitoring_sessions"

Session document shape:
{
    "_id":          ObjectId,
    "session_id":   str  (same as str(_id), echoed to client),
    "start_lat":    float,
    "start_lon":    float,
    "speed_kmh":    float,
    "weather":      str,
    "started_at":   ISO-8601 str (UTC),
    "ended_at":     ISO-8601 str | None,
    "end_lat":      float | None,
    "end_lon":      float | None,
    "duration_sec": float | None,
    "active":       bool,
}
"""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.app_state import get_db          # returns AsyncIOMotorDatabase

router  = APIRouter()
logger  = logging.getLogger("roadguard.routes.alert_session")

SESSIONS_COLLECTION = "monitoring_sessions"


# ─── Request / response models ────────────────────────────────────────────────

class StartRequest(BaseModel):
    lat:       float = Field(...,  description="GPS latitude at ride start")
    lon:       float = Field(...,  description="GPS longitude at ride start")
    speed_kmh: float = Field(30.0, description="Speed at start (km/h)")
    weather:   str   = Field("dry", description="'dry' | 'rain'")


class StartResponse(BaseModel):
    session_id: str
    started_at: str          # ISO-8601 UTC


class StopRequest(BaseModel):
    session_id: str
    end_lat:    float
    end_lon:    float


class StopResponse(BaseModel):
    session_id:   str
    duration_sec: float
    ended_at:     str        # ISO-8601 UTC


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sessions(db):
    """Return the monitoring_sessions collection."""
    return db[SESSIONS_COLLECTION]


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("/alert/start", response_model=StartResponse)
async def alert_start(body: StartRequest):
    """
    Create a monitoring session when rider taps START.

    Returns session_id so Flutter can send it back on /alert/stop.
    If MongoDB is down, raises 503 — Flutter's ApiService falls back
    to a local session automatically (see api_service.dart).
    """

    if not (-90 <= body.lat <= 90) or not (-180 <= body.lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid GPS coordinates")

    if body.speed_kmh < 0 or body.speed_kmh > 300:
        raise HTTPException(status_code=400, detail="Invalid speed_kmh")

    if body.weather not in ("dry", "rain"):
        raise HTTPException(status_code=400, detail="weather must be 'dry' or 'rain'")

    now_iso  = _utc_now_iso()
    now_unix = time.time()

    try:
        db  = get_db()
        col = _sessions(db)

        doc = {
            "start_lat":   body.lat,
            "start_lon":   body.lon,
            "speed_kmh":   body.speed_kmh,
            "weather":     body.weather,
            "started_at":  now_iso,
            "started_unix": now_unix,
            "ended_at":    None,
            "end_lat":     None,
            "end_lon":     None,
            "duration_sec": None,
            "active":      True,
        }

        result     = await col.insert_one(doc)
        session_id = str(result.inserted_id)

        logger.info("Session started: %s  lat=%.5f lon=%.5f", session_id, body.lat, body.lon)

        return StartResponse(session_id=session_id, started_at=now_iso)

    except Exception as exc:
        logger.error("alert/start failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")


@router.post("/alert/stop", response_model=StopResponse)
async def alert_stop(body: StopRequest):
    """
    Mark a monitoring session as ended when rider taps STOP.

    Flutter calls this fire-and-forget — it does NOT wait for the
    response to update its own UI state.  This endpoint always
    returns 200 even if session_id is not found (graceful degradation).
    """

    now_iso  = _utc_now_iso()
    now_unix = time.time()

    if not (-90 <= body.end_lat <= 90) or not (-180 <= body.end_lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid end GPS coordinates")

    # Local sessions (created when backend was unreachable) start with "local_"
    # — skip the DB write but still return a valid response.
    if body.session_id.startswith("local_"):
        logger.info("Local session stopped (no DB write): %s", body.session_id)
        return StopResponse(
            session_id=body.session_id,
            duration_sec=0.0,
            ended_at=now_iso,
        )

    try:
        from bson import ObjectId
        oid = ObjectId(body.session_id)
    except Exception:
        logger.warning("alert/stop: invalid session_id format '%s'", body.session_id)
        # Return gracefully — Flutter doesn't care about the error
        return StopResponse(
            session_id=body.session_id,
            duration_sec=0.0,
            ended_at=now_iso,
        )

    try:
        db  = get_db()
        col = _sessions(db)

        doc = await col.find_one({"_id": oid})

        if doc is None:
            logger.warning("alert/stop: session %s not found", body.session_id)
            return StopResponse(
                session_id=body.session_id,
                duration_sec=0.0,
                ended_at=now_iso,
            )

        started_unix = doc.get("started_unix", now_unix)
        duration_sec = round(now_unix - started_unix, 1)

        await col.update_one(
            {"_id": oid},
            {
                "$set": {
                    "active":       False,
                    "ended_at":     now_iso,
                    "end_lat":      body.end_lat,
                    "end_lon":      body.end_lon,
                    "duration_sec": duration_sec,
                }
            },
        )

        logger.info(
            "Session stopped: %s  duration=%.1fs  end=(%.5f, %.5f)",
            body.session_id, duration_sec, body.end_lat, body.end_lon,
        )

        return StopResponse(
            session_id=body.session_id,
            duration_sec=duration_sec,
            ended_at=now_iso,
        )

    except Exception as exc:
        logger.error("alert/stop DB error: %s", exc)
        # Still return 200 — Flutter doesn't retry STOP
        return StopResponse(
            session_id=body.session_id,
            duration_sec=0.0,
            ended_at=now_iso,
        )
