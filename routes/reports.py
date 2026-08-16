"""
RoadGuard AI — /reports routes

GET  /reports
    Returns all uploaded potholes ordered by upload time (newest first).
    Each document includes photo URL, lat/lon, severity, upload time,
    and whether it has been marked fixed.

PATCH /reports/{pothole_id}/fix
    Marks a pothole as fixed in MongoDB.
    Sets fixed=True, fixed_at=<UTC now>, fixed_by="community".
    Idempotent — marking an already-fixed pothole is a no-op (200 OK).

The /alert route already calls SimulationStore.get_alert_for_speed()
which queries MongoDB. Add { "fixed": False } to that geospatial filter
(see note at bottom of this file) so fixed potholes stop triggering alerts.

MongoDB collection: "potholes"  (same as SimulationStore)

Document shape after upload (existing):
{
    "_id":                 ObjectId,
    "location":            { "type": "Point", "coordinates": [lon, lat] },
    "severity":            "low" | "medium" | "high" | "critical",
    "weather_at_detection": { ... },
    "image_url":           str | None,   # set if you store images in GridFS/S3
    "created_at":          ISO-8601 str,
    "speed_simulation":    [ ... ],
    ...
}

New fields added by this route (on first fix):
    "fixed":      bool  (default False — added retroactively on fix)
    "fixed_at":   ISO-8601 str | None
    "fixed_by":   "community"
"""

import logging
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException, Path

from core.app_state import get_db

router = APIRouter()
logger = logging.getLogger("roadguard.routes.reports")

POTHOLES_COLLECTION = "potholes"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _col(db):
    return db[POTHOLES_COLLECTION]


def _serialize(doc: dict) -> dict:
    """Convert MongoDB document to JSON-safe dict for the Flutter response."""
    return {
        "pothole_id":   str(doc["_id"]),
        "lat":          doc.get("location", {}).get("coordinates", [0, 0])[1],
        "lon":          doc.get("location", {}).get("coordinates", [0, 0])[0],
        "severity":     doc.get("severity", "unknown"),
        "image_url":    doc.get("image_url"),          # None if not stored
        "created_at":   doc.get("created_at"),
        "fixed":        doc.get("fixed", False),
        "fixed_at":     doc.get("fixed_at"),
        "fall_type":    doc.get("fall_type"),
        "confidence":   doc.get("confidence"),
        "weather_condition": (
            doc.get("weather_at_detection", {}).get("road_condition")
            if isinstance(doc.get("weather_at_detection"), dict)
            else None
        ),
    }


# ── GET /reports ──────────────────────────────────────────────────────────────

@router.get("/reports")
async def get_reports(
    include_fixed: bool = True,
    limit: int = 50,
):
    """
    Return all pothole reports, newest first.

    Query params:
        include_fixed (bool, default True)  — set False to hide fixed potholes
        limit         (int, default 50)     — max documents returned
    """
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be 1–200")

    try:
        db  = get_db()
        col = _col(db)

        query = {} if include_fixed else {"fixed": {"$ne": True}}

        cursor = col.find(query).sort("created_at", -1).limit(limit)
        docs   = await cursor.to_list(length=limit)

        return {
            "status":  "ok",
            "count":   len(docs),
            "reports": [_serialize(d) for d in docs],
        }

    except Exception as exc:
        logger.error("GET /reports failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


# ── PATCH /reports/{pothole_id}/fix ───────────────────────────────────────────

@router.patch("/reports/{pothole_id}/fix")
async def mark_fixed(
    pothole_id: str = Path(..., description="MongoDB ObjectId of the pothole"),
):
    """
    Mark a pothole as fixed.
    Idempotent — calling this twice on the same pothole is safe.
    """
    try:
        oid = ObjectId(pothole_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid pothole_id format")

    try:
        db  = get_db()
        col = _col(db)

        doc = await col.find_one({"_id": oid})
        if doc is None:
            raise HTTPException(status_code=404, detail="Pothole not found")

        # Already fixed — idempotent, return current state
        if doc.get("fixed", False):
            logger.info("Pothole %s already fixed — no-op", pothole_id)
            return {
                "status":     "already_fixed",
                "pothole_id": pothole_id,
                "fixed_at":   doc.get("fixed_at"),
            }

        fixed_at = _utc_now_iso()

        await col.update_one(
            {"_id": oid},
            {
                "$set": {
                    "fixed":    True,
                    "fixed_at": fixed_at,
                    "fixed_by": "community",
                }
            },
        )

        logger.info("Pothole %s marked as fixed at %s", pothole_id, fixed_at)

        return {
            "status":     "fixed",
            "pothole_id": pothole_id,
            "fixed_at":   fixed_at,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("PATCH /reports/%s/fix failed: %s", pothole_id, exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# NOTE FOR alert.py / SimulationStore
# ─────────────────────────────────────────────────────────────────────────────
# In core/simulation_store.py, the geospatial query that powers /alert
# should exclude fixed potholes. Find the $near / $geoWithin query and
# add this filter:
#
#   existing query:
#     { "location": { "$near": { ... } } }
#
#   updated query:
#     {
#         "location": { "$near": { ... } },
#         "fixed":    { "$ne": True }        # ← ADD THIS LINE
#     }
#
# That single change ensures riders are never alerted about repaired roads.
# ─────────────────────────────────────────────────────────────────────────────
