"""
Road Guard AI — Simulation Store
Pre-computes speed-band simulation table for each detected pothole
and stores it in MongoDB alongside the pothole record.

At detection time → store pothole + compute speed table once.
At alert time     → look up speed band → instant response, no recalculation.

MongoDB document structure:
{
    "_id": ObjectId,
    "gps": {"lat": 9.9252, "lon": 78.1198},
    "pothole": { width_m, depth_m, confidence },
    "detection": { ... },
    "speed_simulation": {
        "10":  { d_alert_m, severity, fall_type, damage_summary },
        "20":  { ... },
        "30":  { ... },
        ...
        "100": { ... }
    },
    "weather_at_detection": { road_condition, mu, ... },
    "created_at": timestamp,
    "last_seen_at": timestamp,
    "seen_count": 1,
}
"""

import time
import logging
from typing import Optional

from ..core.physics_engine import (
    PotholePhysicsEngine,
    PotholeGeometry,
    BikeConfig,
    RoadCondition,
)

logger = logging.getLogger(__name__)

# Speed bands to pre-compute (km/h)
SPEED_BANDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# MongoDB geo-query radius for deduplication (metres)
# If a pothole is detected within this radius of an existing one → update, not insert
DEDUP_RADIUS_M = 15.0


class SimulationStore:
    """
    Manages pothole records in MongoDB with pre-computed speed simulations.
    Requires a MongoDB collection passed in at init (motor async client).

    Usage (inside FastAPI):
        store = SimulationStore(db["potholes"])
        doc_id = await store.save(detection_payload, weather_condition)
    """

    def __init__(self, collection, bike_config: Optional[BikeConfig] = None):
        self.collection = collection
        self.engine     = PotholePhysicsEngine(bike_config or BikeConfig())

    # ── Public API ────────────────────────────────────────────────────────────

    async def save(
        self,
        detection_payload: dict,
        weather_condition=None,       # WeatherCondition object or None
    ) -> str:
        """
        Save/update a pothole record with full speed simulation table.
        Returns the MongoDB document _id as string.
        """
        gps = detection_payload.get("gps")
        if not gps:
            logger.warning("No GPS in payload — skipping store")
            return None

        pothole_data = detection_payload["physics"]["pothole"]
        pothole = PotholeGeometry(
            width_m    = pothole_data["width_m"],
            depth_m    = pothole_data["depth_m"],
            confidence = pothole_data["confidence"],
        )

        road_condition = RoadCondition.DRY
        if weather_condition:
            road_condition = weather_condition.road_condition

        sim_table = self._build_simulation_table(pothole, road_condition)

        # Check if pothole already exists nearby (deduplication)
        existing = await self._find_nearby(gps["lat"], gps["lon"])

        weather_dict = weather_condition.to_dict() if weather_condition else None

        if existing:
            # Update existing record
            result = await self.collection.update_one(
                {"_id": existing["_id"]},
                {
                    "$set": {
                        "last_seen_at":       time.time(),
                        "speed_simulation":   sim_table,
                        "weather_at_detection": weather_dict,
                        "detection":          detection_payload["detection"],
                    },
                    "$inc": {"seen_count": 1},
                }
            )
            logger.info(f"Updated pothole {existing['_id']} (seen {existing['seen_count']+1}x)")
            return str(existing["_id"])

        else:
            # Insert new pothole record
            document = {
                "gps": {
                    "type":        "Point",
                    "coordinates": [gps["lon"], gps["lat"]],   # GeoJSON: [lon, lat]
                },
                "gps_readable": {
                    "lat": gps["lat"],
                    "lon": gps["lon"],
                },
                "pothole":              pothole_data,
                "detection":            detection_payload["detection"],
                "speed_simulation":     sim_table,
                "weather_at_detection": weather_dict,
                "created_at":           time.time(),
                "last_seen_at":         time.time(),
                "seen_count":           1,
                "verified":             False,   # set True after manual confirmation
            }
            result = await self.collection.insert_one(document)
            logger.info(f"New pothole stored: {result.inserted_id}")
            return str(result.inserted_id)

    async def get_alert_for_speed(
        self,
        lat:       float,
        lon:       float,
        speed_kmh: float,
    ) -> Optional[dict]:
        """
        Geo-query: find nearest pothole within alert distance for given speed.
        Returns alert payload or None.

        Called by Flutter every ~1 second via GET /alert?lat=&lon=&speed=
        """
        # Find the speed band
        band = self._speed_band(speed_kmh)

        # MongoDB $nearSphere query — finds potholes sorted by distance
        # Requires 2dsphere index on 'gps' field (Mithun creates this once)
        cursor = self.collection.find(
            {
                "gps": {
                    "$nearSphere": {
                        "$geometry": {
                            "type":        "Point",
                            "coordinates": [lon, lat],
                        },
                        "$maxDistance": 200,   # only look within 200m
                    }
                }
            }
        ).limit(3)   # nearest 3 potholes

        async for doc in cursor:
            sim = doc["speed_simulation"].get(str(band))
            if not sim:
                continue

            d_alert_m = sim["d_alert_m"]

            # Calculate actual distance to pothole
            dist_m = self._haversine(
                lat, lon,
                doc["gps_readable"]["lat"],
                doc["gps_readable"]["lon"],
            )

            if dist_m <= d_alert_m:
                return {
                    "pothole_id":   str(doc["_id"]),
                    "distance_m":   round(dist_m, 1),
                    "d_alert_m":    round(d_alert_m, 1),
                    "severity":     sim["severity"],
                    "fall_type":    sim["fall_type"],
                    "message":      self._alert_message(sim["severity"]),
                    "color_hex":    self._alert_color(sim["severity"]),
                    "vibration":    self._vibration(sim["severity"]),
                    "sound":        self._sound(sim["severity"]),
                    "seen_count":   doc["seen_count"],
                }

        return None   # No pothole within alert range

    # ── Simulation builder ────────────────────────────────────────────────────

    def _build_simulation_table(
        self,
        pothole:        PotholeGeometry,
        road_condition: RoadCondition,
    ) -> dict:
        """
        Pre-compute physics for all speed bands.
        Stored once, looked up at runtime — zero recalculation overhead.
        """
        table = {}
        for speed in SPEED_BANDS:
            result = self.engine.calculate(
                speed_kmh      = speed,
                pothole        = pothole,
                road_condition = road_condition,
            )
            table[str(speed)] = {
                "d_alert_m":  round(result.d_alert_m, 2),
                "d_stop_m":   round(result.d_stop_m, 2),
                "d_react_m":  round(result.d_react_m, 2),
                "severity":   result.severity,
                "fall_type":  result.fall_type.value,
                "energy_j":   round(result.impact_energy_j, 1),
                # Damage summary (stored for reports, not shown to rider)
                "damage_summary": {
                    k: f"{round(v*100)}%"
                    for k, v in result.damage.items()
                    if v > 0.3   # only notable damage
                },
            }
        return table

    # ── Geo helpers ───────────────────────────────────────────────────────────

    async def _find_nearby(self, lat: float, lon: float) -> Optional[dict]:
        """Check if a pothole already exists within DEDUP_RADIUS_M."""
        try:
            doc = await self.collection.find_one(
                {
                    "gps": {
                        "$nearSphere": {
                            "$geometry": {
                                "type":        "Point",
                                "coordinates": [lon, lat],
                            },
                            "$maxDistance": DEDUP_RADIUS_M,
                        }
                    }
                }
            )
            return doc
        except Exception:
            # 2dsphere index may not exist yet in dev — skip dedup
            return None

    def _haversine(self, lat1, lon1, lat2, lon2) -> float:
        """Straight-line distance between two GPS coordinates (metres)."""
        import math
        R  = 6_371_000   # Earth radius in metres
        φ1 = math.radians(lat1);  φ2 = math.radians(lat2)
        Δφ = math.radians(lat2 - lat1)
        Δλ = math.radians(lon2 - lon1)
        a  = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def _speed_band(self, speed_kmh: float) -> int:
        """Snap speed to nearest pre-computed band."""
        bands = SPEED_BANDS
        return min(bands, key=lambda b: abs(b - speed_kmh))

    # ── Alert helpers ─────────────────────────────────────────────────────────

    def _alert_message(self, severity: str) -> str:
        return {
            "low":      "Minor pothole ahead. Reduce speed.",
            "medium":   "Pothole detected! Slow down now.",
            "high":     "DANGER: Deep pothole! Brake immediately.",
            "critical": "CRITICAL HAZARD! Stop if safe.",
        }[severity]

    def _alert_color(self, severity: str) -> str:
        return {
            "low": "#4CAF50", "medium": "#FF9800",
            "high": "#F44336", "critical": "#B71C1C",
        }[severity]

    def _vibration(self, severity: str) -> str:
        return {
            "low": "short", "medium": "double",
            "high": "long", "critical": "continuous",
        }[severity]

    def _sound(self, severity: str) -> str:
        return {
            "low": "beep", "medium": "warn",
            "high": "alert", "critical": "siren",
        }[severity]
