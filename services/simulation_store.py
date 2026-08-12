"""
RoadGuard AI - Simulation Store

Stores detected potholes in MongoDB and pre-computes
physics results for multiple speed bands.

Flow:

Pothole Detection
       ↓
Pothole Geometry
       ↓
Physics Engine
       ↓
Speed Simulation Table
       ↓
MongoDB

At alert time, the backend can quickly retrieve the
pre-computed result instead of recalculating physics.
"""

import math
import time
import logging

from typing import Optional

from core.physics_engine import (
    PotholePhysicsEngine,
    PotholeGeometry,
    BikeConfig,
    RoadCondition,
)


logger = logging.getLogger("roadguard.simulation_store")


# ============================================================
# SPEED BANDS
# ============================================================

SPEED_BANDS = [
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
]


# ============================================================
# DEDUPLICATION
# ============================================================

DEDUP_RADIUS_M = 15.0


class SimulationStore:
    """
    Handles pothole storage and speed simulations.
    """

    def __init__(
        self,
        collection,
        bike_config: Optional[BikeConfig] = None,
    ):

        self.collection = collection

        self.engine = PotholePhysicsEngine(
            bike_config or BikeConfig()
        )

        logger.info(
            "SimulationStore initialized."
        )

    # ========================================================
    # SAVE POTHOLE
    # ========================================================

    async def save(
        self,
        detection_payload: dict,
        weather_condition=None,
    ) -> Optional[str]:
        """
        Save a detected pothole to MongoDB.

        If a pothole already exists within DEDUP_RADIUS_M,
        update that record instead of creating a duplicate.
        """

        gps = detection_payload.get("gps")

        if not gps:
            logger.warning(
                "Pothole payload has no GPS coordinates."
            )

            return None

        physics_data = detection_payload.get(
            "physics"
        )

        if not physics_data:
            logger.warning(
                "Pothole payload has no physics data."
            )

            return None

        pothole_data = physics_data.get(
            "pothole"
        )

        if not pothole_data:
            logger.warning(
                "Pothole payload has no pothole geometry."
            )

            return None

        # ----------------------------------------------------
        # Create geometry
        # ----------------------------------------------------

        pothole = PotholeGeometry(
            width_m=float(
                pothole_data["width_m"]
            ),

            depth_m=float(
                pothole_data["depth_m"]
            ),

            confidence=float(
                pothole_data.get(
                    "confidence",
                    1.0,
                )
            ),
        )

        # ----------------------------------------------------
        # Road condition
        # ----------------------------------------------------

        road_condition = RoadCondition.DRY

        if weather_condition is not None:

            road_condition = (
                weather_condition.road_condition
            )

        # ----------------------------------------------------
        # Build simulation table
        # ----------------------------------------------------

        simulation_table = (
            self._build_simulation_table(
                pothole,
                road_condition,
            )
        )

        # ----------------------------------------------------
        # Search for nearby pothole
        # ----------------------------------------------------

        existing = await self._find_nearby(
            gps["lat"],
            gps["lon"],
        )

        weather_dict = None

        if weather_condition is not None:

            if hasattr(
                weather_condition,
                "to_dict",
            ):
                weather_dict = (
                    weather_condition.to_dict()
                )

        # ====================================================
        # UPDATE EXISTING
        # ====================================================

        if existing:

            await self.collection.update_one(
                {
                    "_id": existing["_id"]
                },
                {
                    "$set": {
                        "last_seen_at": time.time(),

                        "speed_simulation":
                            simulation_table,

                        "weather_at_detection":
                            weather_dict,

                        "detection":
                            detection_payload.get(
                                "detection",
                                {},
                            ),
                    },

                    "$inc": {
                        "seen_count": 1
                    },
                },
            )

            logger.info(
                "Updated existing pothole %s",
                existing["_id"],
            )

            return str(
                existing["_id"]
            )

        # ====================================================
        # INSERT NEW
        # ====================================================

        now = time.time()

        document = {

            "gps": {
                "type": "Point",

                "coordinates": [
                    gps["lon"],
                    gps["lat"],
                ],
            },

            "gps_readable": {
                "lat": gps["lat"],
                "lon": gps["lon"],
            },

            "pothole": pothole_data,

            "detection":
                detection_payload.get(
                    "detection",
                    {},
                ),

            "speed_simulation":
                simulation_table,

            "weather_at_detection":
                weather_dict,

            "created_at": now,

            "last_seen_at": now,

            "seen_count": 1,

            "verified": False,
        }

        result = await self.collection.insert_one(
            document
        )

        logger.info(
            "New pothole stored: %s",
            result.inserted_id,
        )

        return str(
            result.inserted_id
        )

    # ========================================================
    # ALERT LOOKUP
    # ========================================================

    async def get_alert_for_speed(
        self,
        lat: float,
        lon: float,
        speed_kmh: float,
    ) -> Optional[dict]:
        """
        Find the nearest pothole and determine whether
        the rider is inside the alert distance.
        """

        band = self._speed_band(
            speed_kmh
        )

        cursor = self.collection.find(
            {
                "gps": {
                    "$nearSphere": {
                        "$geometry": {
                            "type": "Point",

                            "coordinates": [
                                lon,
                                lat,
                            ],
                        },

                        "$maxDistance": 200,
                    }
                }
            }
        ).limit(3)

        async for doc in cursor:

            simulation = (
                doc.get(
                    "speed_simulation",
                    {},
                ).get(
                    str(band)
                )
            )

            if not simulation:
                continue

            d_alert_m = float(
                simulation["d_alert_m"]
            )

            gps_readable = doc.get(
                "gps_readable"
            )

            if not gps_readable:
                continue

            distance_m = self._haversine(
                lat,
                lon,
                gps_readable["lat"],
                gps_readable["lon"],
            )

            if distance_m <= d_alert_m:

                severity = simulation[
                    "severity"
                ]

                return {

                    "pothole_id":
                        str(doc["_id"]),

                    "distance_m":
                        round(
                            distance_m,
                            1,
                        ),

                    "d_alert_m":
                        round(
                            d_alert_m,
                            1,
                        ),

                    "severity":
                        severity,

                    "fall_type":
                        simulation[
                            "fall_type"
                        ],

                    "message":
                        self._alert_message(
                            severity
                        ),

                    "color_hex":
                        self._alert_color(
                            severity
                        ),

                    "vibration":
                        self._vibration(
                            severity
                        ),

                    "sound":
                        self._sound(
                            severity
                        ),

                    "seen_count":
                        doc.get(
                            "seen_count",
                            1,
                        ),
                }

        return None

    # ========================================================
    # SIMULATION TABLE
    # ========================================================

    def _build_simulation_table(
        self,
        pothole: PotholeGeometry,
        road_condition: RoadCondition,
    ) -> dict:
        """
        Pre-compute physics for all speed bands.
        """

        table = {}

        for speed in SPEED_BANDS:

            result = self.engine.calculate(
                speed_kmh=speed,

                pothole=pothole,

                road_condition=
                    road_condition,
            )

            table[str(speed)] = {

                "d_alert_m":
                    round(
                        result.d_alert_m,
                        2,
                    ),

                "d_stop_m":
                    round(
                        result.d_stop_m,
                        2,
                    ),

                "d_react_m":
                    round(
                        result.d_react_m,
                        2,
                    ),

                "severity":
                    result.severity,

                "fall_type":
                    result.fall_type.value,

                "energy_j":
                    round(
                        result.impact_energy_j,
                        1,
                    ),

                "damage_summary": {

                    key:
                        f"{round(value * 100)}%"

                    for key, value
                    in result.damage.items()

                    if value > 0.3
                },
            }

        return table

    # ========================================================
    # FIND NEARBY
    # ========================================================

    async def _find_nearby(
        self,
        lat: float,
        lon: float,
    ) -> Optional[dict]:
        """
        Find an existing pothole within
        DEDUP_RADIUS_M.
        """

        try:

            document = (
                await self.collection.find_one(
                    {
                        "gps": {
                            "$nearSphere": {

                                "$geometry": {
                                    "type":
                                        "Point",

                                    "coordinates": [
                                        lon,
                                        lat,
                                    ],
                                },

                                "$maxDistance":
                                    DEDUP_RADIUS_M,
                            }
                        }
                    }
                )
            )

            return document

        except Exception as exc:

            logger.warning(
                "Nearby pothole lookup failed: %s",
                exc,
            )

            return None

    # ========================================================
    # HAVERSINE
    # ========================================================

    @staticmethod
    def _haversine(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """
        Calculate distance between two GPS points
        in metres.
        """

        earth_radius = 6_371_000.0

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)

        delta_phi = math.radians(
            lat2 - lat1
        )

        delta_lambda = math.radians(
            lon2 - lon1
        )

        a = (
            math.sin(
                delta_phi / 2
            ) ** 2

            +

            math.cos(phi1)
            * math.cos(phi2)
            * math.sin(
                delta_lambda / 2
            ) ** 2
        )

        return (
            earth_radius
            * 2
            * math.atan2(
                math.sqrt(a),
                math.sqrt(1 - a),
            )
        )

    # ========================================================
    # SPEED BAND
    # ========================================================

    @staticmethod
    def _speed_band(
        speed_kmh: float,
    ) -> int:
        """
        Select the nearest pre-computed speed band.
        """

        return min(
            SPEED_BANDS,
            key=lambda band:
                abs(
                    band - speed_kmh
                ),
        )

    # ========================================================
    # ALERT MESSAGE
    # ========================================================

    @staticmethod
    def _alert_message(
        severity: str,
    ) -> str:

        messages = {

            "low":
                "Minor pothole ahead. Reduce speed.",

            "medium":
                "Pothole detected! Slow down now.",

            "high":
                "DANGER: Deep pothole! Brake immediately.",

            "critical":
                "CRITICAL HAZARD! Stop if safe.",
        }

        return messages.get(
            severity,
            "Pothole detected. Slow down.",
        )

    # ========================================================
    # ALERT COLOR
    # ========================================================

    @staticmethod
    def _alert_color(
        severity: str,
    ) -> str:

        colors = {

            "low":
                "#4CAF50",

            "medium":
                "#FF9800",

            "high":
                "#F44336",

            "critical":
                "#B71C1C",
        }

        return colors.get(
            severity,
            "#FF9800",
        )

    # ========================================================
    # VIBRATION
    # ========================================================

    @staticmethod
    def _vibration(
        severity: str,
    ) -> str:

        vibration = {

            "low":
                "short",

            "medium":
                "double",

            "high":
                "long",

            "critical":
                "continuous",
        }

        return vibration.get(
            severity,
            "short",
        )

    # ========================================================
    # SOUND
    # ========================================================

    @staticmethod
    def _sound(
        severity: str,
    ) -> str:

        sounds = {

            "low":
                "beep",

            "medium":
                "warn",

            "high":
                "alert",

            "critical":
                "siren",
        }

        return sounds.get(
            severity,
            "beep",
        )