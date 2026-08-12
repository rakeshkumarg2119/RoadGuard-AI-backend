"""
RoadGuard AI - Application State

Connects:

MongoDB
   ↓
SimulationStore

WeatherService
   ↓
Background weather polling

PotholeDetector
   ↓
Roboflow / local YOLO inference
"""

import os
import logging
from typing import Optional

from core.db import (
    connect_to_mongo,
    close_mongo_connection,
    get_potholes_collection,
)

from services.simulation_store import SimulationStore
from services.weather_service import WeatherService
from core.pothole_detector import PotholeDetector


logger = logging.getLogger("roadguard.app_state")


_store: Optional[SimulationStore] = None
_weather: Optional[WeatherService] = None
_detector: Optional[PotholeDetector] = None


async def connect_all() -> None:
    """
    Start all backend services.

    Called from FastAPI application startup.
    """

    global _store
    global _weather
    global _detector

    logger.info("Starting RoadGuard backend services...")

    # ---------------------------------------------------------
    # MongoDB
    # ---------------------------------------------------------

    await connect_to_mongo()

    _store = SimulationStore(
        get_potholes_collection()
    )

    logger.info(
        "SimulationStore connected to MongoDB."
    )

    # ---------------------------------------------------------
    # Weather
    # ---------------------------------------------------------

    lat = float(
        os.getenv(
            "DEFAULT_LAT",
            "9.9252",
        )
    )

    lon = float(
        os.getenv(
            "DEFAULT_LON",
            "78.1198",
        )
    )

    _weather = WeatherService(
        lat=lat,
        lon=lon,
    )

    await _weather.start()

    logger.info(
        "WeatherService started. lat=%s lon=%s",
        lat,
        lon,
    )

    # ---------------------------------------------------------
    # Pothole Detector
    # ---------------------------------------------------------

    _detector = PotholeDetector()

    logger.info("PotholeDetector initialized.")

    logger.info(
        "All RoadGuard backend services started."
    )


async def disconnect_all() -> None:
    """
    Stop all backend services.
    """

    global _store
    global _weather
    global _detector

    logger.info(
        "Stopping RoadGuard backend services..."
    )

    # ---------------------------------------------------------
    # Weather
    # ---------------------------------------------------------

    if _weather is not None:

        try:
            await _weather.stop()

        except Exception as exc:
            logger.warning(
                "WeatherService shutdown warning: %s",
                exc,
            )

        _weather = None

    # ---------------------------------------------------------
    # Pothole Detector
    # ---------------------------------------------------------

    _detector = None

    # ---------------------------------------------------------
    # MongoDB
    # ---------------------------------------------------------

    try:
        await close_mongo_connection()

    except Exception as exc:
        logger.warning(
            "MongoDB shutdown warning: %s",
            exc,
        )

    _store = None

    logger.info(
        "All RoadGuard backend services stopped."
    )


def get_store() -> SimulationStore:
    """
    Return the active SimulationStore.
    """

    if _store is None:
        raise RuntimeError(
            "SimulationStore is not initialized. "
            "Call connect_all() during application startup."
        )

    return _store


def get_weather() -> WeatherService:
    """
    Return the active WeatherService.
    """

    if _weather is None:
        raise RuntimeError(
            "WeatherService is not initialized. "
            "Call connect_all() during application startup."
        )

    return _weather


def get_detector() -> PotholeDetector:
    """
    Return the active PotholeDetector.
    """

    if _detector is None:
        raise RuntimeError(
            "PotholeDetector is not initialized. "
            "Call connect_all() during application startup."
        )

    return _detector