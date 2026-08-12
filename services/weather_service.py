"""
Road Guard AI — Weather Service
Fetches weather every 30 mins from OpenWeatherMap.
Maps weather condition → road friction coefficient (μ).
Physics engine uses μ to recalculate d_alert in real time.

Setup:
    Set OPENWEATHER_API_KEY in your .env file.
    Free tier: 60 calls/min, 1M calls/month — more than enough.

Usage:
    weather = WeatherService(lat=9.9252, lon=78.1198)
    condition = await weather.get_current()
    print(condition.road_condition)   # RoadCondition.WET
    print(condition.mu)               # 0.35
"""

import os
import time
import logging
import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Optional

from core.physics_engine import RoadCondition

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_URL     = "https://api.openweathermap.org/data/2.5/weather"
FETCH_INTERVAL_SEC  = 30 * 60      # 30 minutes


# ── Weather ID → Road condition mapping ───────────────────────────────────────
# OpenWeatherMap condition codes:
# https://openweathermap.org/weather-conditions

WEATHER_ID_MAP = {
    # Thunderstorm (200–299)
    range(200, 300): ("wet",    "Thunderstorm — extreme caution"),
    # Drizzle (300–399)
    range(300, 400): ("wet",    "Drizzle — roads slippery"),
    # Rain (500–599)
    range(500, 600): ("wet",    "Rain — reduced friction"),
    # Snow (600–699) — rare in Tamil Nadu but handled
    range(600, 700): ("gravel", "Wet/icy — severe grip loss"),
    # Atmosphere (700–799): fog, haze, dust
    range(700, 800): ("dry",    "Low visibility — reduce speed"),
    # Clear (800)
    range(800, 801): ("dry",    "Clear roads"),
    # Clouds (801–899)
    range(801, 900): ("dry",    "Cloudy — normal conditions"),
}

# Friction coefficients per condition
MU_MAP = {
    "dry":    0.70,
    "wet":    0.35,
    "gravel": 0.45,
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class WeatherCondition:
    """Current weather state used by physics engine."""
    road_condition: RoadCondition
    mu:             float
    description:    str
    temp_celsius:   float
    humidity_pct:   int
    wind_kmh:       float
    weather_id:     int
    fetched_at:     float = field(default_factory=time.time)

    @property
    def age_minutes(self) -> float:
        return (time.time() - self.fetched_at) / 60

    @property
    def is_storm(self) -> bool:
        return 200 <= self.weather_id < 300

    @property
    def is_rain(self) -> bool:
        return 300 <= self.weather_id < 600

    @property
    def alert_suffix(self) -> str:
        """Extra text appended to alert messages in bad weather."""
        if self.is_storm:
            return " ⚠ STORM: stopping distance doubled."
        if self.is_rain:
            return " ⚠ WET ROAD: brake earlier."
        return ""

    def to_dict(self) -> dict:
        return {
            "road_condition": self.road_condition.value,
            "mu":             self.mu,
            "description":    self.description,
            "temp_celsius":   self.temp_celsius,
            "humidity_pct":   self.humidity_pct,
            "wind_kmh":       self.wind_kmh,
            "weather_id":     self.weather_id,
            "fetched_at":     self.fetched_at,
            "age_minutes":    round(self.age_minutes, 1),
        }


# ── Weather Service ───────────────────────────────────────────────────────────

class WeatherService:
    """
    Async weather poller.
    Keeps one cached WeatherCondition, refreshed every 30 minutes.
    Thread-safe: all async, single event loop.
    """

    def __init__(
        self,
        lat:      float,
        lon:      float,
        api_key:  Optional[str] = None,
        interval: int = FETCH_INTERVAL_SEC,
    ):
        self.lat      = lat
        self.lon      = lon
        self.api_key  = api_key or OPENWEATHER_API_KEY
        self.interval = interval
        self._cache:  Optional[WeatherCondition] = None
        self._task:   Optional[asyncio.Task]     = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self):
        """Start background polling. Call once at app startup."""
        await self._fetch_and_cache()                    # immediate first fetch
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"Weather polling started for ({self.lat}, {self.lon})")

    async def stop(self):
        """Cancel background polling. Call at app shutdown."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def get_current(self) -> WeatherCondition:
        """
        Return cached weather.
        Forces a fresh fetch if cache is empty or stale (> interval).
        """
        if self._cache is None or self._cache.age_minutes > self.interval / 60:
            await self._fetch_and_cache()
        return self._cache

    def get_current_sync(self) -> Optional[WeatherCondition]:
        """
        Synchronous access to cached value (no network call).
        Returns None if never fetched yet.
        Used by FastAPI background tasks.
        """
        return self._cache

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _poll_loop(self):
        """Runs forever, fetching every `interval` seconds."""
        while True:
            await asyncio.sleep(self.interval)
            await self._fetch_and_cache()

    async def _fetch_and_cache(self):
        if not self.api_key:
            logger.warning("No OPENWEATHER_API_KEY set — using DRY fallback")
            self._cache = self._dry_fallback()
            return

        params = {
            "lat":   self.lat,
            "lon":   self.lon,
            "appid": self.api_key,
            "units": "metric",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    OPENWEATHER_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Weather API returned {resp.status}")
                        if self._cache is None:
                            self._cache = self._dry_fallback()
                        return

                    data = await resp.json()
                    self._cache = self._parse(data)
                    logger.info(
                        f"Weather updated: {self._cache.description} "
                        f"(μ={self._cache.mu})"
                    )

        except asyncio.TimeoutError:
            logger.error("Weather API timeout")
            if self._cache is None:
                self._cache = self._dry_fallback()
        except Exception as e:
            logger.error(f"Weather fetch error: {e}")
            if self._cache is None:
                self._cache = self._dry_fallback()

    def _parse(self, data: dict) -> WeatherCondition:
        """Parse OpenWeatherMap /weather JSON response."""
        weather_id  = data["weather"][0]["id"]
        description = data["weather"][0]["description"].capitalize()
        temp        = data["main"]["temp"]
        humidity    = data["main"]["humidity"]
        wind_ms     = data["wind"].get("speed", 0)

        cond_key, note = self._map_condition(weather_id)

        return WeatherCondition(
            road_condition = RoadCondition(cond_key),
            mu             = MU_MAP[cond_key],
            description    = f"{description} — {note}",
            temp_celsius   = round(temp, 1),
            humidity_pct   = humidity,
            wind_kmh       = round(wind_ms * 3.6, 1),
            weather_id     = weather_id,
        )

    def _map_condition(self, weather_id: int) -> tuple[str, str]:
        """Map OpenWeatherMap ID to (road_condition, note)."""
        for id_range, (cond, note) in WEATHER_ID_MAP.items():
            if weather_id in id_range:
                return cond, note
        return "dry", "Unknown — assuming dry"

    def _dry_fallback(self) -> WeatherCondition:
        return WeatherCondition(
            road_condition = RoadCondition.DRY,
            mu             = MU_MAP["dry"],
            description    = "Fallback — no API key",
            temp_celsius   = 30.0,
            humidity_pct   = 60,
            wind_kmh       = 0.0,
            weather_id     = 800,
        )
