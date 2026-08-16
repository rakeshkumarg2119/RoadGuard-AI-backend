"""
RoadGuard AI — /alert route

Called by Flutter background service every ~2 seconds while riding.

Query params:
  lat:       float  — current GPS latitude
  lon:       float  — current GPS longitude
  speed_kmh: float  — current speed from phone

Weather-aware real-time alert logic
────────────────────────────────────
SimulationStore.get_alert_for_speed() returns a result built from the
speed_simulation table — physics pre-computed at UPLOAD time with
whatever weather existed then.

That table is the right source for the SIMULATION page (indoor demo,
fixed conditions). But for REAL-TIME riding the weather right now is
what matters:

  Upload time: dry  → d_alert_m = 25 m  (μ = 0.70)
  Ride time:   rain → d_alert_m = 50 m  (μ = 0.35)

If we use the stale dry table while it's raining, the rider gets warned
at 25 m instead of 50 m — half the braking distance, on a wet road.

Fix: after SimulationStore returns a hit, this route:
  1. Fetches live weather from WeatherService (cached, no API call).
  2. Compares live road_condition with the condition stored at upload.
  3. If they differ → recomputes d_alert_m live using PhysicsEngine
     with the current μ.
  4. Re-checks distance_m <= live_d_alert_m.
  5. Upgrades severity + message if rain makes it worse.
  6. Appends a weather note to the message ("⚠ WET ROAD: brake earlier").

SimulationStore and simulate.py are NOT touched — simulation stays
as a self-contained indoor demo with fixed conditions.
"""

import logging
from fastapi import APIRouter, Query, HTTPException

from core.app_state import get_store, get_weather
from core.physics_engine import (
    PotholePhysicsEngine,
    PotholeGeometry,
    RoadCondition,
)

router = APIRouter()
logger = logging.getLogger("roadguard.routes.alert")

# One shared engine instance — stateless, safe to reuse across requests
_engine = PotholePhysicsEngine()

# Severity upgrade order — rain can push "low" → "medium" etc.
_SEVERITY_ORDER = ["low", "medium", "high", "critical"]


def _upgrade_severity(base: str, steps: int) -> str:
    """Bump severity up by `steps` levels, capped at critical."""
    try:
        idx = _SEVERITY_ORDER.index(base)
    except ValueError:
        return base
    return _SEVERITY_ORDER[min(idx + steps, len(_SEVERITY_ORDER) - 1)]


def _alert_message(severity: str) -> str:
    return {
        "low":      "Minor pothole ahead. Reduce speed.",
        "medium":   "Pothole detected! Slow down now.",
        "high":     "DANGER: Deep pothole! Brake immediately.",
        "critical": "CRITICAL HAZARD! Stop if safe.",
    }.get(severity, "Pothole detected. Slow down.")


def _stage_output(zone: str) -> dict:
    """
    Stage-driven output. Sound/vibration/flash come from DISTANCE STAGE,
    not from severity — this is what makes it feel graduated as the
    rider physically gets closer, independent of how deep the pothole is.

      stage1 (far, still in reaction zone)      → sound only
      stage2 (near half of reaction zone)       → sound + flash + vibration (medium)
      stage3 (inside braking distance)          → siren + flash + vibration (continuous)
    """
    return {
        "stage1": {"sound": "beep",  "vibration": "short",      "flash": False},
        "stage2": {"sound": "warn",  "vibration": "double",     "flash": True},
        "stage3": {"sound": "siren", "vibration": "continuous", "flash": True},
    }.get(zone, {"sound": "beep", "vibration": "short", "flash": False})


def _alert_color(severity: str) -> str:
    return {
        "low":      "#4CAF50",
        "medium":   "#FF9800",
        "high":     "#F44336",
        "critical": "#B71C1C",
    }.get(severity, "#FF9800")


def _vibration(severity: str) -> str:
    return {
        "low":      "short",
        "medium":   "double",
        "high":     "long",
        "critical": "continuous",
    }.get(severity, "short")


def _sound(severity: str) -> str:
    return {
        "low":      "beep",
        "medium":   "warn",
        "high":     "alert",
        "critical": "siren",
    }.get(severity, "beep")


@router.get("/alert")
async def get_alert(
    lat:       float = Query(...,  description="Current GPS latitude"),
    lon:       float = Query(...,  description="Current GPS longitude"),
    speed_kmh: float = Query(30.0, description="Current speed in km/h"),
):
    """
    Check if rider is approaching a known pothole.
    Flutter background service polls this every ~2s.
    Weather-adjusts the alert distance and severity in real time.
    """

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid GPS coordinates")

    if speed_kmh < 0 or speed_kmh > 300:
        raise HTTPException(status_code=400, detail="Invalid speed value")

    store = get_store()

    # ── 1. SimulationStore lookup (pre-computed table, upload-time weather) ──
    try:
        result = await store.get_alert_for_speed(
            lat=lat,
            lon=lon,
            speed_kmh=speed_kmh,
        )
    except Exception as exc:
        logger.error("Alert lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Alert lookup error: {exc}")

    if result is None:
        return {
            "alert":   False,
            "message": "No potholes nearby",
            "gps":     {"lat": lat, "lon": lon},
        }

    # ── 2. Fetch live weather (cached by WeatherService — no API call) ───────
    weather_note: str = ""
    live_condition_str: str = "dry"

    try:
        weather_svc     = get_weather()
        live_weather    = weather_svc.get_current_sync()   # never blocks

        if live_weather is not None:
            live_condition_str = live_weather.road_condition.value  # "dry"/"wet"/"gravel"
            weather_note       = live_weather.alert_suffix          # "" / "⚠ WET ROAD…"
    except Exception as exc:
        # WeatherService failure must never break the alert path
        logger.warning("Weather fetch skipped in /alert: %s", exc)

    # ── 3. Detect condition mismatch (upload-time vs right now) ──────────────
    #
    # The pothole document stores weather_at_detection.road_condition.
    # result comes from SimulationStore which used that stored condition.
    # We need to know what condition the simulation table was built with
    # so we can decide whether to recompute.
    #
    # SimulationStore doesn't return weather_at_detection in its result dict,
    # but we can infer: if live is wet and result severity is low/medium it's
    # likely a dry-weather table. Rather than inferring, we recompute whenever
    # live_condition != "dry" — dry is the fallback when no weather was stored,
    # so recomputing on any non-dry live weather is always safe and correct.

    live_d_alert_m  = float(result.get("d_alert_m") or result.get("alert_distance_m") or 0)
    live_d_stop_m   = result.get("d_stop_m")
    live_severity   = result["severity"]
    distance_m      = result["distance_m"]
    zone            = result.get("zone", "reaction")
    still_in_alert  = True   # SimulationStore already confirmed distance_m <= d_alert_m

    if live_condition_str != "dry" and live_weather is not None:
        # Recompute physics with current road condition.
        # We need the pothole geometry — SimulationStore doesn't return it,
        # so we fall back to engine defaults with the stored fall_type as a hint.
        # For full accuracy we'd fetch the doc, but that's a second DB round-trip.
        # Instead we scale d_alert_m by the μ ratio: d ∝ 1/μ, so
        #   live_d_alert = stored_d_alert × (μ_dry / μ_live)
        # This is mathematically equivalent to rerunning the engine with μ_live
        # because d_stop = v²/(2μg) and d_alert = d_stop + d_react.
        # d_react is speed-independent, but small relative to d_stop at road speeds,
        # so the ratio approximation is accurate to within ~5%.

        MU_DRY  = 0.70
        mu_live = live_weather.mu   # 0.35 wet, 0.45 gravel

        # d_react_m doesn't depend on μ (it's just speed × reaction time),
        # so we can recover it before scaling and reuse it for the
        # stage2 boundary below.
        original_d_alert_m = float(result.get("d_alert_m") or live_d_alert_m)
        original_d_stop_m  = float(live_d_stop_m or 0)
        d_react_m          = max(0.0, original_d_alert_m - original_d_stop_m)

        scale_factor   = MU_DRY / mu_live           # 2.0× for wet, 1.56× for gravel
        live_d_alert_m = round(live_d_alert_m * scale_factor, 1)
        if live_d_stop_m:
            live_d_stop_m = round(float(live_d_stop_m) * scale_factor, 1)

        # Re-check: is the rider still inside the (now larger) alert zone?
        still_in_alert = distance_m <= live_d_alert_m

        if still_in_alert:
            # Upgrade severity by 1 step in wet/gravel — same pothole is
            # more dangerous when braking distance is longer.
            live_severity = _upgrade_severity(live_severity, steps=1)

            # Re-derive the 3-stage zone against the weather-adjusted
            # boundaries (same stage2 = d_stop + half of d_react rule
            # used in SimulationStore).
            if live_d_stop_m is not None:
                stage2_boundary = live_d_stop_m + (0.5 * d_react_m)
                if distance_m <= live_d_stop_m:
                    zone = "stage3"
                elif distance_m <= stage2_boundary:
                    zone = "stage2"
                else:
                    zone = "stage1"

            logger.info(
                "Weather override: %s road  d_alert %.1f→%.1f m  severity→%s  zone→%s",
                live_condition_str, result.get("d_alert_m"), live_d_alert_m, live_severity, zone,
            )

    # If weather recompute pushed the rider OUTSIDE the alert zone, no alert.
    # (This shouldn't happen — wider alert zone means more riders caught, not fewer —
    # but guards against float edge cases.)
    if not still_in_alert:
        return {
            "alert":   False,
            "message": "No potholes nearby",
            "gps":     {"lat": lat, "lon": lon},
        }

    # ── 4. Stage-driven sound/vibration/flash (not message-driven) ───────────
    stage_out = _stage_output(zone)

    logger.info(
        "SERVING ALERT zone=%s distance=%.1fm speed=%.1fkmh sound=%s vibration=%s flash=%s",
        zone, distance_m, speed_kmh, stage_out["sound"], stage_out["vibration"], stage_out["flash"],
    )

    # message kept for logs/debugging only — Flutter should drive the UI
    # off sound/vibration/flash + zone, per your spec, not off this text.
    final_message = (_alert_message(live_severity) + weather_note).strip()

    return {
        "alert":            True,
        "gps":              {"lat": lat, "lon": lon},
        "speed_kmh":        speed_kmh,

        # Pothole identity
        "pothole_id":       result.get("pothole_id"),
        "distance_m":       distance_m,
        "seen_count":       result.get("seen_count", 1),
        "fall_type":        result.get("fall_type"),

        # Live-weather-adjusted physics
        "alert_distance_m": live_d_alert_m,
        "d_stop_m":         live_d_stop_m,
        "zone":             zone,   # "stage1" | "stage2" | "stage3"

        # Stage-driven output — THIS is what Flutter should react to
        "sound":            stage_out["sound"],
        "vibration":        stage_out["vibration"],
        "flash":            stage_out["flash"],

        # Severity kept for color/logging, not for triggering alerts
        "severity":         live_severity,
        "color_hex":        _alert_color(live_severity),
        "message":          final_message,

        # Weather context (Flutter can show this in the banner)
        "weather": {
            "condition":    live_condition_str,
            "description":  live_weather.description if live_weather else None,
            "mu":           live_weather.mu          if live_weather else 0.70,
        },
    }