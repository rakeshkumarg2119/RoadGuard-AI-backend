"""
demo/gps_alert_simulator.py

Stage-demo module: replays the scripted approach toward a pothole
(48.2 m -> 38.5 m -> 28.1 m) through the SAME PotholePhysicsEngine the
real /alert route uses — no live GPS involved, nothing to go wrong on stage.

At 30 km/h on a dry road, d_alert works out to ~30.1 m (matches the
README's worked example for an 8cm pothole), which is exactly why
48.2 m and 38.5 m read "no alert" and 28.1 m is where it fires.

Real-time GPS reading is commented out at the bottom on purpose — only
switch it on for later bench/field testing, never for the stage demo.

Run standalone:
    python -m demo.gps_alert_simulator
"""
import asyncio

from core.physics_engine import (
    PotholePhysicsEngine,
    PotholeGeometry,
    RoadCondition,
)

# ---------------------------------------------------------------------------
# DEMO SCRIPT
# Each tuple: (distance_to_pothole_m, pause_seconds_before_next_step)
# ---------------------------------------------------------------------------
DEMO_STEPS = [
    (48.2, 2),
    (38.5, 2),
    (28.1, 2),
]

DEMO_SPEED_KMH = 30
DEMO_POTHOLE = PotholeGeometry(width_m=0.40, depth_m=0.08)  # README's 8cm example
DEMO_ROAD_CONDITION = RoadCondition.DRY

engine = PotholePhysicsEngine()


async def run_demo(speed_kmh: float = DEMO_SPEED_KMH) -> None:
    """
    Walks through DEMO_STEPS, printing exactly what a judge watching the
    app would see at each step. d_alert is computed once up front since
    it only depends on speed + road condition, not the pothole itself.
    """
    result = engine.calculate(speed_kmh, DEMO_POTHOLE, DEMO_ROAD_CONDITION)
    d_alert = result.d_alert_m

    print(f"Speed: {speed_kmh} km/h | d_alert: {round(d_alert, 1)} m | severity: {result.severity}\n")

    for distance_m, pause_s in DEMO_STEPS:
        should_alert = distance_m <= d_alert

        print(f"--- distance to pothole: {distance_m} m ---")
        print(f"  ALERT FIRING : {'YES' if should_alert else 'no'}")
        if should_alert:
            print("  sound=alert  flashlight=True")
            print('  message="DANGER: Deep pothole ahead! Brake immediately."')
        print()

        await asyncio.sleep(pause_s)


# ---------------------------------------------------------------------------
# REAL GPS MODE — commented out on purpose. Do NOT enable for the stage
# demo. Only for bench/field testing with an actual phone or GPS module.
# ---------------------------------------------------------------------------
#
# import math
# import gpsd  # pip install gpsd-py3
#
# def haversine_m(lat1, lon1, lat2, lon2):
#     R = 6371000
#     p1, p2 = math.radians(lat1), math.radians(lat2)
#     dphi = math.radians(lat2 - lat1)
#     dlambda = math.radians(lon2 - lon1)
#     a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
#     return 2 * R * math.asin(math.sqrt(a))
#
# async def run_live(pothole_lat: float, pothole_lon: float, poll_interval_s: float = 1.0):
#     gpsd.connect()
#     while True:
#         packet = gpsd.get_current()
#         lat, lon, speed_kmh = packet.lat, packet.lon, packet.hspeed * 3.6
#         distance_m = haversine_m(lat, lon, pothole_lat, pothole_lon)
#         result = engine.calculate(speed_kmh, DEMO_POTHOLE, DEMO_ROAD_CONDITION)
#         if distance_m <= result.d_alert_m:
#             print("ALERT: pothole ahead")
#         await asyncio.sleep(poll_interval_s)


if __name__ == "__main__":
    asyncio.run(run_demo())