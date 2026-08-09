"""
Road Guard AI — Pipeline Tests
Tests the physics engine, calibration, and full detector pipeline
using synthetic data (no real image needed).
"""

import sys
import math
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from road_guard_ai.core import (
    PotholePhysicsEngine,
    PotholeGeometry,
    BikeConfig,
    RoadCondition,
    FallType,
    CameraCalibrator,
    BoundingBox,
    PotholeDetector,
)

PASS = "✅"
FAIL = "❌"

results = []

def test(name, condition, details=""):
    status = PASS if condition else FAIL
    results.append((status, name, details))
    print(f"  {status}  {name}", f"({details})" if details else "")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  ROAD GUARD AI — PIPELINE TEST SUITE")
print("═"*60)

# ══ 1. Physics Engine ════════════════════════════════════════════════════════
print("\n▶ PHYSICS ENGINE")
engine = PotholePhysicsEngine()

# Test 1a: d_alert formula
pothole = PotholeGeometry(width_m=0.40, depth_m=0.05)
r = engine.calculate(speed_kmh=40, pothole=pothole, road_condition=RoadCondition.DRY)

v_ms    = 40 / 3.6
exp_react = v_ms * 3.0
exp_stop  = (v_ms**2) / (2 * 0.70 * 9.81)
exp_alert = exp_react + exp_stop

test("d_react matches formula",
     abs(r.d_react_m - exp_react) < 0.01,
     f"{r.d_react_m:.2f}m vs expected {exp_react:.2f}m")

test("d_stop matches formula",
     abs(r.d_stop_m - exp_stop) < 0.01,
     f"{r.d_stop_m:.2f}m vs expected {exp_stop:.2f}m")

test("d_alert = d_stop + d_react",
     abs(r.d_alert_m - exp_alert) < 0.01,
     f"d_alert={r.d_alert_m:.2f}m")

# Test 1b: Wet road increases stopping distance
r_wet = engine.calculate(40, pothole, RoadCondition.WET)
test("Wet road > dry stopping distance",
     r_wet.d_stop_m > r.d_stop_m,
     f"wet={r_wet.d_stop_m:.1f}m dry={r.d_stop_m:.1f}m")

# Test 1c: Zero speed → zero distances
r_zero = engine.calculate(0, pothole, RoadCondition.DRY)
test("Zero speed → zero distances",
     r_zero.d_react_m == 0 and r_zero.d_stop_m == 0)

# Test 1d: Fall type classification
cases = [
    (10, 0.02, FallType.SAFE,       "10 km/h, 2 cm → SAFE"),
    (25, 0.04, FallType.CONTROLLED, "25 km/h, 4 cm → CONTROLLED"),
    (40, 0.08, FallType.SIDE_SLIDE, "40 km/h, 8 cm → SIDE_SLIDE"),
    (70, 0.15, FallType.OVER_BARS,  "70 km/h, 15 cm → OVER_BARS"),
]
print("\n  Fall type classification:")
for spd, dep, expected, label in cases:
    p = PotholeGeometry(width_m=0.5, depth_m=dep)
    r2 = engine.calculate(spd, p, RoadCondition.DRY)
    test(label, r2.fall_type == expected, f"got {r2.fall_type.value}")

# Test 1e: Impact energy scales with speed
r_slow = engine.calculate(20, pothole)
r_fast = engine.calculate(80, pothole)
test("Impact energy scales with speed²",
     r_fast.impact_energy_j > r_slow.impact_energy_j * 3,
     f"20 km/h={r_slow.impact_energy_j:.0f}J  80 km/h={r_fast.impact_energy_j:.0f}J")

# Test 1f: Injury risk is zero for SAFE falls
r_safe = engine.calculate(10, PotholeGeometry(0.3, 0.01))
test("SAFE fall → all injury risks = 0.0",
     all(v == 0.0 for v in r_safe.injury_risk.values()),
     str(r_safe.injury_risk))

# Test 1g: Severity labels
sev_cases = [
    (15, 0.02, "low"),      # E ≈ 113 J
    (30, 0.05, "medium"),   # E ≈ 1127 J
    (45, 0.07, "high"),     # E ≈ 2521 J
    (80, 0.15, "critical"), # E ≈ 18703 J
]
print("\n  Severity labels:")
for spd, dep, exp_sev in sev_cases:
    p = PotholeGeometry(0.5, dep)
    r3 = engine.calculate(spd, p)
    test(f"{spd} km/h, {dep*100:.0f}cm → {exp_sev}",
         r3.severity == exp_sev,
         f"got '{r3.severity}'")


# ══ 2. Camera Calibration ════════════════════════════════════════════════════
print("\n▶ CAMERA CALIBRATION")
cal = CameraCalibrator(
    focal_length_px=900,
    mount_height_m=1.10,
    camera_tilt_deg=15.0,
    img_w=1280,
    img_h=720,
)

# Pothole in lower portion of frame (close to bike)
close_bbox = BoundingBox(540, 580, 740, 650, 0.88)
w_m, d_m, dist_m = cal.bbox_to_real_world(close_bbox)
test("Close pothole: distance < 5 m",
     dist_m < 5.0,
     f"dist={dist_m:.2f}m")
test("Close pothole: width in 0.1–2 m range",
     0.1 <= w_m <= 2.0,
     f"width={w_m:.3f}m")
test("Close pothole: depth 2–30 cm",
     0.02 <= d_m <= 0.30,
     f"depth={d_m*100:.1f}cm")

# Pothole in upper portion of frame (far from bike)
far_bbox = BoundingBox(600, 280, 680, 310, 0.72)
w_m2, d_m2, dist_m2 = cal.bbox_to_real_world(far_bbox)
test("Far pothole: distance > close pothole",
     dist_m2 > dist_m,
     f"far={dist_m2:.2f}m close={dist_m:.2f}m")

# from_yolo_xywh constructor
bbox_norm = BoundingBox.from_yolo_xywh(0.5, 0.8, 0.15, 0.08, 1280, 720, 0.91)
test("BoundingBox from normalised xywh",
     abs(bbox_norm.cx - 640) < 1 and abs(bbox_norm.cy - 576) < 1,
     f"cx={bbox_norm.cx:.0f} cy={bbox_norm.cy:.0f}")


# ══ 3. Full Pipeline — Synthetic Image ═══════════════════════════════════════
print("\n▶ FULL PIPELINE (synthetic frame)")

# Create a blank road-coloured frame with a dark patch simulating a pothole
frame = np.full((720, 1280, 3), fill_value=80, dtype=np.uint8)  # dark grey road
# Pothole region: darker rectangle
frame[560:620, 540:740] = 20   # near-black pothole

detector = PotholeDetector(
    api_key=None,
    use_cloud=False,
    min_confidence=0.10,
)

test("Detector initialised",
     detector is not None,
     f"cloud={detector.use_cloud}")

# process_frame on synthetic image (YOLO likely won't detect anything real,
# but we verify the pipeline runs without errors)
try:
    detections = detector.process_frame(
        frame,
        speed_kmh=45,
        road_condition=RoadCondition.DRY,
        gps_coords=(9.9252, 78.1198),   # Madurai coordinates
    )
    test("process_frame runs without error",
         True,
         f"{len(detections)} detections on synthetic frame")
except Exception as e:
    test("process_frame runs without error", False, str(e))

# Directly test physics integration via process_image_file path (unit test bypass)
# Manually build a payload like the detector would
from road_guard_ai.core.camera_calibration import BoundingBox
from road_guard_ai.core.physics_engine import PotholePhysicsEngine, PotholeGeometry, RoadCondition

test_bbox  = BoundingBox(540, 580, 740, 650, 0.88)
cal2       = CameraCalibrator()
w_m, d_m, dist_m = cal2.bbox_to_real_world(test_bbox)
ph_geom    = PotholeGeometry(width_m=w_m, depth_m=d_m, confidence=0.88)
ph_result  = PotholePhysicsEngine().calculate(50, ph_geom, RoadCondition.DRY)

test("Integrated geometry → physics chain",
     ph_result.d_alert_m > 0,
     f"d_alert={ph_result.d_alert_m:.2f}m  fall={ph_result.fall_type.value}")

out = ph_result.to_dict()
test("to_dict() has all required keys",
     all(k in out for k in ["kinematics", "impact", "damage", "injury_risk"]))


# ══ 4. to_dict Output Format ═════════════════════════════════════════════════
print("\n▶ OUTPUT FORMAT (MongoDB-ready)")
sample = engine.calculate(
    speed_kmh=55,
    pothole=PotholeGeometry(width_m=0.6, depth_m=0.09),
    road_condition=RoadCondition.WET,
)
payload = sample.to_dict()
json_str = json.dumps(payload, indent=2)
test("Output is valid JSON",  True)
test("Kinematics block present", "d_alert_m" in payload["kinematics"])
test("Damage block present",     len(payload["damage"]) > 0)
test("Injury risk block present", len(payload["injury_risk"]) > 0)

print("\n  Sample output payload (55 km/h, 9 cm pothole, wet road):")
print("  " + json_str.replace("\n", "\n  "))


# ══ Summary ══════════════════════════════════════════════════════════════════
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print("\n" + "═"*60)
print(f"  TOTAL: {len(results)} tests — {passed} passed, {failed} failed")
print("═"*60 + "\n")

if failed:
    sys.exit(1)
