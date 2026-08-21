"""
RoadGuard AI — Pre-flight Route Health Check
=============================================
Run this before real-world testing to confirm every
backend route is alive and responding correctly.

Usage:
    python preflight_check.py

Requirements:
    pip install requests Pillow
"""

import io
import sys
import time

import requests
from PIL import Image, ImageDraw

# ── CONFIG ────────────────────────────────────────────
BASE_URL = "https://parsley-oxidizing-stoplight.ngrok-free.dev"

HEADERS = {
    "ngrok-skip-browser-warning": "true",
}

TIMEOUT = 15  # seconds per request

# Madurai test coordinates (same as simulation)
TEST_LAT = 9.9252
TEST_LON = 78.1198

# ── ANSI COLORS ───────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ── HELPERS ───────────────────────────────────────────

def _dummy_image_bytes() -> bytes:
    """Generate a small synthetic JPEG — no real photo needed."""
    img = Image.new("RGB", (320, 240), color=(80, 80, 80))
    draw = ImageDraw.Draw(img)
    # Draw a rough pothole shape
    draw.ellipse([120, 80, 200, 160], fill=(30, 30, 30), outline=(50, 50, 50))
    draw.text((10, 10), "RoadGuard Test Image", fill=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _pass(route: str, status: int, ms: float, note: str = "") -> dict:
    note_str = f"  {CYAN}{note}{RESET}" if note else ""
    print(f"  {GREEN}✅ {route:<28}{RESET}  {status}  ({ms:.0f}ms){note_str}")
    return {"route": route, "ok": True}


def _fail(route: str, status: int | str, ms: float, reason: str = "") -> dict:
    print(f"  {RED}❌ {route:<28}{RESET}  {status}  ({ms:.0f}ms)  {reason}")
    return {"route": route, "ok": False}


def _warn(route: str, status: int, ms: float, note: str = "") -> dict:
    print(f"  {YELLOW}⚠️  {route:<27}{RESET}  {status}  ({ms:.0f}ms)  {note}")
    return {"route": route, "ok": True}  # warn = pass with note


def _get(path: str, params: dict = None):
    t = time.perf_counter()
    r = requests.get(
        f"{BASE_URL}{path}",
        headers=HEADERS,
        params=params,
        timeout=TIMEOUT,
    )
    ms = (time.perf_counter() - t) * 1000
    return r, ms


def _post(path: str, json: dict = None, files=None, data: dict = None):
    t = time.perf_counter()
    r = requests.post(
        f"{BASE_URL}{path}",
        headers=HEADERS,
        json=json,
        files=files,
        data=data,
        timeout=TIMEOUT,
    )
    ms = (time.perf_counter() - t) * 1000
    return r, ms


def _patch(path: str):
    t = time.perf_counter()
    r = requests.patch(f"{BASE_URL}{path}", headers=HEADERS, timeout=TIMEOUT)
    ms = (time.perf_counter() - t) * 1000
    return r, ms


# ── CHECKS ────────────────────────────────────────────

def check_health(results: list):
    print(f"\n{BOLD}── Core ──────────────────────────────────────{RESET}")
    try:
        r, ms = _get("/health")
        if r.status_code == 200:
            body = r.json()
            mongo = body.get("mongodb_uri_loaded", False)
            wx    = body.get("openweather_key_loaded", False)
            note  = f"mongo={'✅' if mongo else '❌'}  weather={'✅' if wx else '❌'}"
            results.append(_pass("GET /health", r.status_code, ms, note))
        else:
            results.append(_fail("GET /health", r.status_code, ms))
    except Exception as e:
        results.append(_fail("GET /health", "ERR", 0, str(e)))


def check_upload(results: list) -> str | None:
    """Returns pothole_id if upload succeeded, else None."""
    print(f"\n{BOLD}── Detection ─────────────────────────────────{RESET}")
    try:
        img_bytes = _dummy_image_bytes()
        files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
        data  = {"lat": str(TEST_LAT), "lon": str(TEST_LON), "speed_kmh": "30"}

        r, ms = _post("/upload", files=files, data=data)

        if r.status_code == 200:
            body        = r.json()
            detections  = body.get("detections", 0)
            pothole_ids = body.get("pothole_ids", [])
            pid         = pothole_ids[0] if pothole_ids else None

            if detections > 0 and pid:
                results.append(_pass(
                    "POST /upload", r.status_code, ms,
                    f"detections={detections}  id={pid[:12]}…",
                ))
                return pid
            else:
                results.append(_warn(
                    "POST /upload", r.status_code, ms,
                    "No pothole detected in synthetic image — YOLO working, "
                    "just no match (expected for a grey test image)",
                ))
                return None
        else:
            results.append(_fail("POST /upload", r.status_code, ms, r.text[:80]))
            return None
    except Exception as e:
        results.append(_fail("POST /upload", "ERR", 0, str(e)))
        return None


def check_alert(results: list):
    print(f"\n{BOLD}── Alert ─────────────────────────────────────{RESET}")
    try:
        r, ms = _get("/alert", params={
            "lat": TEST_LAT, "lon": TEST_LON, "speed_kmh": 30
        })
        if r.status_code == 200:
            body  = r.json()
            alert = body.get("alert", False)
            note  = f"alert={alert}"
            if alert:
                note += f"  zone={body.get('zone')}  dist={body.get('distance_m')}m"
            results.append(_pass("GET /alert", r.status_code, ms, note))
        else:
            results.append(_fail("GET /alert", r.status_code, ms, r.text[:80]))
    except Exception as e:
        results.append(_fail("GET /alert", "ERR", 0, str(e)))


def check_alert_session(results: list) -> str | None:
    """Returns session_id if start succeeded, else None."""
    print(f"\n{BOLD}── Session ───────────────────────────────────{RESET}")

    session_id = None

    # /alert/start
    try:
        r, ms = _post("/alert/start", json={
            "lat": TEST_LAT, "lon": TEST_LON,
            "speed_kmh": 30, "weather": "dry",
        })
        if r.status_code == 200:
            body       = r.json()
            session_id = body.get("session_id")
            results.append(_pass(
                "POST /alert/start", r.status_code, ms,
                f"session={session_id[:12] if session_id else 'None'}…",
            ))
        else:
            results.append(_fail("POST /alert/start", r.status_code, ms, r.text[:80]))
    except Exception as e:
        results.append(_fail("POST /alert/start", "ERR", 0, str(e)))

    # /alert/stop
    try:
        stop_id = session_id or "local_000000000000"
        r, ms = _post("/alert/stop", json={
            "session_id": stop_id,
            "end_lat": TEST_LAT,
            "end_lon": TEST_LON,
        })
        if r.status_code == 200:
            body = r.json()
            results.append(_pass(
                "POST /alert/stop", r.status_code, ms,
                f"duration={body.get('duration_sec')}s",
            ))
        else:
            results.append(_fail("POST /alert/stop", r.status_code, ms, r.text[:80]))
    except Exception as e:
        results.append(_fail("POST /alert/stop", "ERR", 0, str(e)))

    return session_id


def check_simulate(results: list, pothole_id: str | None):
    print(f"\n{BOLD}── Simulation ────────────────────────────────{RESET}")

    # Use uploaded ID if available, else the hardcoded fallback
    pid = pothole_id or "6a7d97d7d97b2fc1c888747c"

    for step in range(6):
        try:
            r, ms = _get("/simulate/step", params={
                "pothole_id": pid,
                "step": step,
                "condition": "dry",
            })
            if r.status_code == 200:
                body  = r.json()
                alert = body.get("alert", False)
                note  = f"step={step}  alert={alert}"
                if alert:
                    note += f"  dist={body.get('distance_m')}m"
                results.append(_pass(
                    f"GET /simulate/step?step={step}",
                    r.status_code, ms, note,
                ))
            else:
                results.append(_fail(
                    f"GET /simulate/step?step={step}",
                    r.status_code, ms, r.text[:60],
                ))
        except Exception as e:
            results.append(_fail(
                f"GET /simulate/step?step={step}", "ERR", 0, str(e),
            ))


def check_reports(results: list, pothole_id: str | None):
    print(f"\n{BOLD}── Reports ───────────────────────────────────{RESET}")

    # GET /reports
    try:
        r, ms = _get("/reports")
        if r.status_code == 200:
            body  = r.json()
            count = body.get("count", 0)
            results.append(_pass("GET /reports", r.status_code, ms, f"count={count}"))
        else:
            results.append(_fail("GET /reports", r.status_code, ms, r.text[:80]))
    except Exception as e:
        results.append(_fail("GET /reports", "ERR", 0, str(e)))

    # PATCH /reports/{id}/fix  — only if we have a real ID from upload
    if pothole_id:
        try:
            r, ms = _patch(f"/reports/{pothole_id}/fix")
            if r.status_code == 200:
                body   = r.json()
                status = body.get("status")
                results.append(_pass(
                    "PATCH /reports/{id}/fix",
                    r.status_code, ms, f"status={status}",
                ))
            else:
                results.append(_fail(
                    "PATCH /reports/{id}/fix",
                    r.status_code, ms, r.text[:80],
                ))
        except Exception as e:
            results.append(_fail("PATCH /reports/{id}/fix", "ERR", 0, str(e)))
    else:
        print(f"  {YELLOW}⚠️  PATCH /reports/{{id}}/fix{RESET}  — skipped (no pothole_id from upload)")


# ── MAIN ──────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════╗")
    print(f"║     RoadGuard AI — Pre-flight Check          ║")
    print(f"║     {BASE_URL[:44]}  ║")
    print(f"╚══════════════════════════════════════════════╝{RESET}")

    results    = []
    pothole_id = None

    check_health(results)
    pothole_id = check_upload(results)
    check_alert(results)
    check_alert_session(results)
    check_simulate(results, pothole_id)
    check_reports(results, pothole_id)

    # ── Summary ───────────────────────────────────────
    total  = len(results)
    passed = sum(1 for r in results if r["ok"])
    failed = total - passed

    print(f"\n{BOLD}{'─' * 48}{RESET}")

    if failed == 0:
        print(f"  {GREEN}{BOLD}{passed}/{total} checks passed — ready for testing 🚀{RESET}")
    else:
        print(f"  {RED}{BOLD}{failed}/{total} checks failed{RESET}")
        print(f"\n  {RED}Failed routes:{RESET}")
        for r in results:
            if not r["ok"]:
                print(f"    • {r['route']}")
        print(f"\n  Fix these before going outside.")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
