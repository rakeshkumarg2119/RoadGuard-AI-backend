"""
RoadGuard AI — /upload route

Accepts: multipart/form-data
  - image: UploadFile  (JPEG/PNG from phone camera)
  - lat:   float
  - lon:   float
  - speed_kmh: float (optional, default 30)

Pipeline:
  image bytes → numpy array → PotholeDetector
  → SimulationStore.save() with current weather
  → response with detections + pothole_id
"""

import io
import logging
import uuid
from pathlib import Path

import numpy as np

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from PIL import Image, ImageDraw, ImageFont

from core.app_state import get_detector, get_store, get_weather

router = APIRouter()
logger = logging.getLogger("roadguard.routes.upload")

# Where annotated (boxed) images are saved so the dashboard/app can view them.
# Mounted as static files at /static in main.py.
ANNOTATED_DIR = Path(__file__).parent.parent / "static" / "annotated"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# Bundled fonts folder — optional. Only needed if you deploy on Linux later
# and want a consistent look. On Windows this isn't required at all; see
# _load_font() below, which uses the system Consolas/Arial fonts instead.
FONT_DIR = Path(__file__).parent.parent / "fonts"

_SEVERITY_COLOR = {
    "low":      (76, 175, 80),
    "medium":   (255, 152, 0),
    "high":     (244, 67, 54),
    "critical": (183, 28, 28),
}


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _load_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    """
    Tries a list of font file candidates in order (Windows system fonts,
    then anything bundled in FONT_DIR, then Linux system paths) and
    returns the first one that loads. Falls back to PIL's built-in
    bitmap font if none of them exist — never raises.
    """
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _annotate_image(
    pil_img: Image.Image,
    detections: list[dict],
    gps: dict,
    speed_kmh: float,
    weather: dict | None,
) -> tuple[str, str]:
    """
    Draws each detection's bbox on the photo, then appends a full data
    panel below it listing EVERY computed value — detection, kinematics,
    impact, damage probabilities, injury risk, alert — plus the upload's
    GPS/speed/weather context. Saves as PNG locally.

    Returns (absolute_disk_path, url_path).
    """
    photo = pil_img.copy()
    draw = ImageDraw.Draw(photo)

    # Bold sans, for the on-photo severity tag.
    font_label = _load_font([
        "C:/Windows/Fonts/arialbd.ttf",              # Windows: Arial Bold
        str(FONT_DIR / "DejaVuSans-Bold.ttf"),        # bundled fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    ], 18)

    # Bold monospace, for panel section headers.
    font_panel_h = _load_font([
        "C:/Windows/Fonts/consolab.ttf",              # Windows: Consolas Bold
        str(FONT_DIR / "DejaVuSansMono-Bold.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ], 20)

    # Regular monospace, for panel body text.
    font_panel = _load_font([
        "C:/Windows/Fonts/consola.ttf",               # Windows: Consolas Regular
        str(FONT_DIR / "DejaVuSansMono.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ], 16)

    # ── 1. Draw bbox + short on-photo tag for each detection ────────────
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["detection"]["bbox_px"]
        severity = d["physics"]["impact"]["severity"]
        color = _SEVERITY_COLOR.get(severity, (255, 193, 61))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        tag = f"#{i+1} {severity.upper()}"
        tag_bbox = draw.textbbox((x1, y1), tag, font=font_label)
        tag_h = tag_bbox[3] - tag_bbox[1]
        tag_y = max(0, y1 - tag_h - 8)
        draw.rectangle([x1, tag_y, tag_bbox[2] + 8, tag_y + tag_h + 8], fill=color)
        draw.text((x1 + 4, tag_y + 4), tag, fill=(255, 255, 255), font=font_label)

    # ── 2. Build the full data-panel text, per detection ─────────────────
    def block_lines(i: int, d: dict) -> list[str]:
        det    = d["detection"]
        phys   = d["physics"]
        poth   = phys["pothole"]
        kin    = phys["kinematics"]
        imp    = phys["impact"]
        alert  = d["alert"]
        damage = phys.get("damage", {})
        injury = phys.get("injury_risk", {})

        lines = [f"── DETECTION #{i+1} " + "─" * 40]
        lines.append(f"bbox_px: {det['bbox_px']}   source: {det['source']}   inference: {d['inference_ms']}ms")
        lines.append("")
        lines.append("POTHOLE GEOMETRY")
        lines.append(
            f"  width: {poth['width_m']}m   depth: {poth['depth_m']}m   "
            f"area: {poth['area_m2']}m2   yolo_confidence: {poth['confidence']}"
        )
        lines.append("")
        lines.append(f"KINEMATICS  (speed={phys['speed_kmh']} km/h, road={phys['road_condition']})")
        lines.append(
            f"  d_react: {kin['d_react_m']}m   d_stop: {kin['d_stop_m']}m   "
            f"d_alert: {kin['d_alert_m']}m   distance_to_pothole: {det['distance_m']}m"
        )
        lines.append("")
        lines.append("IMPACT")
        lines.append(
            f"  energy: {imp['energy_j']} J   fall_type: {imp['fall_type']}   severity: {imp['severity']}"
        )
        lines.append("")
        if damage:
            lines.append("BIKE DAMAGE PROBABILITY")
            lines.append("  " + "   ".join(f"{k}: {_fmt_pct(v)}" for k, v in damage.items()))
            lines.append("")
        if injury:
            lines.append("RIDER INJURY RISK")
            lines.append("  " + "   ".join(f"{k}: {_fmt_pct(v)}" for k, v in injury.items()))
            lines.append("")
        lines.append("ALERT")
        lines.append(
            f"  trigger: {alert['trigger']}   danger: {alert['danger']}   "
            f"sound: {alert['sound_type']}   vibration: {alert['vibration_type']}"
        )
        lines.append(f"  \"{alert['message']}\"")
        return lines

    header_lines = [
        f"UPLOAD  gps: {gps['lat']}, {gps['lon']}   speed: {speed_kmh} km/h   "
        f"weather: {weather.get('description') if weather else 'n/a'}",
        "=" * 60,
    ]

    all_lines = list(header_lines)
    for i, d in enumerate(detections):
        all_lines += block_lines(i, d)
        all_lines.append("")

    line_h = 22
    pad = 20
    panel_h = pad * 2 + line_h * len(all_lines)

    # Panel must be wide enough for its longest line — narrow source photos
    # would otherwise clip text on the right.
    tmp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_text_w = max(
        tmp_draw.textbbox((0, 0), line, font=font_panel_h)[2] for line in all_lines
    )
    panel_w = max(photo.width, max_text_w + pad * 2)

    # ── 3. Compose photo + panel into one PNG ────────────────────────────
    canvas = Image.new("RGB", (panel_w, photo.height + panel_h), (14, 16, 18))
    canvas.paste(photo, (0, 0))
    pdraw = ImageDraw.Draw(canvas)

    y = photo.height + pad
    for line in all_lines:
        f = font_panel_h if line.startswith("UPLOAD") or line.startswith("──") else font_panel
        color = (255, 197, 61) if line.startswith("──") else (231, 228, 220)
        pdraw.text((pad, y), line, fill=color, font=f)
        y += line_h

    filename = f"{uuid.uuid4().hex}.png"
    out_path = ANNOTATED_DIR / filename
    canvas.save(out_path, "PNG")

    return str(out_path.resolve()), f"/static/annotated/{filename}"


@router.post("/upload")
async def upload_image(
    image: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    speed_kmh: float = Form(30.0),
):
    """
    Receive a photo + GPS from Flutter, run YOLO, store result.
    Returns detections list and pothole_id (or null if no pothole).
    """

    # ── Read image bytes → numpy BGR array ──────────────────
    try:
        raw = await image.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        # YOLO expects BGR (cv2 format)
        frame = np.array(pil_img)[:, :, ::-1].copy()
    except Exception as exc:
        logger.error("Image decode failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    # ── Get services ─────────────────────────────────────────
    detector = get_detector()
    store    = get_store()
    weather  = get_weather()

    current_weather = weather.get_current_sync()  # WeatherCondition or None

    # ── Road condition from weather ──────────────────────────
    from core.physics_engine import RoadCondition
    road_condition = RoadCondition.DRY
    if current_weather is not None:
        road_condition = current_weather.road_condition

    # ── Run detection ────────────────────────────────────────
    try:
        detections = detector.process_frame(
            frame=frame,
            speed_kmh=speed_kmh,
            road_condition=road_condition,
            gps_coords=(lat, lon),
        )
    except Exception as exc:
        logger.error("Detection failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Detection error: {exc}")

    # ── Save each detection to MongoDB ───────────────────────
    pothole_ids = []
    for detection in detections:
        try:
            pid = await store.save(detection, weather_condition=current_weather)
            if pid:
                pothole_ids.append(pid)
        except Exception as exc:
            logger.warning("Save failed for detection: %s", exc)

    # ── Draw bbox + full physics data panel onto the image ───
    # Only worth doing if something was actually detected.
    annotated_image_path = None
    annotated_image_url = None
    if detections:
        try:
            annotated_image_path, annotated_image_url = _annotate_image(
                pil_img,
                detections,
                gps={"lat": lat, "lon": lon},
                speed_kmh=speed_kmh,
                weather=current_weather.to_dict() if current_weather else None,
            )
        except Exception as exc:
            logger.warning("Annotation failed: %s", exc)

    # ── Response ─────────────────────────────────────────────
    return {
        "status":               "ok",
        "gps":                  {"lat": lat, "lon": lon},
        "speed_kmh":            speed_kmh,
        "detections":           len(detections),
        "pothole_ids":          pothole_ids,
        "annotated_image_url":   annotated_image_url,
        "annotated_image_path":  annotated_image_path,
        "results": [
            {
                "pothole_id":   pothole_ids[i] if i < len(pothole_ids) else None,
                "bbox_px":      d["detection"]["bbox_px"],
                "confidence":   d["detection"]["confidence"],
                "distance_m":   d["detection"]["distance_m"],
                "severity":     d["physics"]["impact"]["severity"],
                "fall_type":    d["physics"]["impact"]["fall_type"],
                "d_alert_m":    d["physics"]["kinematics"]["d_alert_m"],
                "alert":        d["alert"],
                "inference_ms": d["inference_ms"],
                # Full calculation trail — previously computed but never
                # returned. See physics_engine.PhysicsResult.to_dict().
                "physics":      d["physics"],
            }
            for i, d in enumerate(detections)
            ],
        "weather":      current_weather.to_dict() if current_weather else None,
    }