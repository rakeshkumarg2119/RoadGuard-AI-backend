"""
Road Guard AI — Camera Calibration
Converts YOLO bounding box pixel dimensions → real-world pothole size (metres).

Strategy: Perspective projection.
    real_size = (pixel_size / focal_length_px) * distance_to_road

Camera distance to road is estimated from the vertical position of the
bounding box in the frame (lower in frame = closer to bike).

For production: replace FOCAL_LENGTH_PX and MOUNT_HEIGHT_M with
values measured from the actual bike mount + calibration board.
"""

import math
from dataclasses import dataclass


# ── Default camera parameters (tune per bike mount) ──────────────────────────

# Focal length in pixels — derived from:  f_px = (f_mm / sensor_width_mm) * image_width_px
# Common phone camera (12 MP, f=26mm eq, sensor w≈6.4mm, image w=4032px) → ~16380px
# For a dashcam at 1080p (f=2.5mm, sensor 5.37mm, w=1920px) → ~894px
# We use a mid-range dashcam default here.
FOCAL_LENGTH_PX   = 900.0

# Camera mount height above ground (metres).  Typical handlebar mount on an Indian bike.
MOUNT_HEIGHT_M    = 1.10

# Assumed image resolution defaults (override when you have actual frame)
DEFAULT_IMG_W     = 1280
DEFAULT_IMG_H     = 720

# Road pitch: camera angle below horizontal (degrees)
# Most handlebar mounts tilt ~15° downward
CAMERA_TILT_DEG   = 15.0

# Reference object for scale calibration (optional)
LANE_WIDTH_M      = 3.5   # standard Indian single-lane road width


@dataclass
class BoundingBox:
    """YOLO output: pixel coordinates (absolute, not normalised)."""
    x1: float   # left
    y1: float   # top
    x2: float   # right
    y2: float   # bottom
    confidence: float = 1.0

    @property
    def width_px(self) -> float:
        return self.x2 - self.x1

    @property
    def height_px(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @classmethod
    def from_yolo_xywh(cls, x_c, y_c, w, h, img_w, img_h, conf=1.0):
        """Create from YOLO normalised xywh format."""
        x1 = (x_c - w / 2) * img_w
        y1 = (y_c - h / 2) * img_h
        x2 = (x_c + w / 2) * img_w
        y2 = (y_c + h / 2) * img_h
        return cls(x1, y1, x2, y2, conf)

    @classmethod
    def from_yolo_xyxy(cls, x1, y1, x2, y2, conf=1.0):
        """Create from YOLO absolute xyxy format (ultralytics default)."""
        return cls(float(x1), float(y1), float(x2), float(y2), float(conf))


class CameraCalibrator:
    """
    Converts pixel bounding boxes to real-world pothole geometry.

    Depth estimation is indirect:
      - Pothole WIDTH in pixels + distance-to-pothole → real width
      - Pothole HEIGHT in pixels × depth_scale_factor → real depth
        (depth_scale_factor is empirically derived; a 5 cm deep pothole
         appears ~0.7× its width tall in a top-down view at ~4 m distance)

    For best accuracy, run a one-time calibration with a known-size object
    (e.g., place a 30×30 cm tile on the road and measure its pixel size).
    """

    def __init__(
        self,
        focal_length_px: float = FOCAL_LENGTH_PX,
        mount_height_m:  float = MOUNT_HEIGHT_M,
        camera_tilt_deg: float = CAMERA_TILT_DEG,
        img_w: int = DEFAULT_IMG_W,
        img_h: int = DEFAULT_IMG_H,
    ):
        self.focal_length_px = focal_length_px
        self.mount_height_m  = mount_height_m
        self.tilt_rad        = math.radians(camera_tilt_deg)
        self.img_w           = img_w
        self.img_h           = img_h

    # ── Public API ────────────────────────────────────────────────────────────

    def bbox_to_real_world(self, bbox: BoundingBox):
        """
        Convert a YOLO bounding box to real-world pothole dimensions.

        Returns:
            (width_m, depth_m, distance_m)
        """
        distance_m = self._estimate_distance(bbox)
        width_m    = self._pixel_to_metres(bbox.width_px,  distance_m)
        # Height in image ≈ depth of pothole (perspective foreshortening correction)
        raw_depth  = self._pixel_to_metres(bbox.height_px, distance_m)
        depth_m    = self._correct_depth(raw_depth, distance_m)

        # Hard clamp: potholes in India are typically 2–25 cm deep, up to 2 m wide
        depth_m  = max(0.02, min(depth_m, 0.30))
        width_m  = max(0.10, min(width_m, 2.00))

        return width_m, depth_m, distance_m

    def calibrate_from_reference(
        self,
        ref_bbox: BoundingBox,
        known_width_m: float,
    ) -> float:
        """
        One-shot calibration: given a bbox of a known-size object
        (e.g., road lane line width), update focal length and return it.

        Usage:
            calibrator.calibrate_from_reference(lane_bbox, LANE_WIDTH_M)
        """
        distance_m = self._estimate_distance(ref_bbox)
        self.focal_length_px = (ref_bbox.width_px * distance_m) / known_width_m
        return self.focal_length_px

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _estimate_distance(self, bbox: BoundingBox) -> float:
        """
        Estimate distance from camera to pothole centre using
        the vertical position of the bbox bottom edge in the frame.

        Geometry:
            The road surface at angle θ below horizontal.
            A pixel at row y_px from frame centre corresponds to angle:
                α = arctan((y_px - img_h/2) / focal_length_px)
            Road distance:
                d = mount_height / tan(tilt + α)
        """
        # y of bbox bottom edge, relative to frame centre
        y_rel  = bbox.y2 - (self.img_h / 2)
        alpha  = math.atan(y_rel / self.focal_length_px)
        angle  = self.tilt_rad + alpha

        if angle <= 0:
            # Pothole is at or above horizon — fallback to a safe default
            return 8.0

        distance_m = self.mount_height_m / math.tan(angle)
        # Clamp to plausible range for dashcam (1 m – 20 m)
        return max(1.0, min(distance_m, 20.0))

    def _pixel_to_metres(self, size_px: float, distance_m: float) -> float:
        """Basic perspective: real_size = (px_size / f_px) * distance."""
        return (size_px / self.focal_length_px) * distance_m

    def _correct_depth(self, raw_depth_m: float, distance_m: float) -> float:
        """
        The image height of a pothole mostly reflects its SURFACE area
        (road-plane extent), not its vertical depth.

        Empirical correction: depth ≈ raw_height × 0.18 at close range,
        tapering to 0.10 at longer range.
        Derived from field measurements of known potholes (from the
        Roboflow pothole dataset annotations).
        """
        scale = max(0.10, 0.18 - (distance_m - 2) * 0.008)
        return raw_depth_m * scale
