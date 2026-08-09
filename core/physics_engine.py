"""
Road Guard AI — Physics Engine
Kinematic alert distance calculator for bikes.

Formula:
    d_alert = d_stop + d_react
    d_stop  = v² / (2 * μ * g)
    d_react = v * t_react

Pothole severity is estimated from real-world width/depth
derived from YOLO bounding box + camera calibration.
"""

import math
from dataclasses import dataclass, field
from enum import Enum


# ── Constants ────────────────────────────────────────────────────────────────

G = 9.81                  # gravitational acceleration (m/s²)

# Friction coefficients for bikes (front tyre on tarmac)
MU = {
    "dry":   0.70,
    "wet":   0.35,
    "gravel": 0.45,
}

REACTION_TIME = 3.0       # seconds (conservative: accounts for distraction)

# Bike-specific damage thresholds (impact energy in Joules)
BIKE_DAMAGE_THRESHOLDS = {
    "rim_buckle":     120,
    "fork_bend":      280,
    "handlebar_slip":  80,
    "frame_stress":   450,
}

# Rider mass defaults
DEFAULT_RIDER_MASS_KG  = 75
DEFAULT_BIKE_MASS_KG   = 110    # avg Indian commuter/sports bike (Hero, Bajaj, Royal Enfield)
WHEEL_RADIUS_M         = 0.285  # 17-inch tyre (common on Indian bikes)


# ── Data structures ───────────────────────────────────────────────────────────

class RoadCondition(str, Enum):
    DRY    = "dry"
    WET    = "wet"
    GRAVEL = "gravel"


class FallType(str, Enum):
    SAFE       = "safe"           # pothole minor, rider brakes in time
    CONTROLLED = "controlled"     # low speed tip / stall
    SIDE_SLIDE = "side_slide"     # mid-speed, bike lays down sideways
    OVER_BARS  = "over_bars"      # high speed, front wheel drops, rider launches


@dataclass
class PotholeGeometry:
    """Real-world dimensions of a detected pothole."""
    width_m:  float          # horizontal span (metres)
    depth_m:  float          # vertical depth  (metres)
    area_m2:  float = 0.0    # surface area (metres²)
    confidence: float = 1.0  # YOLO detection confidence

    def __post_init__(self):
        if self.area_m2 == 0.0:
            # Approximate as ellipse
            self.area_m2 = math.pi * (self.width_m / 2) * (self.depth_m / 2)


@dataclass
class BikeConfig:
    """Vehicle parameters — default Indian commuter bike."""
    rider_mass_kg: float = DEFAULT_RIDER_MASS_KG
    bike_mass_kg:  float = DEFAULT_BIKE_MASS_KG
    wheel_radius_m: float = WHEEL_RADIUS_M

    @property
    def total_mass_kg(self) -> float:
        return self.rider_mass_kg + self.bike_mass_kg


@dataclass
class PhysicsResult:
    """Full output of one physics calculation pass."""
    # Inputs (echoed back)
    speed_kmh:       float
    road_condition:  str
    pothole:         PotholeGeometry

    # Kinematic distances
    d_react_m:       float      # distance covered during reaction time
    d_stop_m:        float      # braking distance
    d_alert_m:       float      # d_react + d_stop  ← the key output

    # Impact analysis
    impact_energy_j: float      # kinetic energy absorbed at pothole edge
    fall_type:       FallType

    # Bike damage flags
    damage: dict = field(default_factory=dict)

    # Rider injury risk (0–1 probability)
    injury_risk: dict = field(default_factory=dict)

    # Human-readable severity
    severity:    str = "low"    # low / medium / high / critical

    def to_dict(self) -> dict:
        return {
            "speed_kmh":      self.speed_kmh,
            "road_condition": self.road_condition,
            "pothole": {
                "width_m":    round(self.pothole.width_m, 3),
                "depth_m":    round(self.pothole.depth_m, 3),
                "area_m2":    round(self.pothole.area_m2, 4),
                "confidence": round(self.pothole.confidence, 3),
            },
            "kinematics": {
                "d_react_m": round(self.d_react_m, 2),
                "d_stop_m":  round(self.d_stop_m,  2),
                "d_alert_m": round(self.d_alert_m, 2),
            },
            "impact": {
                "energy_j":  round(self.impact_energy_j, 1),
                "fall_type": self.fall_type.value,
                "severity":  self.severity,
            },
            "damage":      self.damage,
            "injury_risk": self.injury_risk,
        }


# ── Core calculator ───────────────────────────────────────────────────────────

class PotholePhysicsEngine:
    """
    Stateless calculator.  Call calculate() for each detection event.
    """

    def __init__(self, bike: BikeConfig = None):
        self.bike = bike or BikeConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def calculate(
        self,
        speed_kmh: float,
        pothole: PotholeGeometry,
        road_condition: RoadCondition = RoadCondition.DRY,
    ) -> PhysicsResult:
        """
        Main entry point.

        Args:
            speed_kmh:      Current bike speed in km/h
            pothole:        Detected pothole geometry (real-world metres)
            road_condition: Surface friction category

        Returns:
            PhysicsResult with all computed values
        """
        v_ms = speed_kmh / 3.6                         # km/h → m/s
        mu   = MU[road_condition.value]

        d_react = self._reaction_distance(v_ms)
        d_stop  = self._braking_distance(v_ms, mu)
        d_alert = d_react + d_stop

        impact_energy = self._impact_energy(v_ms, pothole)
        fall_type     = self._classify_fall(speed_kmh, pothole, impact_energy)
        damage        = self._bike_damage(impact_energy, fall_type)
        injury_risk   = self._rider_injury_risk(speed_kmh, pothole, fall_type)
        severity      = self._severity_label(speed_kmh, pothole, impact_energy)

        return PhysicsResult(
            speed_kmh       = speed_kmh,
            road_condition  = road_condition.value,
            pothole         = pothole,
            d_react_m       = d_react,
            d_stop_m        = d_stop,
            d_alert_m       = d_alert,
            impact_energy_j = impact_energy,
            fall_type       = fall_type,
            damage          = damage,
            injury_risk     = injury_risk,
            severity        = severity,
        )

    # ── Kinematic helpers ─────────────────────────────────────────────────────

    def _reaction_distance(self, v_ms: float) -> float:
        """Distance covered before rider begins braking."""
        return v_ms * REACTION_TIME

    def _braking_distance(self, v_ms: float, mu: float) -> float:
        """Minimum stopping distance under constant deceleration."""
        return (v_ms ** 2) / (2 * mu * G)

    # ── Impact mechanics ──────────────────────────────────────────────────────

    def _impact_energy(self, v_ms: float, pothole: PotholeGeometry) -> float:
        """
        Energy (Joules) transferred to the bike when the front wheel
        strikes the pothole edge.

        Model: the wheel must lift over an effective obstacle of height h
        equal to the pothole depth (capped at wheel radius).
        E = ½mv² × (h / R)  — proportional energy fraction absorbed
        """
        h = min(pothole.depth_m, self.bike.wheel_radius_m)
        fraction = h / self.bike.wheel_radius_m
        kinetic  = 0.5 * self.bike.total_mass_kg * (v_ms ** 2)
        return kinetic * fraction

    # ── Fall classifier ───────────────────────────────────────────────────────

    def _classify_fall(
        self,
        speed_kmh: float,
        pothole: PotholeGeometry,
        impact_energy_j: float,
    ) -> FallType:
        """
        Rule-based fall type using speed and pothole depth.

        Thresholds tuned for Indian roads / commuter bikes:
          SAFE       : depth < 3 cm  OR  speed < 15 km/h
          CONTROLLED : depth < 6 cm  AND speed < 30 km/h
          SIDE_SLIDE : depth < 10 cm OR  speed < 50 km/h
          OVER_BARS  : everything else
        """
        d_cm = pothole.depth_m * 100

        if d_cm < 3 or speed_kmh < 15:
            return FallType.SAFE
        if d_cm < 6 and speed_kmh < 30:
            return FallType.CONTROLLED
        if d_cm < 10 or speed_kmh < 50:
            return FallType.SIDE_SLIDE
        return FallType.OVER_BARS

    # ── Damage & injury models ────────────────────────────────────────────────

    def _bike_damage(self, energy_j: float, fall_type: FallType) -> dict:
        """Return damage flags and probability (0–1) per bike component."""
        t = BIKE_DAMAGE_THRESHOLDS

        def prob(threshold: float, steepness: float = 0.008) -> float:
            # Sigmoid: P = 1 / (1 + e^(-k*(E - threshold)))
            return round(1 / (1 + math.exp(-steepness * (energy_j - threshold))), 3)

        damage = {
            "rim_buckle":      prob(t["rim_buckle"]),
            "fork_bend":       prob(t["fork_bend"]),
            "handlebar_slip":  prob(t["handlebar_slip"]),
            "frame_stress":    prob(t["frame_stress"]),
        }

        # OTB amplifies fork and handlebar damage
        if fall_type == FallType.OVER_BARS:
            damage["fork_bend"]      = min(1.0, damage["fork_bend"]      * 1.4)
            damage["handlebar_slip"] = min(1.0, damage["handlebar_slip"] * 1.3)

        return {k: round(v, 3) for k, v in damage.items()}

    def _rider_injury_risk(
        self,
        speed_kmh: float,
        pothole: PotholeGeometry,
        fall_type: FallType,
    ) -> dict:
        """
        Injury probability per body zone, indexed to fall type.
        Values are 0–1 (multiply by 100 for %).
        """
        spd  = min(speed_kmh / 100, 1.0)
        dep  = min(pothole.depth_m / 0.20, 1.0)   # normalised to 20 cm max

        if fall_type == FallType.SAFE:
            return {zone: 0.0 for zone in ["wrist", "head", "shoulder", "hip", "knee"]}

        if fall_type == FallType.CONTROLLED:
            return {
                "knee":   round(spd * 0.55 + dep * 0.15, 3),
                "ankle":  round(spd * 0.35 + dep * 0.10, 3),
                "hip":    round(spd * 0.25 + dep * 0.10, 3),
                "wrist":  round(spd * 0.20 + dep * 0.08, 3),
                "head":   round(spd * 0.10 + dep * 0.05, 3),
            }

        if fall_type == FallType.SIDE_SLIDE:
            return {
                "hip":      round(min(0.95, spd * 0.80 + dep * 0.15), 3),
                "elbow":    round(min(0.90, spd * 0.70 + dep * 0.15), 3),
                "knee":     round(min(0.85, spd * 0.65 + dep * 0.15), 3),
                "shoulder": round(min(0.80, spd * 0.55 + dep * 0.20), 3),
                "head":     round(min(0.65, spd * 0.40 + dep * 0.15), 3),
            }

        # OVER_BARS
        return {
            "wrist":      round(min(0.98, spd * 0.85 + dep * 0.13), 3),
            "head":       round(min(0.95, spd * 0.80 + dep * 0.15), 3),
            "shoulder":   round(min(0.92, spd * 0.75 + dep * 0.17), 3),
            "collarbone": round(min(0.88, spd * 0.70 + dep * 0.15), 3),
            "chest":      round(min(0.82, spd * 0.60 + dep * 0.18), 3),
        }

    def _severity_label(
        self,
        speed_kmh: float,
        pothole: PotholeGeometry,
        impact_energy_j: float,
    ) -> str:
        """
        Four severity bands derived from speed + depth combination.
        Energy thresholds validated against the actual output range:
          low      →  E < 600 J   (slow speed, shallow pothole)
          medium   →  E < 1800 J
          high     →  E < 4000 J
          critical →  E ≥ 4000 J
        """
        if impact_energy_j < 600:
            return "low"
        if impact_energy_j < 1800:
            return "medium"
        if impact_energy_j < 4000:
            return "high"
        return "critical"
