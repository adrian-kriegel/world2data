from __future__ import annotations

"""Shared data model and component interfaces for particle tracking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from random import Random
from typing import Mapping, Sequence


@dataclass(frozen=True)
class AABB2D:
    """Axis-aligned box in image space (pixels)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("AABB2D requires x_max > x_min")
        if self.y_max <= self.y_min:
            raise ValueError("AABB2D requires y_max > y_min")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.width * 0.5, self.y_min + self.height * 0.5)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "AABB2D") -> float:
        ix_min = max(self.x_min, other.x_min)
        iy_min = max(self.y_min, other.y_min)
        ix_max = min(self.x_max, other.x_max)
        iy_max = min(self.y_max, other.y_max)

        inter_w = max(0.0, ix_max - ix_min)
        inter_h = max(0.0, iy_max - iy_min)
        intersection = inter_w * inter_h
        if intersection <= 0.0:
            return 0.0

        union = self.area + other.area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union


@dataclass(frozen=True)
class Detection2D:
    """2D detector output (mock YOLO in tests)."""

    label: str
    aabb: AABB2D
    confidence: float = 1.0


@dataclass(frozen=True)
class CameraIntrinsics:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float


@dataclass(frozen=True)
class CameraPose:
    """World-to-camera extrinsics: x_cam = R * x_world + t."""

    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    translation: tuple[float, float, float]

    @staticmethod
    def identity() -> "CameraPose":
        return CameraPose(
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(0.0, 0.0, 0.0),
        )


@dataclass(frozen=True)
class ParticleState:
    """Particle state requested in README: x,y,z,bounding_box,mass,velocity."""

    position: tuple[float, float, float]
    bounding_box: tuple[float, float, float]
    mass: float
    velocity: tuple[float, float, float]


@dataclass(frozen=True)
class Particle:
    particle_id: str
    track_id: str
    label: str
    state: ParticleState
    weight: float = 1.0
    age: int = 0

    def evolved(
        self,
        *,
        state: ParticleState | None = None,
        weight: float | None = None,
        age: int | None = None,
        particle_id: str | None = None,
    ) -> "Particle":
        return replace(
            self,
            state=state if state is not None else self.state,
            weight=weight if weight is not None else self.weight,
            age=age if age is not None else self.age,
            particle_id=particle_id if particle_id is not None else self.particle_id,
        )


@dataclass(frozen=True)
class FrameContext:
    frame_index: int
    timestamp_s: float
    dt_s: float
    camera_pose: CameraPose
    camera_intrinsics: CameraIntrinsics
    detections: Sequence[Detection2D]


class MotionModel(ABC):
    @abstractmethod
    def propagate(self, particle: Particle, frame: FrameContext, rng: Random) -> Particle:
        """Propagate one particle forward by one frame."""


class CostFunction(ABC):
    @abstractmethod
    def score(
        self,
        particle: Particle,
        detection: Detection2D,
        frame: FrameContext,
    ) -> float:
        """Likelihood score for particle given one detection."""


class Resampler(ABC):
    @abstractmethod
    def resample(self, particles: Sequence[Particle], count: int, rng: Random) -> list[Particle]:
        """Resample particles according to current particle weights."""


@dataclass(frozen=True)
class TrackEstimate:
    track_id: str
    label: str
    position: tuple[float, float, float]
    bounding_box: tuple[float, float, float]
    mass: float
    velocity: tuple[float, float, float]
    particle_count: int


@dataclass(frozen=True)
class FilterResult:
    frame_index: int
    estimates: Mapping[str, TrackEstimate]
    particles_by_track: Mapping[str, tuple[Particle, ...]]
