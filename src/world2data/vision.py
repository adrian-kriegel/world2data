from __future__ import annotations

"""Projection/back-projection helpers used by the particle filter and tests."""

import math

import numpy as np

from .model import AABB2D, CameraIntrinsics, CameraPose


def _as_matrix3(rotation: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    return matrix


def _as_vec3(vector: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError("vector must be length 3")
    return arr


def world_to_camera(point_world: tuple[float, float, float], pose: CameraPose) -> np.ndarray:
    rotation = _as_matrix3(pose.rotation)
    translation = _as_vec3(pose.translation)
    return rotation @ _as_vec3(point_world) + translation


def _distort_normalized(
    x: float,
    y: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    if not (math.isfinite(x) and math.isfinite(y)):
        return (x, y)
    coeffs = intrinsics.distortion_coeffs
    if not coeffs:
        return (x, y)

    k1 = float(coeffs[0]) if len(coeffs) > 0 else 0.0
    k2 = float(coeffs[1]) if len(coeffs) > 1 else 0.0
    p1 = float(coeffs[2]) if len(coeffs) > 2 else 0.0
    p2 = float(coeffs[3]) if len(coeffs) > 3 else 0.0
    k3 = float(coeffs[4]) if len(coeffs) > 4 else 0.0

    r2 = x * x + y * y
    if not math.isfinite(r2) or r2 > 1e8:
        return (x, y)
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    if not math.isfinite(radial):
        return (x, y)

    x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    x_out = x * radial + x_tan
    y_out = y * radial + y_tan
    if not (math.isfinite(x_out) and math.isfinite(y_out)):
        return (x, y)
    return (x_out, y_out)


def _undistort_normalized(
    x_dist: float,
    y_dist: float,
    intrinsics: CameraIntrinsics,
    *,
    iterations: int = 6,
) -> tuple[float, float]:
    if not intrinsics.distortion_coeffs:
        return (x_dist, y_dist)

    x = float(x_dist)
    y = float(y_dist)
    for _ in range(max(1, int(iterations))):
        if not (math.isfinite(x) and math.isfinite(y)):
            return (x_dist, y_dist)
        x_proj, y_proj = _distort_normalized(x, y, intrinsics)
        x += (x_dist - x_proj)
        y += (y_dist - y_proj)
        if not (math.isfinite(x) and math.isfinite(y)):
            return (x_dist, y_dist)
    return (x, y)


def camera_to_world(point_camera: tuple[float, float, float], pose: CameraPose) -> np.ndarray:
    rotation = _as_matrix3(pose.rotation)
    translation = _as_vec3(pose.translation)
    return rotation.T @ (_as_vec3(point_camera) - translation)


def project_point(
    point_world: tuple[float, float, float],
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
) -> tuple[float, float, float] | None:
    point_camera = world_to_camera(point_world, pose)
    x_cam, y_cam, z_cam = (float(point_camera[0]), float(point_camera[1]), float(point_camera[2]))

    if z_cam <= 1e-6:
        return None

    x_norm = x_cam / z_cam
    y_norm = y_cam / z_cam
    x_dist, y_dist = _distort_normalized(x_norm, y_norm, intrinsics)

    u = intrinsics.fx_px * x_dist + intrinsics.cx_px
    v = intrinsics.fy_px * y_dist + intrinsics.cy_px
    return (u, v, z_cam)


def back_project_pixel(
    pixel: tuple[float, float],
    depth_m: float,
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
) -> tuple[float, float, float]:
    if depth_m <= 0.0:
        raise ValueError("depth_m must be > 0")

    x_dist = (pixel[0] - intrinsics.cx_px) / intrinsics.fx_px
    y_dist = (pixel[1] - intrinsics.cy_px) / intrinsics.fy_px
    x_norm, y_norm = _undistort_normalized(x_dist, y_dist, intrinsics)

    x_cam = x_norm * depth_m
    y_cam = y_norm * depth_m
    point_world = camera_to_world((x_cam, y_cam, depth_m), pose)
    return (float(point_world[0]), float(point_world[1]), float(point_world[2]))


def box3d_corners(
    center_world: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    hx, hy, hz = (size_xyz[0] * 0.5, size_xyz[1] * 0.5, size_xyz[2] * 0.5)
    cx, cy, cz = center_world
    corners: list[tuple[float, float, float]] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append((cx + sx * hx, cy + sy * hy, cz + sz * hz))
    return corners


def project_box_to_image_aabb(
    center_world: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    intrinsics: CameraIntrinsics,
    pose: CameraPose,
) -> AABB2D | None:
    projected = [project_point(corner, intrinsics, pose) for corner in box3d_corners(center_world, size_xyz)]
    if any(point is None for point in projected):
        return None

    us = [point[0] for point in projected if point is not None]
    vs = [point[1] for point in projected if point is not None]

    x_min = max(0.0, min(us))
    y_min = max(0.0, min(vs))
    x_max = min(float(intrinsics.width_px), max(us))
    y_max = min(float(intrinsics.height_px), max(vs))

    if x_max <= x_min or y_max <= y_min:
        return None
    return AABB2D(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def iou_2d(box_a: AABB2D, box_b: AABB2D) -> float:
    return box_a.iou(box_b)


def camera_pose_from_world_position(
    position_world: tuple[float, float, float],
    *,
    yaw_rad: float = 0.0,
) -> CameraPose:
    """Create world-to-camera pose for level camera (yaw only) at world position."""
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)

    # camera-to-world rotation (yaw around world Y), then invert for world-to-camera
    r_cw = np.asarray(
        (
            (cy, 0.0, sy),
            (0.0, 1.0, 0.0),
            (-sy, 0.0, cy),
        ),
        dtype=np.float64,
    )
    r_wc = r_cw.T
    t_wc = -r_wc @ np.asarray(position_world, dtype=np.float64)
    return CameraPose(
        rotation=(
            (float(r_wc[0, 0]), float(r_wc[0, 1]), float(r_wc[0, 2])),
            (float(r_wc[1, 0]), float(r_wc[1, 1]), float(r_wc[1, 2])),
            (float(r_wc[2, 0]), float(r_wc[2, 1]), float(r_wc[2, 2])),
        ),
        translation=(float(t_wc[0]), float(t_wc[1]), float(t_wc[2])),
    )
