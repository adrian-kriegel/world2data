#!/usr/bin/env python3
"""Quick viewer for merged .npy point clouds produced by particle filter."""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import open3d as o3d
except Exception as exc:  # pragma: no cover - CLI helper
    raise SystemExit(
        "open3d is required to view point clouds. Install deps from pyproject.toml."
    ) from exc


def _discover_merged_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        {
            path
            for path in root.rglob("*_merged_track_point_clouds")
            if path.is_dir()
        }
    )


def _iter_npy_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.npy")))
        elif path.is_file() and path.suffix.lower() == ".npy":
            files.append(path)
    return files


def _color_for_index(index: int) -> np.ndarray:
    hue = (index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return np.array([r, g, b], dtype=np.float32)


def _load_cloud(path: Path, *, max_points: int | None) -> np.ndarray:
    points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path} does not contain an Nx3 point array (got {points.shape})")
    if max_points is not None and len(points) > max_points:
        idx = np.random.choice(len(points), size=max_points, replace=False)
        points = points[idx]
    return points.astype(np.float32, copy=False)


def _build_geometry(files: Sequence[Path], *, max_points: int | None) -> o3d.geometry.PointCloud:
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for index, path in enumerate(files):
        points = _load_cloud(path, max_points=max_points)
        if len(points) == 0:
            continue
        color = _color_for_index(index)
        all_points.append(points)
        all_colors.append(np.repeat(color[None, :], len(points), axis=0))
    if not all_points:
        raise SystemExit("No points found in the selected .npy files.")
    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    return pcd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View merged track point clouds (.npy).")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output"),
        help="Root directory to search (default: output).",
    )
    parser.add_argument(
        "--cloud-dir",
        type=Path,
        action="append",
        default=[],
        help="Merged point cloud dir to view (can be repeated).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="Max points per track (default: 200000).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list discovered merged point cloud directories.",
    )
    return parser.parse_args()


def _resolve_roots(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    if args.cloud_dir:
        roots.extend(args.cloud_dir)
    else:
        root = args.root
        if not root.exists() and root.name == "output":
            alt = root.with_name("outputs")
            if alt.exists():
                root = alt
        roots.append(root)
    return roots


def _main() -> None:
    args = _parse_args()
    roots = _resolve_roots(args)
    merged_dirs: list[Path] = []
    for root in roots:
        merged_dirs.extend(_discover_merged_dirs(root))
    if args.list_only:
        if not merged_dirs:
            print("No merged point cloud directories found.")
        else:
            for path in merged_dirs:
                print(path.resolve())
        return
    if not merged_dirs:
        raise SystemExit("No merged point cloud directories found. Use --list-only to confirm.")

    files = _iter_npy_files(merged_dirs)
    if not files:
        raise SystemExit("No .npy files found in merged point cloud directories.")

    pcd = _build_geometry(files, max_points=args.max_points)
    o3d.visualization.draw_geometries([pcd], window_name="Merged track point clouds")


if __name__ == "__main__":
    _main()

