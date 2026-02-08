from __future__ import annotations

import numpy as np
import pytest

from world2data.point_cloud import NumpyPointCloudOps, create_point_cloud_ops


def test_create_point_cloud_ops_prefers_pcl_but_fallback_is_valid() -> None:
    selection = create_point_cloud_ops(requested_backend="pcl", require_backend=False)
    assert selection.requested_backend == "pcl"
    assert selection.backend.backend_name in {"pcl", "torch", "numpy"}


def test_numpy_point_cloud_ops_crop_merge_and_nn() -> None:
    ops = NumpyPointCloudOps()

    points = [
        (0.0, 0.0, 0.0),
        (0.1, 0.1, 0.1),
        (2.0, 2.0, 2.0),
    ]
    cropped = ops.crop_aabb(points, center=(0.0, 0.0, 0.0), size_xyz=(0.5, 0.5, 0.5))
    assert len(cropped) == 2

    merged = ops.merge_downsample(
        np.asarray([(0.0, 0.0, 0.0), (0.01, 0.01, 0.01)], dtype=np.float64),
        np.asarray([(0.0, 0.0, 0.0), (0.4, 0.4, 0.4)], dtype=np.float64),
        voxel_size_m=0.05,
        max_points=10,
    )
    assert len(merged) == 2

    nn = ops.nearest_neighbor_distances(
        np.asarray([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], dtype=np.float64),
        np.asarray([(0.1, 0.0, 0.0)], dtype=np.float64),
    )
    assert nn.shape == (2,)
    assert nn[0] < nn[1]


def test_numpy_point_cloud_ops_icp_aligns_translated_cloud() -> None:
    ops = NumpyPointCloudOps()

    target = np.asarray(
        [
            (-0.4, -0.3, 7.6),
            (-0.4, 0.3, 7.6),
            (0.4, -0.3, 7.6),
            (0.4, 0.3, 7.6),
            (-0.4, -0.3, 8.4),
            (-0.4, 0.3, 8.4),
            (0.4, -0.3, 8.4),
            (0.4, 0.3, 8.4),
        ],
        dtype=np.float64,
    )
    source = target + np.asarray([0.22, -0.10, 0.08], dtype=np.float64)

    before = float(np.mean(ops.nearest_neighbor_distances(source, target)))
    result = ops.register_icp(
        source,
        target,
        max_iterations=30,
        tolerance_m=1e-6,
        max_correspondence_distance_m=0.8,
    )
    after = float(
        np.mean(ops.nearest_neighbor_distances(result.transformed_source_points, target))
    )

    assert before > 0.10
    assert result.correspondence_ratio > 0.99
    assert result.mean_error_m < 1e-3
    assert after < 1e-3


def test_create_point_cloud_ops_require_backend_raises_if_missing() -> None:
    try:
        __import__("pcl")
        has_pcl = True
    except Exception:
        try:
            __import__("pclpy")
            has_pcl = True
        except Exception:
            has_pcl = False

    if has_pcl:
        pytest.skip("PCL binding installed; missing-backend behavior not applicable")

    with pytest.raises(ImportError):
        create_point_cloud_ops(requested_backend="pcl", require_backend=True)


def test_create_point_cloud_ops_torch_backend_is_available_or_falls_back() -> None:
    selection = create_point_cloud_ops(requested_backend="torch", require_backend=False)
    assert selection.requested_backend == "torch"
    assert selection.backend.backend_name in {"torch", "numpy"}
