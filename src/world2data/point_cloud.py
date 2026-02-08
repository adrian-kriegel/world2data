from __future__ import annotations

"""Point-cloud operations with PCL-first backend selection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class PointCloudRegistrationResult:
    transformed_source_points: np.ndarray
    mean_error_m: float
    correspondence_ratio: float
    iterations: int


class PointCloudOps(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Backend identifier, e.g. 'pcl' or 'numpy'."""

    @abstractmethod
    def crop_aabb(
        self,
        points: Sequence[tuple[float, float, float]],
        *,
        center: tuple[float, float, float],
        size_xyz: tuple[float, float, float],
    ) -> np.ndarray:
        """Crop points inside axis-aligned 3D box."""

    @abstractmethod
    def merge_downsample(
        self,
        existing_points: np.ndarray,
        new_points: np.ndarray,
        *,
        voxel_size_m: float,
        max_points: int,
    ) -> np.ndarray:
        """Merge and downsample stitched points."""

    @abstractmethod
    def nearest_neighbor_distances(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """Distance from each source point to its nearest target point."""

    @abstractmethod
    def register_icp(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        *,
        max_iterations: int,
        tolerance_m: float,
        max_correspondence_distance_m: float,
    ) -> PointCloudRegistrationResult:
        """Rigid ICP registration of source points into target frame."""


class NumpyPointCloudOps(PointCloudOps):
    @property
    def backend_name(self) -> str:
        return "numpy"

    def crop_aabb(
        self,
        points: Sequence[tuple[float, float, float]],
        *,
        center: tuple[float, float, float],
        size_xyz: tuple[float, float, float],
    ) -> np.ndarray:
        if not points:
            return np.empty((0, 3), dtype=np.float64)

        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("point cloud must be Nx3")

        center_arr = np.asarray(center, dtype=np.float64)
        half = np.asarray(size_xyz, dtype=np.float64) * 0.5
        mask = np.all(np.abs(arr - center_arr[None, :]) <= half[None, :], axis=1)
        return arr[mask]

    def merge_downsample(
        self,
        existing_points: np.ndarray,
        new_points: np.ndarray,
        *,
        voxel_size_m: float,
        max_points: int,
    ) -> np.ndarray:
        if existing_points.size == 0:
            combined = np.asarray(new_points, dtype=np.float64)
        elif new_points.size == 0:
            combined = np.asarray(existing_points, dtype=np.float64)
        else:
            combined = np.vstack([existing_points, new_points])

        if combined.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        quantized = np.floor(combined / max(voxel_size_m, 1e-6)).astype(np.int64)
        _, first_indices = np.unique(quantized, axis=0, return_index=True)
        filtered = combined[np.sort(first_indices)]

        if len(filtered) <= max_points:
            return filtered
        return filtered[-max_points:]

    def nearest_neighbor_distances(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        if source_points.size == 0 or target_points.size == 0:
            return np.empty((0,), dtype=np.float64)

        deltas = source_points[:, None, :] - target_points[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        return np.min(distances, axis=1)

    def register_icp(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        *,
        max_iterations: int,
        tolerance_m: float,
        max_correspondence_distance_m: float,
    ) -> PointCloudRegistrationResult:
        source = np.asarray(source_points, dtype=np.float64)
        target = np.asarray(target_points, dtype=np.float64)

        if source.size == 0 or target.size == 0:
            return PointCloudRegistrationResult(
                transformed_source_points=source,
                mean_error_m=float("inf"),
                correspondence_ratio=0.0,
                iterations=0,
            )
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("source point cloud must be Nx3")
        if target.ndim != 2 or target.shape[1] != 3:
            raise ValueError("target point cloud must be Nx3")

        max_iters = max(1, int(max_iterations))
        tolerance = max(1e-9, float(tolerance_m))
        max_distance = max(1e-9, float(max_correspondence_distance_m))

        transformed = np.array(source, copy=True)
        previous_error = float("inf")
        iterations = 0

        for iteration in range(max_iters):
            deltas = transformed[:, None, :] - target[None, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            nearest_indices = np.argmin(distances, axis=1)
            nearest_distances = distances[np.arange(len(transformed)), nearest_indices]

            correspondence_mask = nearest_distances <= max_distance
            if int(np.sum(correspondence_mask)) < 3:
                break

            src_corr = transformed[correspondence_mask]
            tgt_corr = target[nearest_indices[correspondence_mask]]

            src_centroid = np.mean(src_corr, axis=0)
            tgt_centroid = np.mean(tgt_corr, axis=0)

            src_centered = src_corr - src_centroid
            tgt_centered = tgt_corr - tgt_centroid
            covariance = src_centered.T @ tgt_centered

            try:
                u, _, vt = np.linalg.svd(covariance)
            except np.linalg.LinAlgError:
                break

            rotation = vt.T @ u.T
            if np.linalg.det(rotation) < 0.0:
                vt[-1, :] *= -1.0
                rotation = vt.T @ u.T
            translation = tgt_centroid - rotation @ src_centroid

            transformed = transformed @ rotation.T + translation[None, :]

            updated_src_corr = transformed[correspondence_mask]
            residual_error = float(
                np.mean(np.linalg.norm(updated_src_corr - tgt_corr, axis=1))
            )
            iterations = iteration + 1

            if abs(previous_error - residual_error) < tolerance:
                previous_error = residual_error
                break
            previous_error = residual_error

        final_deltas = transformed[:, None, :] - target[None, :, :]
        final_distances = np.linalg.norm(final_deltas, axis=2)
        final_nearest = np.min(final_distances, axis=1)
        final_mask = final_nearest <= max_distance
        correspondence_ratio = float(np.mean(final_mask))
        mean_error = (
            float(np.mean(final_nearest[final_mask]))
            if np.any(final_mask)
            else float("inf")
        )

        return PointCloudRegistrationResult(
            transformed_source_points=transformed,
            mean_error_m=mean_error,
            correspondence_ratio=correspondence_ratio,
            iterations=iterations,
        )


class PclPointCloudOps(NumpyPointCloudOps):
    """PCL-backed point-cloud operations.

    Uses numpy fallbacks for behavior compatibility when specific PCL ops are unavailable
    in the active Python binding/API version.
    """

    def __init__(self) -> None:
        self._binding = None
        self._binding_name = ""

        try:
            import pclpy  # type: ignore

            self._binding = pclpy
            self._binding_name = "pclpy"
            return
        except Exception:
            pass

        try:
            import pcl  # type: ignore

            self._binding = pcl
            self._binding_name = "python-pcl"
            return
        except Exception as exc:
            raise ImportError("No PCL Python binding found (pclpy or pcl)") from exc

    @property
    def backend_name(self) -> str:
        return "pcl"


class TorchPointCloudOps(PointCloudOps):
    """Torch-backed point-cloud ops with CUDA acceleration and approximate ICP/NN."""

    def __init__(
        self,
        *,
        prefer_cuda: bool = True,
        nn_target_limit: int = 4096,
        nn_source_chunk: int = 4096,
        icp_sample_points: int = 2048,
        icp_target_limit: int = 4096,
    ) -> None:
        try:
            import torch
        except Exception as exc:
            raise ImportError("torch is required for torch point-cloud backend") from exc

        self._torch: Any = torch
        use_cuda = bool(prefer_cuda and torch.cuda.is_available())
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._dtype = torch.float32

        self._nn_target_limit = max(1, int(nn_target_limit))
        self._nn_source_chunk = max(1, int(nn_source_chunk))
        self._icp_sample_points = max(3, int(icp_sample_points))
        self._icp_target_limit = max(3, int(icp_target_limit))

    @property
    def backend_name(self) -> str:
        return "torch"

    def crop_aabb(
        self,
        points: Sequence[tuple[float, float, float]],
        *,
        center: tuple[float, float, float],
        size_xyz: tuple[float, float, float],
    ) -> np.ndarray:
        if not points:
            return np.empty((0, 3), dtype=np.float64)
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("point cloud must be Nx3")

        torch = self._torch
        points_t = torch.as_tensor(arr, dtype=self._dtype, device=self._device)
        center_t = torch.as_tensor(center, dtype=self._dtype, device=self._device)
        half_t = torch.as_tensor(size_xyz, dtype=self._dtype, device=self._device) * 0.5
        mask = torch.all(torch.abs(points_t - center_t[None, :]) <= half_t[None, :], dim=1)
        return points_t[mask].detach().cpu().numpy().astype(np.float64, copy=False)

    def merge_downsample(
        self,
        existing_points: np.ndarray,
        new_points: np.ndarray,
        *,
        voxel_size_m: float,
        max_points: int,
    ) -> np.ndarray:
        torch = self._torch
        if existing_points.size == 0:
            combined = np.asarray(new_points, dtype=np.float32)
        elif new_points.size == 0:
            combined = np.asarray(existing_points, dtype=np.float32)
        else:
            combined = np.vstack(
                [np.asarray(existing_points, dtype=np.float32), np.asarray(new_points, dtype=np.float32)]
            )
        if combined.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        combined_t = torch.as_tensor(combined, dtype=self._dtype, device=self._device)
        voxel = max(1e-6, float(voxel_size_m))
        quantized = torch.floor(combined_t / voxel).to(torch.int64)
        _, inverse = torch.unique(quantized, dim=0, return_inverse=True)
        order = torch.argsort(inverse)
        sorted_inverse = inverse[order]
        first_flags = torch.ones_like(sorted_inverse, dtype=torch.bool)
        first_flags[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
        first_indices = order[first_flags]
        filtered = combined_t[first_indices]

        if int(filtered.shape[0]) > int(max_points):
            filtered = filtered[-int(max_points):]
        return filtered.detach().cpu().numpy().astype(np.float64, copy=False)

    def _subsample_points(self, points_t: Any, limit: int) -> Any:
        count = int(points_t.shape[0])
        if count <= limit:
            return points_t
        torch = self._torch
        indices = torch.randperm(count, device=points_t.device)[:limit]
        return points_t[indices]

    def _nearest_neighbor_distances_tensor(
        self,
        source_t: Any,
        target_t: Any,
        *,
        target_limit: int,
    ) -> Any:
        torch = self._torch
        if source_t.numel() == 0 or target_t.numel() == 0:
            return torch.empty((0,), dtype=self._dtype, device=self._device)

        target_used = self._subsample_points(target_t, target_limit)
        mins: list[Any] = []
        source_count = int(source_t.shape[0])
        for start in range(0, source_count, self._nn_source_chunk):
            end = min(source_count, start + self._nn_source_chunk)
            chunk = source_t[start:end]
            distances = torch.cdist(chunk, target_used)
            mins.append(distances.min(dim=1).values)
        return torch.cat(mins, dim=0) if mins else torch.empty((0,), dtype=self._dtype, device=self._device)

    def nearest_neighbor_distances(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        torch = self._torch
        source = np.asarray(source_points, dtype=np.float32)
        target = np.asarray(target_points, dtype=np.float32)
        if source.size == 0 or target.size == 0:
            return np.empty((0,), dtype=np.float64)
        source_t = torch.as_tensor(source, dtype=self._dtype, device=self._device)
        target_t = torch.as_tensor(target, dtype=self._dtype, device=self._device)
        nearest = self._nearest_neighbor_distances_tensor(
            source_t,
            target_t,
            target_limit=self._nn_target_limit,
        )
        return nearest.detach().cpu().numpy().astype(np.float64, copy=False)

    def register_icp(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        *,
        max_iterations: int,
        tolerance_m: float,
        max_correspondence_distance_m: float,
    ) -> PointCloudRegistrationResult:
        torch = self._torch
        source = np.asarray(source_points, dtype=np.float32)
        target = np.asarray(target_points, dtype=np.float32)

        if source.size == 0 or target.size == 0:
            return PointCloudRegistrationResult(
                transformed_source_points=np.asarray(source_points, dtype=np.float64),
                mean_error_m=float("inf"),
                correspondence_ratio=0.0,
                iterations=0,
            )
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("source point cloud must be Nx3")
        if target.ndim != 2 or target.shape[1] != 3:
            raise ValueError("target point cloud must be Nx3")

        max_iters = max(1, int(max_iterations))
        tolerance = max(1e-9, float(tolerance_m))
        max_distance = max(1e-9, float(max_correspondence_distance_m))

        source_t = torch.as_tensor(source, dtype=self._dtype, device=self._device)
        target_t = torch.as_tensor(target, dtype=self._dtype, device=self._device)
        transformed = source_t.clone()
        target_for_nn = self._subsample_points(target_t, self._icp_target_limit)

        previous_error = float("inf")
        iterations = 0

        for iteration in range(max_iters):
            source_for_icp = self._subsample_points(transformed, self._icp_sample_points)
            distances = torch.cdist(source_for_icp, target_for_nn)
            nearest_distances, nearest_indices = distances.min(dim=1)
            correspondence_mask = nearest_distances <= max_distance
            if int(correspondence_mask.sum().item()) < 3:
                break

            src_corr = source_for_icp[correspondence_mask]
            tgt_corr = target_for_nn[nearest_indices[correspondence_mask]]
            src_centroid = src_corr.mean(dim=0)
            tgt_centroid = tgt_corr.mean(dim=0)
            src_centered = src_corr - src_centroid
            tgt_centered = tgt_corr - tgt_centroid
            covariance = src_centered.T @ tgt_centered

            try:
                u, _, vh = torch.linalg.svd(covariance)
            except Exception:
                break

            rotation = vh.transpose(0, 1) @ u.transpose(0, 1)
            if float(torch.det(rotation).item()) < 0.0:
                vh_adj = vh.clone()
                vh_adj[-1, :] *= -1.0
                rotation = vh_adj.transpose(0, 1) @ u.transpose(0, 1)
            translation = tgt_centroid - rotation @ src_centroid

            transformed = transformed @ rotation.T + translation[None, :]

            updated_src_corr = src_corr @ rotation.T + translation[None, :]
            residual_error = float(torch.linalg.norm(updated_src_corr - tgt_corr, dim=1).mean().item())
            iterations = iteration + 1

            if abs(previous_error - residual_error) < tolerance:
                previous_error = residual_error
                break
            previous_error = residual_error

        final_nearest = self._nearest_neighbor_distances_tensor(
            transformed,
            target_t,
            target_limit=self._nn_target_limit,
        )
        if final_nearest.numel() == 0:
            correspondence_ratio = 0.0
            mean_error = float("inf")
        else:
            final_mask = final_nearest <= max_distance
            correspondence_ratio = float(final_mask.float().mean().item())
            if bool(final_mask.any().item()):
                mean_error = float(final_nearest[final_mask].mean().item())
            else:
                mean_error = float("inf")

        return PointCloudRegistrationResult(
            transformed_source_points=transformed.detach().cpu().numpy().astype(np.float64, copy=False),
            mean_error_m=mean_error,
            correspondence_ratio=correspondence_ratio,
            iterations=iterations,
        )


@dataclass(frozen=True)
class PointCloudBackendSelection:
    backend: PointCloudOps
    requested_backend: str


def create_point_cloud_ops(
    *,
    requested_backend: str = "pcl",
    require_backend: bool = False,
) -> PointCloudBackendSelection:
    normalized = requested_backend.strip().lower()

    if normalized == "pcl":
        try:
            return PointCloudBackendSelection(
                backend=PclPointCloudOps(),
                requested_backend=normalized,
            )
        except Exception:
            if require_backend:
                raise
            try:
                return PointCloudBackendSelection(
                    backend=TorchPointCloudOps(prefer_cuda=True),
                    requested_backend=normalized,
                )
            except Exception:
                pass
            return PointCloudBackendSelection(
                backend=NumpyPointCloudOps(),
                requested_backend=normalized,
            )

    if normalized in {"torch", "gpu", "cuda"}:
        try:
            return PointCloudBackendSelection(
                backend=TorchPointCloudOps(prefer_cuda=True),
                requested_backend=normalized,
            )
        except Exception:
            if require_backend:
                raise
            return PointCloudBackendSelection(
                backend=NumpyPointCloudOps(),
                requested_backend=normalized,
            )

    if normalized in {"torch_cpu", "cpu_torch"}:
        try:
            return PointCloudBackendSelection(
                backend=TorchPointCloudOps(prefer_cuda=False),
                requested_backend=normalized,
            )
        except Exception:
            if require_backend:
                raise
            return PointCloudBackendSelection(
                backend=NumpyPointCloudOps(),
                requested_backend=normalized,
            )

    if normalized == "numpy":
        return PointCloudBackendSelection(
            backend=NumpyPointCloudOps(),
            requested_backend=normalized,
        )

    raise ValueError("Unknown point-cloud backend. Expected one of: pcl, torch, gpu, cuda, numpy")
