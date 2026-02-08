"""world2data -- Video-to-USD pipeline with particle tracking and 4D scene graphs.

Core modules:
  - model:            Data model (AABB2D, Detection2D, CameraPose, etc.)
  - particle_filter:  Multi-object particle filter with separated motion/cost/resampling
  - vision:           Projection/back-projection helpers
  - calibration:      ChArUco camera calibration
  - openusd:          OpenUSD export for particle-filter track estimates
  - usd_layers:       OpenUSD Layering Protocol (multi-layer composition)

Pipeline modules (world2data.pipeline):
  - controller:       World2DataPipeline (the "Ralph Loop")
  - model_interfaces: YOLOv8, SAM3, Gemini, Reasoning Engine
  - scene_fusion:     4D Scene Fusion, GroundTruthEvaluator, VideoAnnotator
"""

from .model import (
    AABB2D,
    CameraIntrinsics,
    CameraPose,
    CostFunction,
    Detection2D,
    FilterResult,
    FrameContext,
    MotionModel,
    Particle,
    ParticleState,
    Resampler,
    TrackEstimate,
)
from .openusd import track_estimates_to_stage, track_estimates_to_usda
from .point_cloud import (
    NumpyPointCloudOps,
    PclPointCloudOps,
    PointCloudOps,
    create_point_cloud_ops,
)
from .particle_filter import (
    BackProjectionIoUCost,
    ConstantVelocityMotionModel,
    IoUPointCloudCost,
    MultiObjectParticleFilter,
    MultinomialResampler,
    ParticleFilterConfig,
    PointCloudAlignmentCost,
    SingleObjectParticleFilter,
    StratifiedResampler,
    SystematicResampler,
    build_resampler,
)
from .vision import (
    back_project_pixel,
    box3d_corners,
    camera_pose_from_world_position,
    camera_to_world,
    iou_2d,
    project_box_to_image_aabb,
    project_point,
    world_to_camera,
)

__all__ = [
    # model
    "AABB2D",
    "BackProjectionIoUCost",
    "CameraIntrinsics",
    "CameraPose",
    "ConstantVelocityMotionModel",
    "CostFunction",
    "Detection2D",
    "FilterResult",
    "FrameContext",
    "IoUPointCloudCost",
    "MotionModel",
    "MultiObjectParticleFilter",
    "MultinomialResampler",
    "NumpyPointCloudOps",
    "Particle",
    "ParticleFilterConfig",
    "ParticleState",
    "PointCloudAlignmentCost",
    "PointCloudOps",
    "PclPointCloudOps",
    "Resampler",
    "SingleObjectParticleFilter",
    "StratifiedResampler",
    "SystematicResampler",
    "TrackEstimate",
    # vision
    "back_project_pixel",
    "box3d_corners",
    "build_resampler",
    "camera_pose_from_world_position",
    "camera_to_world",
    "create_point_cloud_ops",
    "iou_2d",
    "project_box_to_image_aabb",
    "project_point",
    "track_estimates_to_stage",
    "track_estimates_to_usda",
    "world_to_camera",
]
