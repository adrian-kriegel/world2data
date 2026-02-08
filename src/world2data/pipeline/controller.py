"""World2Data Pipeline Controller -- The "Ralph Loop"

Converts 2D video into a 3D Dynamic Scene Graph (OpenUSD) using:
  Step 1: Smart keyframe extraction (L1 diff, upgradeable to LPIPS)
  Step 2: Metric 3D reconstruction via MASt3R  (per-frame temporal data)
  Step 3: Multi-model 3D object detection (YOLOv8 + SAM3 + Gemini Video)
  Step 4: 4D Scene Fusion + Reasoning (cross-model validation + confidence)
  Step 5: OpenUSD export with physics joint metadata + confidence + human flags
  Step 6: Save interactive Rerun .rrd recording (with annotated video overlay)
  Step 7: PLY colored point cloud export
"""
import os
import sys
import json
import tempfile
import shutil
import re
import subprocess
import cv2
import torch
import numpy as np
import rerun as rr
from collections import defaultdict
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, Vt

# ---------------------------------------------------------------------------
# Load .env file (GOOGLE_API_KEY, etc.)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

# Resolve project root: src/world2data/pipeline/ -> three levels up
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# MASt3R: add the local clone to sys.path so imports resolve
# ---------------------------------------------------------------------------
MAST3R_ROOT = str(_PROJECT_ROOT / "mast3r")
if MAST3R_ROOT not in sys.path:
    sys.path.insert(0, MAST3R_ROOT)


def _ensure_curope_cuda_extension():
    """Build the cuRoPE2D CUDA extension once if it is not importable."""
    if os.environ.get("WORLD2DATA_SKIP_CUROPE_BUILD", "0") == "1":
        return
    if not torch.cuda.is_available():
        return

    ext_dir = Path(MAST3R_ROOT) / "dust3r" / "croco" / "models" / "curope"
    if not ext_dir.is_dir():
        return
    # If already compiled in-place for this environment, skip building.
    if (
        list(ext_dir.glob("curope*.pyd"))
        or list(ext_dir.glob("curope*.so"))
        or list(ext_dir.glob("curope*.dylib"))
    ):
        return

    if shutil.which("nvcc") is None:
        print("WARNING: CUDA compiler (nvcc) not found; cuRoPE2D CUDA extension not built.")
        return

    print("INFO: Building cuRoPE2D CUDA extension (one-time setup)...")
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    try:
        res = subprocess.run(
            cmd,
            cwd=str(ext_dir),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"WARNING: Failed to launch cuRoPE2D build: {exc}")
        return

    if res.returncode != 0:
        tail = "\n".join((res.stdout or "").splitlines()[-20:])
        print("WARNING: cuRoPE2D CUDA build failed; using slow PyTorch fallback.")
        if tail:
            print(tail)
        return

    print("INFO: cuRoPE2D CUDA extension is ready.")


_ensure_curope_cuda_extension()

_HAS_MAST3R = False
try:
    from mast3r.model import AsymmetricMASt3R, load_model  # noqa: E402,F401
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment  # noqa: E402
    from mast3r.image_pairs import make_pairs  # noqa: E402
    from dust3r.utils.image import load_images  # noqa: E402
    from dust3r.utils.device import to_numpy  # noqa: E402
    _HAS_MAST3R = True
except ImportError as e:
    print(f"WARNING: MASt3R not available ({e}). Step 2 will run in mock mode.")

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
_HAS_GEMINI = False
try:
    from google import genai  # noqa: E402
    _HAS_GEMINI = True
except ImportError:
    print("WARNING: google-genai not installed. Steps 3+4 will be skipped.")

# ---------------------------------------------------------------------------
# New Model Interfaces (YOLOv8, SAM3, Gemini Video, Scene Fusion)
# ---------------------------------------------------------------------------
_HAS_MODEL_INTERFACES = False
_HAS_YOLO = False
try:
    from .model_interfaces import (
        YOLODetector, SAM3Segmenter, GeminiVideoAnalyzer, ReasoningEngine,
        DetectionResult, SegmentationResult, SceneDescription, ReasoningResult,
        _HAS_YOLO as _YOLO_OK, _HAS_SAM3 as _SAM3_OK,
    )
    _HAS_MODEL_INTERFACES = True
    _HAS_YOLO = _YOLO_OK
    _HAS_SAM3 = _SAM3_OK
except ImportError as e:
    print(f"WARNING: model_interfaces not available ({e}). Using legacy pipeline.")
    _HAS_SAM3 = False

_HAS_FUSION = False
try:
    from .scene_fusion import (
        SceneFusion4D, SceneObject4D, GroundTruthEvaluator, VideoAnnotator,
    )
    _HAS_FUSION = True
except ImportError as e:
    print(f"WARNING: scene_fusion not available ({e}).")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAST3R_CHECKPOINT = os.path.join(
    MAST3R_ROOT, "checkpoints",
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_output_usd_path(path: str) -> str:
    """Ensure output path has a valid USD extension."""
    valid_ext = {".usda", ".usd", ".usdc"}
    p = Path(path)
    if p.suffix.lower() in valid_ext:
        return str(p)
    return str(p) + ".usda"


# =========================================================================
# Data containers
# =========================================================================
class FrameData:
    """Per-keyframe 3D reconstruction data."""
    __slots__ = ("index", "pts3d", "colors", "confidence",
                 "pose", "focal", "principal_point", "image_rgb",
                 "depth_map")

    def __init__(self, index, pts3d, colors, confidence,
                 pose, focal, principal_point, image_rgb, depth_map=None):
        self.index = index
        self.pts3d = pts3d                # (N, 3) float32
        self.colors = colors              # (N, 3) float32 0-1
        self.confidence = confidence      # (N,)   float32
        self.pose = pose                  # (4, 4) float64 cam2world
        self.focal = focal                # float
        self.principal_point = principal_point  # (2,) float
        self.image_rgb = image_rgb        # (H, W, 3) uint8 RGB
        self.depth_map = depth_map        # (H, W) float32 or None


class Object3D:
    """A detected 3D object with bounding box, label, and physics metadata."""
    def __init__(self, entity, obj_type, bbox_3d_min, bbox_3d_max,
                 component_type="FixedJoint", initial_state="unknown",
                 final_state="unknown", state_changes=None,
                 first_seen_frame=0, last_seen_frame=0,
                 observation_count=1, tracking_confidence=1.0):
        self.entity = entity              # "Door_01"
        self.obj_type = obj_type          # "door"
        self.bbox_3d_min = np.array(bbox_3d_min, dtype=np.float32)  # (3,)
        self.bbox_3d_max = np.array(bbox_3d_max, dtype=np.float32)  # (3,)
        self.component_type = component_type  # "RevoluteJoint"
        self.initial_state = initial_state
        self.final_state = final_state
        self.state_changes = state_changes or []
        self.first_seen_frame = first_seen_frame
        self.last_seen_frame = last_seen_frame
        self.observation_count = int(observation_count)
        self.tracking_confidence = float(tracking_confidence)

    @property
    def center(self):
        return (self.bbox_3d_min + self.bbox_3d_max) / 2

    @property
    def size(self):
        return self.bbox_3d_max - self.bbox_3d_min

    def to_dict(self):
        return {
            "entity": self.entity,
            "type": self.obj_type,
            "component_type": self.component_type,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "state_changes": self.state_changes,
            "bbox_3d_min": self.bbox_3d_min.tolist(),
            "bbox_3d_max": self.bbox_3d_max.tolist(),
            "center": self.center.tolist(),
            "size": self.size.tolist(),
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "observation_count": self.observation_count,
            "tracking_confidence": self.tracking_confidence,
        }


class ParticleFilter3D:
    """Small 3D particle filter for temporal object center smoothing."""

    def __init__(self, num_particles=256, process_noise=0.03,
                 measurement_noise=0.15, rng=None):
        self.num_particles = int(num_particles)
        self.process_noise = float(process_noise)
        self.measurement_noise = max(float(measurement_noise), 1e-5)
        self.rng = rng if rng is not None else np.random.RandomState(42)
        self.particles = np.zeros((self.num_particles, 3), dtype=np.float32)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float32)
        self.initialized = False

    def initialize(self, measurement):
        measurement = np.asarray(measurement, dtype=np.float32)
        self.particles = measurement + self.rng.normal(
            0.0, self.measurement_noise, size=(self.num_particles, 3)
        ).astype(np.float32)
        self.weights.fill(1.0 / self.num_particles)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return
        self.particles += self.rng.normal(
            0.0, self.process_noise, size=self.particles.shape
        ).astype(np.float32)

    def update(self, measurement, measurement_confidence=1.0):
        measurement = np.asarray(measurement, dtype=np.float32)
        if not self.initialized:
            self.initialize(measurement)
            return

        measurement_confidence = float(np.clip(measurement_confidence, 0.05, 1.0))
        sigma = self.measurement_noise / measurement_confidence

        # If particles are far from the measurement, inject a portion around it
        # to recover from poor initialization and keep tracking stable.
        estimate = self.estimate()
        if np.linalg.norm(estimate - measurement) > (3.0 * sigma):
            n_inject = max(1, int(0.35 * self.num_particles))
            inject_idx = self.rng.choice(self.num_particles, n_inject, replace=False)
            self.particles[inject_idx] = measurement + self.rng.normal(
                0.0, sigma, size=(n_inject, 3)
            ).astype(np.float32)

        d2 = np.sum((self.particles - measurement) ** 2, axis=1)
        likelihood = np.exp(-0.5 * d2 / (sigma ** 2)).astype(np.float32)
        self.weights *= likelihood
        weight_sum = float(np.sum(self.weights))
        if weight_sum <= 1e-12:
            self.weights.fill(1.0 / self.num_particles)
        else:
            self.weights /= weight_sum
        self._resample_if_degenerate()

    def estimate(self):
        if not self.initialized:
            return np.zeros(3, dtype=np.float32)
        return np.average(self.particles, axis=0, weights=self.weights).astype(np.float32)

    def spread(self):
        if not self.initialized:
            return 0.0
        center = self.estimate()
        d = np.linalg.norm(self.particles - center, axis=1)
        return float(np.average(d, weights=self.weights))

    def _resample_if_degenerate(self):
        neff = 1.0 / np.sum(np.square(self.weights))
        if neff >= self.num_particles * 0.5:
            return
        positions = (np.arange(self.num_particles) + self.rng.rand()) / self.num_particles
        cumulative = np.cumsum(self.weights)
        indexes = np.zeros(self.num_particles, dtype=np.int32)
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.num_particles)


# =========================================================================
# Pipeline
# =========================================================================
class World2DataPipeline:
    """Self-correcting pipeline: video -> 3D scene graph -> USD."""

    def __init__(self, video_path, output_path="output.usda",
                 keyframe_dir=None, cache_dir=None, rerun_enabled=True,
                 seed=42):
        self.video_path = video_path
        self.output_path = _normalize_output_usd_path(output_path)
        self.keyframe_dir = keyframe_dir or tempfile.mkdtemp(prefix="w2d_keyframes_")
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="w2d_cache_")
        self._owns_keyframe_dir = keyframe_dir is None
        self._owns_cache_dir = cache_dir is None
        self.rng = np.random.RandomState(seed)

        # Pipeline state
        self.keyframes = []
        self.keyframe_paths = []
        self.keyframe_timestamps = []
        self.video_fps = 30.0

        # Temporal 3D data
        self.frame_data = []          # list[FrameData]

        # Aggregated geometry
        self.poses = []
        self.focals = []
        self.point_cloud = None
        self.point_colors = None

        # Semantic scene graph
        self.scene_graph = {}         # raw Gemini JSON
        self.objects_3d = []          # list[Object3D] -- positioned in 3D

        # New multi-model outputs
        self.yolo_detections = []     # list[DetectionResult]
        self.sam3_segmentations = []  # list[SegmentationResult]
        self.scene_description = None # SceneDescription from Gemini Video
        self.reasoning_result = None  # ReasoningResult
        self.objects_4d = []          # list[SceneObject4D] from fusion
        self.evaluation = {}          # GroundTruthEvaluator output

        # Rerun
        self.rerun_enabled = rerun_enabled
        if rerun_enabled:
            rr.init("world2data_debug", spawn=False)

    @staticmethod
    def _safe_name(text):
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(text)).strip("_")
        return safe or "Object"

    def _derive_output_path(self, suffix):
        p = Path(self.output_path)
        if p.suffix.lower() == ".usda":
            if suffix.startswith("."):
                return str(p.with_suffix(suffix))
            return str(p.with_suffix("")) + suffix
        return f"{self.output_path}{suffix}"

    def _cleanup_temp_dirs(self):
        if self._owns_keyframe_dir and os.path.isdir(self.keyframe_dir):
            shutil.rmtree(self.keyframe_dir, ignore_errors=True)
        if self._owns_cache_dir and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    # =====================================================================
    # RALPH LOOP
    # =====================================================================
    def run_ralph_loop(self, threshold=15.0, strategy="swin-3",
                       max_keyframes=20, target_fps=30.0,
                       use_multimodel=True, demo_fps=0):
        """Run the full pipeline with self-correction.

        Args:
            threshold: Keyframe extraction L1 diff threshold.
            strategy: MASt3R pair strategy.
            max_keyframes: Maximum keyframes to reconstruct. `None`/`0` means no cap.
            target_fps: Frame sampling rate for extraction (0/None keeps source FPS).
            use_multimodel: If True, use YOLOv8 + SAM3 + Gemini Video (v2).
                           If False, use legacy Gemini-only detection (v1).
            demo_fps: If > 0, extract keyframes uniformly at this FPS and use
                      sliding-window MASt3R reconstruction. Use 5 for demos.
        """
        has_v2_stack = _HAS_MODEL_INTERFACES and _HAS_FUSION
        mode_str = " (MULTI-MODEL v2)" if (use_multimodel and has_v2_stack) else " (LEGACY v1)"
        if demo_fps and demo_fps > 0:
            mode_str += f" [DEMO {demo_fps} FPS, sliding-window]"
        print(f">>> STARTING RALPH LOOP{mode_str}...")
        try:
            # STEP 1: KEYFRAME EXTRACTION
            if demo_fps and demo_fps > 0:
                # High-FPS demo mode: uniform extraction, no cap, no diff threshold
                if not self.step_1_smart_extraction(
                    threshold=0, max_keyframes=0, target_fps=demo_fps,
                ):
                    print("!!! Step 1 Failed at demo FPS. Check your video input.")
                    return False
            else:
                if not self.step_1_smart_extraction(
                    threshold=threshold, max_keyframes=max_keyframes, target_fps=target_fps
                ):
                    print("!!! Step 1 Failed. Retrying with lower threshold...")
                    if not self.step_1_smart_extraction(
                        threshold=max(1.0, threshold * 0.33),
                        max_keyframes=max_keyframes,
                        target_fps=target_fps,
                    ):
                        print("!!! Step 1 still failing. Check your video input.")
                        return False

            # STEP 2: METRIC GEOMETRY (MASt3R)
            use_sliding_window = (demo_fps and demo_fps > 0
                                  and len(self.keyframes) > 30)
            if use_sliding_window:
                # Sliding-window mode for high-FPS demos
                if not self.step_2_sliding_window_reconstruction(
                    strategy=strategy, window_sec=5.0, overlap_sec=1.0,
                ):
                    print("!!! Step 2 sliding-window failed. Falling back to standard...")
                    if not self.step_2_geometric_reconstruction(strategy=strategy):
                        print("!!! Step 2 still failing. Video may lack parallax.")
                        return False
            elif not self.step_2_geometric_reconstruction(strategy=strategy):
                print("!!! Step 2 Failed. Retrying with exhaustive matching...")
                if not self.step_2_geometric_reconstruction(strategy="complete"):
                    print("!!! Step 2 still failing. Video may lack parallax.")
                    return False

            can_use_multimodel = use_multimodel and has_v2_stack
            if can_use_multimodel:
                # ========== NEW v2 MULTI-MODEL PIPELINE ==========
                # STEP 3a: YOLOv8 Detection (fast, all keyframes)
                self.step_3a_yolo_detection()

                # STEP 3b: SAM3 Video Segmentation (pixel-perfect + tracking)
                self.step_3b_sam3_segmentation()

                # STEP 3c: Gemini Video Analysis (full video upload)
                self.step_3c_gemini_video_analysis()

                # STEP 4: 4D Scene Fusion + Reasoning
                self.step_4_scene_fusion_and_reasoning()
            else:
                # ========== LEGACY v1 (fallback) ==========
                self.step_3_semantic_detection()
                self.step_4_causal_reasoning()

            # STEP 5: OPENUSD EXPORT (with physics + confidence + human flags)
            self.step_5_export_usd()

            # STEP 6: TEMPORAL RERUN RECORDING
            if self.rerun_enabled:
                rrd_path = self._derive_output_path(".rrd")
                self.save_temporal_rrd(rrd_path)
            else:
                print("SKIP: Rerun recording disabled (--no-rerun).")

            # STEP 7: PLY EXPORT (for easy 3D viewer access)
            ply_path = self._derive_output_path(".ply")
            self.export_ply(ply_path)

            print(">>> PIPELINE COMPLETE. ARTIFACT GENERATED.")
            return True
        finally:
            self._cleanup_temp_dirs()

    # =====================================================================
    # STEP 1: KEYFRAME EXTRACTION
    # =====================================================================
    def step_1_smart_extraction(self, threshold=15.0, max_keyframes=20,
                                target_fps=30.0):
        """Extract keyframes via L1 pixel diff over the full video timeline."""
        max_kf_label = "unlimited" if not max_keyframes else str(max_keyframes)
        fps_label = "source" if not target_fps else f"{float(target_fps):.1f}"
        print(
            f"--- Step 1: Extracting Keyframes "
            f"(Threshold: {threshold}, Max: {max_kf_label}, Target FPS: {fps_label}) ---"
        )

        if not os.path.isfile(self.video_path):
            print(f"FAIL: Video not found at '{self.video_path}'.")
            return False

        if max_keyframes is not None and max_keyframes > 0 and max_keyframes < 2:
            print(f"FAIL: max_keyframes must be >= 2, got {max_keyframes}.")
            return False

        # Normalize sentinel values: 0/None means no cap.
        max_keyframes = None if (max_keyframes is None or max_keyframes <= 0) else int(max_keyframes)

        # Pass 1: scan full video to get candidate keyframe indices.
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"FAIL: Could not open video '{self.video_path}'.")
            return False

        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if target_fps is None or target_fps <= 0:
            sample_step = 1
            effective_fps = self.video_fps
        else:
            sample_step = max(1, int(round(self.video_fps / float(target_fps))))
            effective_fps = self.video_fps / sample_step

        print(
            f"  Source FPS: {self.video_fps:.3f} | "
            f"Sampling FPS: {effective_fps:.3f} (step={sample_step})"
        )

        last_frame = None
        candidate_idxs = []
        last_frame_idx = -1

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if last_frame is None:
                candidate_idxs.append(frame_idx)
                last_frame = frame
                last_frame_idx = frame_idx
                frame_idx += 1
                continue

            if sample_step > 1 and (frame_idx % sample_step != 0):
                last_frame_idx = frame_idx
                frame_idx += 1
                continue

            if threshold <= 0:
                candidate_idxs.append(frame_idx)
                last_frame = frame
                last_frame_idx = frame_idx
                frame_idx += 1
                continue

            diff = np.mean(np.abs(frame.astype(float) - last_frame.astype(float)))
            if diff > threshold:
                candidate_idxs.append(frame_idx)
                last_frame = frame
            last_frame_idx = frame_idx
            frame_idx += 1
        cap.release()

        if last_frame_idx < 1:
            print("FAIL: Video is too short or unreadable.")
            return False

        # Always include the last frame so temporal coverage reaches video end.
        if not candidate_idxs or candidate_idxs[-1] != last_frame_idx:
            candidate_idxs.append(last_frame_idx)

        if len(candidate_idxs) < 2:
            print(f"FAIL: Only {len(candidate_idxs)} keyframe(s) extracted.")
            return False

        # If capped, keep first/last and spread the rest uniformly in-between.
        if max_keyframes is None or len(candidate_idxs) <= max_keyframes:
            selected_idxs = candidate_idxs
        else:
            first_idx = candidate_idxs[0]
            last_idx = candidate_idxs[-1]
            middle = candidate_idxs[1:-1]
            n_middle_keep = max_keyframes - 2
            if n_middle_keep <= 0:
                selected_idxs = [first_idx, last_idx]
            elif n_middle_keep >= len(middle):
                selected_idxs = [first_idx] + middle + [last_idx]
            else:
                positions = np.linspace(0, len(middle) - 1, num=n_middle_keep, dtype=int)
                selected_idxs = [first_idx] + [middle[i] for i in positions.tolist()] + [last_idx]

        # Guard against duplicated indices from coarse linspace rounding.
        selected_idxs = list(dict.fromkeys(selected_idxs))

        # Pass 2: decode only the selected frames.
        self.keyframes = []
        self.keyframe_paths = []
        self.keyframe_timestamps = []
        os.makedirs(self.keyframe_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"FAIL: Could not re-open video '{self.video_path}'.")
            return False

        pick_ptr = 0
        frame_idx = 0
        while cap.isOpened() and pick_ptr < len(selected_idxs):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx == selected_idxs[pick_ptr]:
                self.keyframes.append(frame)
                self.keyframe_timestamps.append(frame_idx)
                pick_ptr += 1
            frame_idx += 1
        cap.release()

        for i, kf in enumerate(self.keyframes):
            path = os.path.join(self.keyframe_dir, f"keyframe_{i:04d}.jpg")
            cv2.imwrite(path, kf)
            self.keyframe_paths.append(path)

        n = len(self.keyframes)
        if n < 2:
            print(f"FAIL: Only {n} keyframe(s) extracted.")
            return False

        ts_sec = [t / self.video_fps for t in self.keyframe_timestamps]
        total_duration = last_frame_idx / self.video_fps
        print(
            f"PASS: Extracted {n}/{len(candidate_idxs)} keyframes "
            f"[{ts_sec[0]:.1f}s .. {ts_sec[-1]:.1f}s] "
            f"across video duration {total_duration:.1f}s"
        )
        return True

    # =====================================================================
    # STEP 2: MASt3R RECONSTRUCTION (TEMPORAL)
    # =====================================================================
    def step_2_geometric_reconstruction(self, strategy="swin-3",
                                        image_size=512, min_points=100):
        """MASt3R reconstruction storing per-frame temporal 3D + depth maps."""
        print(f"--- Step 2: MASt3R Reconstruction (Strategy: {strategy}) ---")

        if not self.keyframe_paths:
            print("FAIL: No keyframes. Run Step 1 first.")
            return False
        if not _HAS_MAST3R:
            print("MOCK: MASt3R not available.")
            return self._mock_geometry()
        if not os.path.isfile(MAST3R_CHECKPOINT):
            print(f"FAIL: Checkpoint not found at {MAST3R_CHECKPOINT}")
            return False

        print(f"  Loading MASt3R model...")
        model = load_model(MAST3R_CHECKPOINT, DEVICE)
        imgs = load_images(self.keyframe_paths, size=image_size, verbose=True)
        pairs = make_pairs(imgs, scene_graph=strategy, symmetrize=True)
        print(f"  Created {len(pairs)} image pairs")
        if not pairs:
            print("FAIL: No image pairs.")
            return False

        os.makedirs(self.cache_dir, exist_ok=True)
        print("  Running sparse global alignment...")
        scene = sparse_global_alignment(
            self.keyframe_paths, pairs, self.cache_dir, model, device=DEVICE,
        )

        focals_t = to_numpy(scene.get_focals().cpu())
        cams2world_t = to_numpy(scene.get_im_poses().cpu())
        pp_t = to_numpy(scene.get_principal_points().cpu())
        pts3d_list, depthmaps_list, confs_list = to_numpy(
            scene.get_dense_pts3d(clean_depth=True)
        )

        min_conf = 1.5
        self.frame_data = []
        self.poses = []
        self.focals = []
        all_pts, all_colors = [], []

        for i in range(len(self.keyframe_paths)):
            pose = cams2world_t[i]
            focal = float(focals_t[i])
            pp = pp_t[i]

            pts = pts3d_list[i]
            conf = confs_list[i]
            depth = depthmaps_list[i] if depthmaps_list is not None else None

            pts_flat = pts.reshape(-1, 3)
            conf_flat = conf.reshape(-1)
            n = min(len(pts_flat), len(conf_flat))
            pts_flat, conf_flat = pts_flat[:n], conf_flat[:n]

            mask = conf_flat > min_conf
            valid_pts = pts_flat[mask]

            if i < len(scene.imgs):
                img_f = scene.imgs[i].reshape(-1, 3)[:n]
                valid_colors = img_f[mask]
            else:
                valid_colors = np.ones_like(valid_pts) * 0.5

            image_rgb = cv2.cvtColor(self.keyframes[i], cv2.COLOR_BGR2RGB)

            # Depth map for back-projection (keep as 2D)
            depth_2d = None
            if depth is not None:
                if depth.ndim == 1:
                    # Try to reshape to image dimensions
                    h, w = image_rgb.shape[:2]
                    if len(depth) == h * w:
                        depth_2d = depth.reshape(h, w).astype(np.float32)
                else:
                    depth_2d = depth.astype(np.float32)

            fd = FrameData(
                index=i,
                pts3d=valid_pts.astype(np.float32),
                colors=valid_colors.astype(np.float32),
                confidence=conf_flat[mask].astype(np.float32),
                pose=pose.astype(np.float64),
                focal=focal,
                principal_point=pp.astype(np.float64),
                image_rgb=image_rgb,
                depth_map=depth_2d,
            )
            self.frame_data.append(fd)
            self.poses.append(pose.tolist())
            self.focals.append(focal)
            all_pts.append(valid_pts)
            all_colors.append(valid_colors)

        if all_pts:
            self.point_cloud = np.concatenate(all_pts, axis=0)
            self.point_colors = np.concatenate(all_colors, axis=0)
        else:
            self.point_cloud = np.empty((0, 3))
            self.point_colors = np.empty((0, 3))

        n_pts = self.point_cloud.shape[0]
        per_frame = [fd.pts3d.shape[0] for fd in self.frame_data]
        if n_pts < min_points:
            print(f"FAIL: Only {n_pts} points (need {min_points}).")
            return False

        print(f"PASS: {n_pts} points across {len(self.frame_data)} frames "
              f"(per-frame: {min(per_frame)}-{max(per_frame)}, avg={int(np.mean(per_frame))})")
        return True

    def _mock_geometry(self):
        """Synthetic per-frame data for testing."""
        rng = np.random.RandomState(42)
        self.frame_data, self.poses, self.focals = [], [], []
        all_pts, all_colors = [], []
        for i in range(len(self.keyframes)):
            n = 2000
            pts = rng.randn(n, 3).astype(np.float32)
            pts[:, 0] += i * 0.3
            pts[:, 2] += 3
            colors = rng.rand(n, 3).astype(np.float32)
            conf = rng.rand(n).astype(np.float32) * 3
            pose = np.eye(4); pose[0, 3] = i * 0.1
            image_rgb = cv2.cvtColor(self.keyframes[i], cv2.COLOR_BGR2RGB)
            fd = FrameData(i, pts, colors, conf, pose, 500.0,
                           np.array([320., 240.]), image_rgb)
            self.frame_data.append(fd)
            self.poses.append(pose.tolist())
            self.focals.append(500.0)
            all_pts.append(pts)
            all_colors.append(colors)
        self.point_cloud = np.concatenate(all_pts, axis=0)
        self.point_colors = np.concatenate(all_colors, axis=0)
        print(f"MOCK PASS: {self.point_cloud.shape[0]} points, "
              f"{len(self.frame_data)} frames")
        return True

    # =====================================================================
    # STEP 2b: SLIDING-WINDOW MASt3R RECONSTRUCTION
    # =====================================================================
    def step_2_sliding_window_reconstruction(self, strategy="swin-3",
                                              image_size=512, min_points=100,
                                              window_sec=5.0, overlap_sec=1.0):
        """MASt3R reconstruction via sliding windows for high-FPS demos.

        Splits keyframes into overlapping time windows, runs MASt3R
        independently on each, then stitches them together using
        rigid alignment on the overlapping frames.

        Args:
            strategy: MASt3R pair strategy within each window.
            image_size: Image resize for MASt3R.
            min_points: Minimum total points for success.
            window_sec: Window duration in seconds (default 5.0).
            overlap_sec: Overlap between consecutive windows (default 1.0).
        """
        n_kf = len(self.keyframe_paths)
        print(f"--- Step 2: Sliding-Window MASt3R ({n_kf} keyframes, "
              f"window={window_sec}s, overlap={overlap_sec}s) ---")

        if not self.keyframe_paths:
            print("FAIL: No keyframes. Run Step 1 first.")
            return False
        if not _HAS_MAST3R:
            print("MOCK: MASt3R not available.")
            return self._mock_geometry()
        if not os.path.isfile(MAST3R_CHECKPOINT):
            print(f"FAIL: Checkpoint not found at {MAST3R_CHECKPOINT}")
            return False

        # Convert keyframe timestamps to seconds
        kf_times = [t / self.video_fps for t in self.keyframe_timestamps]
        total_duration = kf_times[-1] if kf_times else 0

        # Build windows: each window covers [start_sec, start_sec + window_sec)
        stride_sec = window_sec - overlap_sec
        windows = []
        start = 0.0
        while start < total_duration:
            end = min(start + window_sec, total_duration + 0.01)
            # Find keyframe indices in this window
            win_indices = [
                i for i, t in enumerate(kf_times) if start <= t < end
            ]
            # Make sure we always include at least the boundary frame
            if not win_indices:
                start += stride_sec
                continue
            # Extend to include 1 frame past boundary for overlap continuity
            if win_indices[-1] + 1 < n_kf and win_indices[-1] + 1 not in win_indices:
                next_t = kf_times[win_indices[-1] + 1]
                if next_t < end + overlap_sec:
                    win_indices.append(win_indices[-1] + 1)
            windows.append(win_indices)
            start += stride_sec
            if end >= total_duration:
                break

        # Minimum 2 frames per window
        windows = [w for w in windows if len(w) >= 2]

        if not windows:
            print("FAIL: No valid windows. Falling back to standard mode.")
            return self.step_2_geometric_reconstruction(strategy=strategy)

        print(f"  {len(windows)} windows: " +
              ", ".join(f"W{i}[{len(w)}fr, {kf_times[w[0]]:.1f}-{kf_times[w[-1]]:.1f}s]"
                        for i, w in enumerate(windows)))

        # Load MASt3R model once
        print("  Loading MASt3R model...")
        model = load_model(MAST3R_CHECKPOINT, DEVICE)

        # Process each window
        window_results = []  # list of per-window FrameData lists
        for wi, win_idxs in enumerate(windows):
            win_paths = [self.keyframe_paths[i] for i in win_idxs]
            print(f"  Window {wi}/{len(windows)-1}: "
                  f"{len(win_idxs)} frames [{kf_times[win_idxs[0]]:.1f}s - "
                  f"{kf_times[win_idxs[-1]]:.1f}s]")

            imgs = load_images(win_paths, size=image_size, verbose=False)
            pairs = make_pairs(imgs, scene_graph=strategy, symmetrize=True)
            if not pairs:
                print(f"    WARN: No pairs for window {wi}, skipping.")
                window_results.append([])
                continue

            print(f"    {len(pairs)} pairs, aligning...")
            win_cache = os.path.join(self.cache_dir, f"window_{wi:03d}")
            os.makedirs(win_cache, exist_ok=True)
            scene = sparse_global_alignment(
                win_paths, pairs, win_cache, model, device=DEVICE,
            )

            focals_t = to_numpy(scene.get_focals().cpu())
            cams2world_t = to_numpy(scene.get_im_poses().cpu())
            pp_t = to_numpy(scene.get_principal_points().cpu())
            pts3d_list, depthmaps_list, confs_list = to_numpy(
                scene.get_dense_pts3d(clean_depth=True)
            )

            min_conf = 1.5
            win_frame_data = []
            for li, gi in enumerate(win_idxs):
                pose = cams2world_t[li]
                focal = float(focals_t[li])
                pp = pp_t[li]
                pts = pts3d_list[li]
                conf = confs_list[li]
                depth = depthmaps_list[li] if depthmaps_list is not None else None

                pts_flat = pts.reshape(-1, 3)
                conf_flat = conf.reshape(-1)
                n = min(len(pts_flat), len(conf_flat))
                pts_flat, conf_flat = pts_flat[:n], conf_flat[:n]
                mask = conf_flat > min_conf
                valid_pts = pts_flat[mask]

                if li < len(scene.imgs):
                    img_f = scene.imgs[li].reshape(-1, 3)[:n]
                    valid_colors = img_f[mask]
                else:
                    valid_colors = np.ones_like(valid_pts) * 0.5

                image_rgb = cv2.cvtColor(self.keyframes[gi], cv2.COLOR_BGR2RGB)
                depth_2d = None
                if depth is not None:
                    if depth.ndim == 1:
                        h, w = image_rgb.shape[:2]
                        if len(depth) == h * w:
                            depth_2d = depth.reshape(h, w).astype(np.float32)
                    else:
                        depth_2d = depth.astype(np.float32)

                fd = FrameData(
                    index=gi,
                    pts3d=valid_pts.astype(np.float32),
                    colors=valid_colors.astype(np.float32),
                    confidence=conf_flat[mask].astype(np.float32),
                    pose=pose.astype(np.float64),
                    focal=focal,
                    principal_point=pp.astype(np.float64),
                    image_rgb=image_rgb,
                    depth_map=depth_2d,
                )
                win_frame_data.append(fd)

            n_pts_win = sum(fd.pts3d.shape[0] for fd in win_frame_data)
            print(f"    {n_pts_win:,} points from {len(win_frame_data)} frames")
            window_results.append(win_frame_data)

        # =====================================================================
        # Stitch windows using rigid alignment on overlapping frames
        # =====================================================================
        print("  Stitching windows via overlap alignment...")
        if not window_results or not window_results[0]:
            print("FAIL: First window produced no data.")
            return False

        # Start with the first window as the reference frame
        stitched = {}  # global_index -> FrameData (in global coords)
        for fd in window_results[0]:
            stitched[fd.index] = fd

        for wi in range(1, len(window_results)):
            if not window_results[wi]:
                continue

            curr_frames = window_results[wi]
            curr_indices = {fd.index for fd in curr_frames}

            # Find overlapping global indices (present in both stitched + current)
            overlap_indices = sorted(curr_indices & set(stitched.keys()))

            if len(overlap_indices) >= 2:
                # Compute rigid transform from current window to stitched coords
                # using camera positions of overlapping frames
                pts_ref = np.array([stitched[gi].pose[:3, 3]
                                    for gi in overlap_indices], dtype=np.float64)
                pts_cur = np.array([next(fd for fd in curr_frames
                                         if fd.index == gi).pose[:3, 3]
                                    for gi in overlap_indices], dtype=np.float64)

                R, t, s = self._rigid_align(pts_cur, pts_ref)
                print(f"    W{wi}: {len(overlap_indices)} overlap frames, "
                      f"scale={s:.4f}, shift={np.linalg.norm(t):.3f}")
            elif len(overlap_indices) == 1:
                # Single overlap: translate only
                gi = overlap_indices[0]
                ref_pos = stitched[gi].pose[:3, 3]
                cur_fd = next(fd for fd in curr_frames if fd.index == gi)
                offset = ref_pos - cur_fd.pose[:3, 3]
                R = np.eye(3)
                t = offset
                s = 1.0
                print(f"    W{wi}: 1 overlap frame, translation-only offset={np.linalg.norm(t):.3f}")
            else:
                # No overlap: just concatenate (will be slightly misaligned)
                R = np.eye(3)
                t = np.zeros(3)
                s = 1.0
                print(f"    W{wi}: no overlap, appending unaligned")

            # Apply transform to all frames in this window
            for fd in curr_frames:
                if fd.index in stitched:
                    # Overlap frame: average the point clouds
                    existing = stitched[fd.index]
                    new_pts = (s * (fd.pts3d @ R.T) + t).astype(np.float32)
                    combined_pts = np.concatenate([existing.pts3d, new_pts])
                    combined_cols = np.concatenate([existing.colors, fd.colors])
                    combined_conf = np.concatenate([existing.confidence, fd.confidence])
                    # Downsample if too many points
                    if combined_pts.shape[0] > 200000:
                        idx = self.rng.choice(combined_pts.shape[0], 200000, replace=False)
                        combined_pts = combined_pts[idx]
                        combined_cols = combined_cols[idx]
                        combined_conf = combined_conf[idx]
                    existing.pts3d = combined_pts
                    existing.colors = combined_cols
                    existing.confidence = combined_conf
                else:
                    # New frame: transform and add
                    new_pts = (s * (fd.pts3d @ R.T) + t).astype(np.float32)
                    new_pose = fd.pose.copy()
                    new_pose[:3, :3] = R @ fd.pose[:3, :3]
                    new_pose[:3, 3] = s * (R @ fd.pose[:3, 3]) + t
                    fd.pts3d = new_pts
                    fd.pose = new_pose
                    stitched[fd.index] = fd

        # Build final frame_data sorted by global index
        self.frame_data = [stitched[gi] for gi in sorted(stitched.keys())]
        self.poses = [fd.pose.tolist() for fd in self.frame_data]
        self.focals = [fd.focal for fd in self.frame_data]

        all_pts = [fd.pts3d for fd in self.frame_data]
        all_colors = [fd.colors for fd in self.frame_data]
        self.point_cloud = np.concatenate(all_pts, axis=0) if all_pts else np.empty((0, 3))
        self.point_colors = np.concatenate(all_colors, axis=0) if all_colors else np.empty((0, 3))

        n_pts = self.point_cloud.shape[0]
        per_frame = [fd.pts3d.shape[0] for fd in self.frame_data]
        effective_fps = len(self.frame_data) / max(total_duration, 0.001)

        if n_pts < min_points:
            print(f"FAIL: Only {n_pts} points (need {min_points}).")
            return False

        print(f"PASS: {n_pts:,} points across {len(self.frame_data)} frames "
              f"({effective_fps:.1f} FPS, "
              f"per-frame: {min(per_frame)}-{max(per_frame)}, "
              f"avg={int(np.mean(per_frame))})")
        return True

    @staticmethod
    def _rigid_align(src, dst):
        """Compute rigid alignment (R, t, s) such that dst ~ s * R @ src + t.

        Uses Umeyama's method (SVD-based) for similarity transform.
        Returns (R, t, s) where R is 3x3 rotation, t is 3-vector, s is scale.
        """
        assert src.shape == dst.shape and src.shape[1] == 3
        n = src.shape[0]
        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)
        src_c = src - mu_src
        dst_c = dst - mu_dst
        var_src = np.sum(src_c ** 2) / n

        H = (dst_c.T @ src_c) / n
        U, D, Vt = np.linalg.svd(H)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1
        R = U @ S @ Vt
        s = np.trace(np.diag(D) @ S) / max(var_src, 1e-12)
        t = mu_dst - s * R @ mu_src
        return R, t, s

    # =====================================================================
    # STEP 3a: YOLOv8 OBJECT DETECTION (fast, per-keyframe)
    # =====================================================================
    def step_3a_yolo_detection(self):
        """Run YOLOv8 on all keyframes for fast object detection."""
        print("--- Step 3a: YOLOv8 Object Detection ---")
        if not _HAS_MODEL_INTERFACES or not _HAS_YOLO:
            print("  SKIP: YOLOv8 not available (ultralytics not installed).")
            return

        try:
            detector = YOLODetector(model_name="yolov8x-seg.pt", device=DEVICE)
        except Exception as e:
            print(f"  WARN: Could not load YOLOv8: {e}")
            return

        self.yolo_detections = []
        for i, kf in enumerate(self.keyframes):
            timestamp = (self.keyframe_timestamps[i] / self.video_fps
                         if i < len(self.keyframe_timestamps) else 0.0)
            det = detector.process_frame(kf, frame_idx=i, timestamp=timestamp)
            self.yolo_detections.append(det)

        # Summary
        all_classes = detector.get_unique_classes(self.yolo_detections)
        freq = detector.get_class_frequencies(self.yolo_detections)
        total_dets = sum(len(d.class_names) for d in self.yolo_detections)
        print(f"  PASS: {total_dets} detections across {len(self.yolo_detections)} frames")
        print(f"  Classes: {', '.join(f'{k}({v})' for k, v in list(freq.items())[:10])}")

    # =====================================================================
    # STEP 3b: SAM3 VIDEO SEGMENTATION (pixel-perfect + tracking)
    # =====================================================================
    def step_3b_sam3_segmentation(self):
        """Run SAM3 video segmentation using YOLO class names as text prompts."""
        print("--- Step 3b: SAM3 Video Segmentation ---")
        if not _HAS_MODEL_INTERFACES or not _HAS_SAM3:
            print("  SKIP: SAM3 not available.")
            return
        if not self.keyframes:
            print("  SKIP: No keyframes.")
            return

        # Get text prompts from YOLO detections
        if self.yolo_detections:
            detector_tmp = YOLODetector.__new__(YOLODetector) if _HAS_YOLO else None
            if detector_tmp and hasattr(detector_tmp, 'get_unique_classes'):
                text_prompts = sorted(set(
                    name for det in self.yolo_detections
                    for name in det.class_names
                ))[:8]  # limit to top 8 classes
            else:
                text_prompts = sorted(set(
                    name for det in self.yolo_detections
                    for name in det.class_names
                ))[:8]
        else:
            text_prompts = ["object"]  # generic fallback

        if not text_prompts:
            print("  SKIP: No text prompts from YOLO detections.")
            return

        print(f"  Text prompts from YOLO: {text_prompts}")

        try:
            segmenter = SAM3Segmenter(device=DEVICE)
            # Convert keyframes to RGB
            frames_rgb = [cv2.cvtColor(kf, cv2.COLOR_BGR2RGB) for kf in self.keyframes]
            self.sam3_segmentations = segmenter.segment_video_with_text(
                frames_rgb, text_prompts, video_fps=self.video_fps,
                max_frames=len(frames_rgb),
            )
            total_masks = sum(len(s.masks) for s in self.sam3_segmentations)
            print(f"  PASS: {total_masks} masks across "
                  f"{len(self.sam3_segmentations)} frames")
        except Exception as e:
            print(f"  WARN: SAM3 segmentation failed: {e}")
            self.sam3_segmentations = []

    # =====================================================================
    # STEP 3c: GEMINI VIDEO ANALYSIS (full video upload)
    # =====================================================================
    def step_3c_gemini_video_analysis(self):
        """Upload full video to Gemini for holistic scene understanding."""
        print("--- Step 3c: Gemini Video Analysis (full video upload) ---")
        if not _HAS_MODEL_INTERFACES or not _HAS_GEMINI:
            print("  SKIP: google-genai not installed.")
            return

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("  SKIP: No Gemini API key.")
            return

        try:
            analyzer = GeminiVideoAnalyzer(api_key=api_key)

            # If we have YOLO detections, give Gemini context
            if self.yolo_detections:
                mast3r_summary = {
                    "num_points": int(self.point_cloud.shape[0])
                        if self.point_cloud is not None else 0,
                    "num_frames": len(self.frame_data),
                }
                self.scene_description = analyzer.analyze_scene_with_context(
                    self.video_path, self.yolo_detections, mast3r_summary
                )
            else:
                self.scene_description = analyzer.analyze_video(self.video_path)

            n_obj = len(self.scene_description.objects) if self.scene_description else 0
            n_evt = len(self.scene_description.events) if self.scene_description else 0
            print(f"  PASS: Gemini found {n_obj} objects, {n_evt} events")
            if self.scene_description and self.scene_description.narrative:
                narr = self.scene_description.narrative[:200]
                print(f"  Narrative: {narr}...")

        except Exception as e:
            print(f"  WARN: Gemini video analysis failed: {e}")
            self.scene_description = None

    # =====================================================================
    # STEP 4 (v2): 4D SCENE FUSION + REASONING
    # =====================================================================
    def step_4_scene_fusion_and_reasoning(self):
        """Fuse all model outputs into 4D scene graph + cross-validate."""
        print("--- Step 4: 4D Scene Fusion + Reasoning ---")

        if not _HAS_FUSION:
            print("  WARN: scene_fusion not available, falling back to legacy.")
            self.step_3_semantic_detection()
            self.step_4_causal_reasoning()
            return

        # Run fusion
        fusion = SceneFusion4D(
            frame_data=self.frame_data,
            detections=self.yolo_detections,
            segmentations=self.sam3_segmentations,
            scene_description=self.scene_description,
            reasoning=None,  # will add after reasoning step
        )
        self.objects_4d = fusion.fuse()

        # Convert SceneObject4D to Object3D for backward compatibility
        self.objects_3d = []
        for obj4d in self.objects_4d:
            obj3d = Object3D(
                entity=obj4d.entity_id,
                obj_type=obj4d.obj_type,
                bbox_3d_min=obj4d.bbox_3d_min,
                bbox_3d_max=obj4d.bbox_3d_max,
                component_type=obj4d.component_type,
                initial_state=obj4d.initial_state,
                final_state=obj4d.final_state,
                state_changes=obj4d.state_changes,
                first_seen_frame=obj4d.first_seen_frame,
                last_seen_frame=obj4d.last_seen_frame,
            )
            self.objects_3d.append(obj3d)

        # Run reasoning engine (cross-model validation)
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key and _HAS_GEMINI:
            try:
                # Try OpenAI first if available, else Gemini
                openai_key = os.environ.get("OPENAI_API_KEY")
                if openai_key and openai_key.startswith("sk-"):
                    try:
                        engine = ReasoningEngine(provider="openai", api_key=openai_key)
                    except ImportError:
                        engine = ReasoningEngine(provider="gemini", api_key=api_key)
                else:
                    engine = ReasoningEngine(provider="gemini", api_key=api_key)

                self.reasoning_result = engine.synthesize(
                    detections=self.yolo_detections,
                    segmentations=self.sam3_segmentations,
                    scene=self.scene_description,
                    objects_3d=self.objects_3d,
                )

                # Apply reasoning results back to objects
                fusion.reasoning = self.reasoning_result
                fusion._apply_reasoning()
                fusion._compute_confidences()
                fusion._flag_for_review()

                # Update Object3D list from enriched 4D objects
                for i, obj4d in enumerate(self.objects_4d):
                    if i < len(self.objects_3d):
                        self.objects_3d[i].component_type = obj4d.component_type
                        self.objects_3d[i].initial_state = obj4d.initial_state
                        self.objects_3d[i].final_state = obj4d.final_state
                        self.objects_3d[i].state_changes = obj4d.state_changes

                print(f"  Reasoning: {len(self.reasoning_result.verified_objects)} "
                      f"verified, {len(self.reasoning_result.human_review_flags)} flagged")

            except Exception as e:
                print(f"  WARN: Reasoning failed: {e}")

        # Evaluate quality
        try:
            evaluator = GroundTruthEvaluator()
            self.evaluation = evaluator.evaluate(self.objects_4d)
            print(f"  Quality score: {self.evaluation.get('overall_score', 0):.2f}")
            for suggestion in self.evaluation.get("improvement_suggestions", []):
                print(f"    -> {suggestion}")
        except Exception as e:
            print(f"  WARN: Evaluation failed: {e}")

        # Store scene graph for backward compat
        self.scene_graph = {
            "objects": [
                {"entity": o.entity_id, "type": o.obj_type,
                 "component_type": o.component_type,
                 "initial_state": o.initial_state,
                 "final_state": o.final_state,
                 "state_changes": o.state_changes}
                for o in self.objects_4d
            ]
        }

        print(f"  PASS: {len(self.objects_4d)} objects in 4D scene graph")

    # =====================================================================
    # STEP 3 (LEGACY): SEMANTIC 3D OBJECT DETECTION (Gemini Vision)
    # Replaces SAM 3 stub -- uses Gemini to detect objects with 2D bboxes,
    # then back-projects to 3D using MASt3R per-frame point clouds.
    # =====================================================================
    def step_3_semantic_detection(self):
        """Detect objects in keyframes via Gemini, project to 3D."""
        print("--- Step 3: Semantic 3D Object Detection (Gemini Vision) ---")

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key or not _HAS_GEMINI:
            print("  SKIP: No API key or google-genai not installed.")
            return
        if not self.frame_data:
            print("  SKIP: No 3D frame data. Run Step 2 first.")
            return

        client = genai.Client(api_key=api_key)

        # Send up to 6 evenly-spaced keyframes for better coverage
        n_kf = len(self.keyframe_paths)
        if n_kf <= 6:
            indices = list(range(n_kf))
        else:
            step = max(1, n_kf // 6)
            indices = list(range(0, n_kf, step))[:6]
            if indices[-1] != n_kf - 1:
                indices[-1] = n_kf - 1  # always include last
        indices = sorted(set(indices))

        prompt = """Detect all distinct objects in these images from a video.
For EACH object, provide:
- entity: unique name like "Door_01", "Table_01", "Chair_01"
- type: category (door, table, chair, cup, window, cabinet, shelf, person, etc.)
- bbox: [x_min, y_min, x_max, y_max] as percentages 0-100 of image width/height
- frame_idx: which image (0-indexed) you see it in most clearly

Return ONLY valid JSON:
{
  "detections": [
    {"entity": "Door_01", "type": "door", "bbox": [10, 5, 30, 95], "frame_idx": 0},
    {"entity": "Table_01", "type": "table", "bbox": [40, 50, 80, 90], "frame_idx": 1}
  ]
}
"""
        contents = []
        for idx in indices:
            with open(self.keyframe_paths[idx], "rb") as f:
                contents.append(genai.types.Part.from_bytes(
                    data=f.read(), mime_type="image/jpeg"
                ))
        contents.append(prompt)

        print(f"  Sending {len(indices)} keyframes to Gemini for detection...")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=contents,
            )
            result_text = response.text.strip()

            # Parse JSON
            json_str = result_text
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                start = 1 if lines[0].startswith("```") else 0
                end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
                json_str = "\n".join(lines[start:end])

            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                detections = parsed
            elif isinstance(parsed, dict):
                detections = parsed.get("detections", parsed.get("objects", []))
            else:
                detections = []
            print(f"  Gemini detected {len(detections)} objects in 2D")

            # Back-project each detection to 3D and fuse temporally per-entity.
            observations = []
            for det in detections:
                obj3d = self._backproject_bbox_to_3d(det, indices)
                if obj3d is None:
                    continue
                conf = det.get("confidence", det.get("score", 1.0))
                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = 1.0
                conf = float(np.clip(conf, 0.05, 1.0))
                observations.append((obj3d, conf))

            if not observations:
                print("  WARN: No 3D objects could be back-projected.")
                self.objects_3d = []
                return

            tracks = {}
            unknown_counters = defaultdict(int)

            for obj3d, conf in observations:
                raw_name = self._safe_name(obj3d.entity)
                if raw_name.lower() in {"", "unknown", "object"}:
                    typ = self._safe_name(obj3d.obj_type).lower()
                    unknown_counters[typ] += 1
                    raw_name = f"{typ}_{unknown_counters[typ]:02d}"
                track_key = raw_name.lower()

                if track_key not in tracks:
                    pf = ParticleFilter3D(
                        num_particles=256,
                        process_noise=0.04,
                        measurement_noise=0.20,
                        rng=self.rng,
                    )
                    pf.initialize(obj3d.center)
                    tracks[track_key] = {
                        "entity": raw_name,
                        "type": obj3d.obj_type,
                        "bbox_min": obj3d.bbox_3d_min.copy(),
                        "bbox_max": obj3d.bbox_3d_max.copy(),
                        "first_seen": obj3d.first_seen_frame,
                        "last_seen": obj3d.last_seen_frame,
                        "obs_count": 1,
                        "pf": pf,
                    }
                else:
                    tr = tracks[track_key]
                    tr["pf"].predict()
                    tr["pf"].update(obj3d.center, measurement_confidence=conf)
                    tr["bbox_min"] = np.minimum(tr["bbox_min"], obj3d.bbox_3d_min)
                    tr["bbox_max"] = np.maximum(tr["bbox_max"], obj3d.bbox_3d_max)
                    tr["first_seen"] = min(tr["first_seen"], obj3d.first_seen_frame)
                    tr["last_seen"] = max(tr["last_seen"], obj3d.last_seen_frame)
                    tr["obs_count"] += 1

            self.objects_3d = []
            for tr in tracks.values():
                center = tr["pf"].estimate()
                spread = tr["pf"].spread()
                size = np.maximum(tr["bbox_max"] - tr["bbox_min"], 0.05)
                bbox_min = center - (size / 2.0)
                bbox_max = center + (size / 2.0)
                track_conf = float(np.clip(np.exp(-spread * 5.0), 0.0, 1.0))

                fused = Object3D(
                    entity=tr["entity"],
                    obj_type=tr["type"],
                    bbox_3d_min=bbox_min,
                    bbox_3d_max=bbox_max,
                    first_seen_frame=tr["first_seen"],
                    last_seen_frame=tr["last_seen"],
                    observation_count=tr["obs_count"],
                    tracking_confidence=track_conf,
                )
                self.objects_3d.append(fused)

            self.objects_3d.sort(key=lambda o: o.entity.lower())

            print(f"  PASS: {len(self.objects_3d)} objects positioned in 3D "
                  f"(particle-filter fused)")
            for obj3d in self.objects_3d:
                print(f"    - {obj3d.entity} ({obj3d.obj_type}): "
                      f"center={np.round(obj3d.center, 2).tolist()}, "
                      f"size={np.round(obj3d.size, 2).tolist()}, "
                      f"obs={obj3d.observation_count}, "
                      f"track_conf={obj3d.tracking_confidence:.2f}")

        except Exception as e:
            print(f"  WARN: Detection failed: {e}")
            self.objects_3d = []

    def _backproject_bbox_to_3d(self, detection, frame_indices):
        """Back-project a 2D bounding box to a 3D bounding box.

        Uses the per-frame 3D point cloud: find all 3D points whose
        2D projection falls within the bbox, compute their 3D extent.
        """
        bbox = detection.get("bbox", [0, 0, 100, 100])  # % of image
        frame_idx_in_set = detection.get("frame_idx", 0)
        try:
            frame_idx_in_set = int(frame_idx_in_set)
        except (TypeError, ValueError):
            frame_idx_in_set = 0

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            bbox = [0, 0, 100, 100]
        try:
            bbox = [float(v) for v in bbox]
        except (TypeError, ValueError):
            bbox = [0, 0, 100, 100]
        bbox = [float(np.clip(v, 0.0, 100.0)) for v in bbox]
        if bbox[0] > bbox[2]:
            bbox[0], bbox[2] = bbox[2], bbox[0]
        if bbox[1] > bbox[3]:
            bbox[1], bbox[3] = bbox[3], bbox[1]

        # Map back to actual keyframe index
        if frame_idx_in_set < len(frame_indices):
            kf_idx = frame_indices[frame_idx_in_set]
        else:
            kf_idx = 0

        if kf_idx >= len(self.frame_data):
            return None

        fd = self.frame_data[kf_idx]
        if fd.pts3d.shape[0] == 0:
            return None

        h, w = fd.image_rgb.shape[:2]
        x_min = bbox[0] / 100.0 * w
        y_min = bbox[1] / 100.0 * h
        x_max = bbox[2] / 100.0 * w
        y_max = bbox[3] / 100.0 * h

        # Project 3D points to 2D using camera intrinsics + pose
        pts_world = fd.pts3d  # (N, 3)
        pose_inv = np.linalg.inv(fd.pose)  # world2cam
        pts_cam = (pose_inv[:3, :3] @ pts_world.T).T + pose_inv[:3, 3]

        # Only keep points in front of camera
        valid_z = pts_cam[:, 2] > 0.01
        if not np.any(valid_z):
            return None

        pts_cam_valid = pts_cam[valid_z]
        pts_world_valid = pts_world[valid_z]

        # Project to pixel coords
        fx = fd.focal
        fy = fd.focal
        cx, cy = fd.principal_point
        u = fx * pts_cam_valid[:, 0] / pts_cam_valid[:, 2] + cx
        v = fy * pts_cam_valid[:, 1] / pts_cam_valid[:, 2] + cy

        # Find points inside the 2D bbox
        in_bbox = (u >= x_min) & (u <= x_max) & (v >= y_min) & (v <= y_max)
        if np.sum(in_bbox) < 5:
            return None

        pts_in_box = pts_world_valid[in_bbox]

        # 3D bounding box from the inlier points (robust: use 5-95 percentile)
        bbox_min = np.percentile(pts_in_box, 5, axis=0)
        bbox_max = np.percentile(pts_in_box, 95, axis=0)

        return Object3D(
            entity=detection.get("entity", "Unknown"),
            obj_type=detection.get("type", "unknown"),
            bbox_3d_min=bbox_min,
            bbox_3d_max=bbox_max,
            first_seen_frame=kf_idx,
            last_seen_frame=kf_idx,
        )

    # =====================================================================
    # STEP 4: CAUSAL REASONING (Gemini)
    # Merges physics/state info into the 3D objects from Step 3.
    # =====================================================================
    def step_4_causal_reasoning(self):
        """Gemini analyzes state changes and assigns physics metadata."""
        print("--- Step 4: Gemini Causal Reasoning ---")

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key or not _HAS_GEMINI:
            print("  SKIP: No API key or google-genai not installed.")
            return
        if not self.keyframe_paths:
            print("  SKIP: No keyframes.")
            return

        client = genai.Client(api_key=api_key)

        # Build object context for the prompt
        obj_list = ", ".join(
            f"{o.entity} ({o.obj_type})" for o in self.objects_3d
        ) if self.objects_3d else "unknown objects"

        prompt = f"""Analyze this video sequence for object state changes.
Known objects in the scene: {obj_list}

For each object determine:
1. Physics joint type: RevoluteJoint (doors, lids), PrismaticJoint (drawers, sliders), FixedJoint (tables, walls)
2. Initial state (e.g., closed, open, stationary, on_table)
3. Final state
4. Any state changes with timing (early/mid/late) and cause

Return ONLY valid JSON:
{{
  "objects": [
    {{
      "entity": "Door_01",
      "type": "door",
      "component_type": "RevoluteJoint",
      "initial_state": "closed",
      "final_state": "open",
      "state_changes": [
        {{"time": "mid", "from": "closed", "to": "open", "cause": "person pushed"}}
      ]
    }}
  ]
}}
"""
        contents = []
        step = max(1, len(self.keyframe_paths) // 5)
        for kf_path in self.keyframe_paths[::step][:6]:
            with open(kf_path, "rb") as f:
                contents.append(genai.types.Part.from_bytes(
                    data=f.read(), mime_type="image/jpeg"
                ))
        contents.append(prompt)

        print(f"  Sending {min(6, len(self.keyframe_paths))} keyframes + "
              f"{len(self.objects_3d)} object context...")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=contents,
            )
            result_text = response.text.strip()
            json_str = result_text
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                start = 1 if lines[0].startswith("```") else 0
                end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
                json_str = "\n".join(lines[start:end])

            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                self.scene_graph = {"objects": parsed}
            elif isinstance(parsed, dict):
                self.scene_graph = parsed
            else:
                self.scene_graph = {"objects": []}

            # Merge reasoning results into Object3D instances
            objects_list = self.scene_graph.get("objects", [])
            reasoning_map = {}
            reasoning_by_type = defaultdict(list)
            for obj in objects_list:
                if not isinstance(obj, dict):
                    continue
                key = obj.get("entity", "")
                if key:
                    reasoning_map[self._safe_name(key).lower()] = obj
                otype = obj.get("type", "")
                if otype:
                    reasoning_by_type[str(otype).lower()].append(obj)

            merged = 0
            for obj3d in self.objects_3d:
                r = reasoning_map.get(self._safe_name(obj3d.entity).lower())
                if not r:
                    # Only fallback by type when unambiguous.
                    candidates = reasoning_by_type.get(str(obj3d.obj_type).lower(), [])
                    if len(candidates) == 1:
                        r = candidates[0]
                if r:
                    obj3d.component_type = r.get("component_type", obj3d.component_type)
                    obj3d.initial_state = r.get("initial_state", obj3d.initial_state)
                    obj3d.final_state = r.get("final_state", obj3d.final_state)
                    obj3d.state_changes = r.get("state_changes", [])
                    merged += 1

            print(f"  PASS: Reasoning merged {merged}/{len(self.objects_3d)} objects.")
            for obj3d in self.objects_3d:
                print(f"    - {obj3d.entity}: {obj3d.component_type} "
                      f"({obj3d.initial_state} -> {obj3d.final_state})")

        except Exception as e:
            print(f"  WARN: Reasoning failed: {e}")
            self.scene_graph = {"objects": []}

    # =====================================================================
    # STEP 5: OPENUSD EXPORT (with physics joint metadata)
    # =====================================================================
    def step_5_export_usd(self):
        """Export scene to OpenUSD with positioned objects + physics joints."""
        print(f"--- Step 5: Writing OpenUSD to {self.output_path} ---")

        if os.path.isfile(self.output_path):
            os.remove(self.output_path)

        stage = Usd.Stage.CreateNew(self.output_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        UsdGeom.Xform.Define(stage, "/World")

        # --- Point Cloud (with colors) ---
        if self.point_cloud is not None and self.point_cloud.shape[0] > 0:
            pc_prim = UsdGeom.Points.Define(stage, "/World/PointCloud")
            pts = self.point_cloud
            cols = self.point_colors
            if pts.shape[0] > 50000:
                idx = self.rng.choice(pts.shape[0], 50000, replace=False)
                pts = pts[idx]
                if cols is not None and cols.shape[0] > 50000:
                    cols = cols[idx]
            pc_prim.GetPointsAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts.tolist()])
            )
            pc_prim.GetWidthsAttr().Set(Vt.FloatArray([0.005] * len(pts)))
            # Per-point colors
            if cols is not None and cols.shape[0] == pts.shape[0]:
                display_colors = Vt.Vec3fArray(
                    [Gf.Vec3f(*c.tolist()) for c in np.clip(cols, 0, 1)]
                )
                pc_prim.GetDisplayColorAttr().Set(display_colors)

        # --- Cameras ---
        for i, pose in enumerate(self.poses):
            cam_path = f"/World/Cameras/Cam_{i:02d}"
            cam_xform = UsdGeom.Xform.Define(stage, cam_path)
            pose_np = np.array(pose)
            mat = Gf.Matrix4d(*pose_np.flatten().tolist())
            cam_xform.AddTransformOp().Set(mat)

            if i < len(self.focals):
                cam = UsdGeom.Camera.Define(stage, f"{cam_path}/Camera")
                cam.GetFocalLengthAttr().Set(float(self.focals[i]) * 0.036)

        # --- 3D Objects with physics ---
        used_names = set()
        for obj3d in self.objects_3d:
            entity_safe = obj3d.entity.replace(" ", "_").replace("/", "_")
            # Deduplicate names for USD
            base_name = entity_safe
            suffix = 1
            while entity_safe in used_names:
                entity_safe = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(entity_safe)
            obj_path = f"/World/Objects/{entity_safe}"
            obj_xform = UsdGeom.Xform.Define(stage, obj_path)

            # Position at 3D center
            center = obj3d.center
            obj_xform.AddTranslateOp().Set(Gf.Vec3d(*center.tolist()))

            # Cube sized to bounding box
            cube = UsdGeom.Cube.Define(stage, f"{obj_path}/Shape")
            size = obj3d.size
            max_dim = max(float(np.max(size)), 0.1)
            cube.GetSizeAttr().Set(float(max_dim))
            sx = float(size[0]) / max_dim if max_dim > 0 else 1
            sy = float(size[1]) / max_dim if max_dim > 0 else 1
            sz = float(size[2]) / max_dim if max_dim > 0 else 1
            cube.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))

            # Metadata
            prim = obj_xform.GetPrim()
            prim.SetMetadata("displayName", f"{obj3d.entity} ({obj3d.obj_type})")

            # Custom attributes for scene graph data
            prim.CreateAttribute("world2data:type",
                                 Sdf.ValueTypeNames.String).Set(obj3d.obj_type)
            prim.CreateAttribute("world2data:component_type",
                                 Sdf.ValueTypeNames.String).Set(obj3d.component_type)
            prim.CreateAttribute("world2data:initial_state",
                                 Sdf.ValueTypeNames.String).Set(obj3d.initial_state)
            prim.CreateAttribute("world2data:final_state",
                                 Sdf.ValueTypeNames.String).Set(obj3d.final_state)

            # v2: Add confidence and provenance from 4D fusion
            obj4d_match = None
            for o4d in self.objects_4d:
                if o4d.entity_id == obj3d.entity or o4d.obj_type == obj3d.obj_type:
                    obj4d_match = o4d
                    break

            if obj4d_match:
                prim.CreateAttribute("world2data:confidence",
                                     Sdf.ValueTypeNames.Float).Set(
                    float(obj4d_match.confidence))
                prim.CreateAttribute("world2data:detected_by",
                                     Sdf.ValueTypeNames.String).Set(
                    ",".join(obj4d_match.detected_by))
                prim.CreateAttribute("world2data:human_review",
                                     Sdf.ValueTypeNames.Bool).Set(
                    obj4d_match.human_review)
                if obj4d_match.review_reason:
                    prim.CreateAttribute("world2data:review_reason",
                                         Sdf.ValueTypeNames.String).Set(
                        obj4d_match.review_reason)
                if obj4d_match.reasoning_trace:
                    prim.CreateAttribute("world2data:reasoning_trace",
                                         Sdf.ValueTypeNames.String).Set(
                        obj4d_match.reasoning_trace[:500])  # truncate

            # USD Physics API
            if obj3d.component_type != "FixedJoint":
                UsdPhysics.RigidBodyAPI.Apply(prim)
                UsdPhysics.CollisionAPI.Apply(prim)

        stage.GetRootLayer().Save()
        print(f"PASS: Saved {self.output_path} "
              f"({len(self.objects_3d)} positioned objects with physics)")

        # JSON sidecar (enriched with 4D data when available)
        json_path = self._derive_output_path("_scene_graph.json")
        sidecar = {
            "scene_graph": self.scene_graph,
            "objects_3d": [o.to_dict() for o in self.objects_3d],
            "num_cameras": len(self.poses),
            "num_points": int(self.point_cloud.shape[0])
                if self.point_cloud is not None else 0,
            "num_frames": len(self.frame_data),
            "focals": self.focals,
            "video_fps": self.video_fps,
            "keyframe_timestamps": self.keyframe_timestamps,
        }

        # v2 enrichments
        if self.objects_4d:
            sidecar["objects_4d"] = [o.to_dict() for o in self.objects_4d]
        if self.scene_description and hasattr(self.scene_description, 'to_dict'):
            sidecar["gemini_scene"] = self.scene_description.to_dict()
        if self.yolo_detections:
            sidecar["yolo_summary"] = {
                "total_detections": sum(
                    len(d.class_names) for d in self.yolo_detections),
                "unique_classes": sorted(set(
                    n for d in self.yolo_detections for n in d.class_names)),
                "frames_processed": len(self.yolo_detections),
            }
        if self.sam3_segmentations:
            sidecar["sam3_summary"] = {
                "total_segments": int(sum(len(s.masks) for s in self.sam3_segmentations)),
                "frames_processed": len(self.sam3_segmentations),
                "unique_labels": sorted(set(
                    lbl for s in self.sam3_segmentations for lbl in s.labels
                )),
            }
        if self.evaluation:
            sidecar["evaluation"] = self.evaluation

        with open(json_path, "w") as f:
            json.dump(sidecar, f, indent=2, default=str)
        print(f"PASS: Saved scene graph -> {json_path}")

    # =====================================================================
    # TEMPORAL RERUN RECORDING (.rrd)
    # Now includes: depth maps, 3D bounding boxes, state change colors
    # =====================================================================
    def save_temporal_rrd(self, rrd_path):
        """Save interactive Rerun .rrd with full temporal 3D scene."""
        if not self.frame_data:
            print("SKIP: No temporal frame data to save.")
            return

        print(f"--- Saving temporal Rerun recording -> {rrd_path} ---")

        rr.init("World2Data", spawn=False)

        cam_positions = []
        accumulated_pts = []
        accumulated_colors = []

        # Color mapping for object types
        type_colors = {
            "door": [220, 40, 40],
            "table": [40, 80, 220],
            "chair": [40, 200, 200],
            "cup": [200, 200, 40],
            "window": [200, 100, 40],
            "cabinet": [40, 200, 40],
            "shelf": [140, 80, 200],
            "person": [255, 140, 0],
        }

        for fd in self.frame_data:
            video_frame = self.keyframe_timestamps[fd.index] \
                if fd.index < len(self.keyframe_timestamps) else fd.index
            rr.set_time("keyframe", sequence=fd.index)
            rr.set_time("video_time", duration=video_frame / self.video_fps)

            # --- Current frame point cloud ---
            if fd.pts3d.shape[0] > 0:
                colors_u8 = (np.clip(fd.colors, 0, 1) * 255).astype(np.uint8)
                rr.log("world/current_view", rr.Points3D(
                    fd.pts3d, colors=colors_u8, radii=0.003,
                ))

            # --- Accumulated point cloud ---
            accumulated_pts.append(fd.pts3d)
            accumulated_colors.append(fd.colors)
            acc_pts = np.concatenate(accumulated_pts, axis=0)
            acc_cols = np.concatenate(accumulated_colors, axis=0)
            if acc_pts.shape[0] > 200000:
                idx = self.rng.choice(acc_pts.shape[0], 200000, replace=False)
                acc_pts, acc_cols = acc_pts[idx], acc_cols[idx]
            rr.log("world/accumulated", rr.Points3D(
                acc_pts,
                colors=(np.clip(acc_cols, 0, 1) * 255).astype(np.uint8),
                radii=0.002,
            ))

            # --- Camera ---
            h, w = fd.image_rgb.shape[:2]
            rr.log("world/camera", rr.Transform3D(
                translation=fd.pose[:3, 3], mat3x3=fd.pose[:3, :3],
            ))
            rr.log("world/camera", rr.Pinhole(
                focal_length=fd.focal,
                principal_point=fd.principal_point,
                resolution=[w, h],
            ))
            rr.log("world/camera/image", rr.Image(fd.image_rgb))

            # --- Depth map ---
            if fd.depth_map is not None:
                rr.log("world/camera/depth", rr.DepthImage(fd.depth_map))

            # --- Camera trajectory ---
            cam_positions.append(fd.pose[:3, 3].tolist())
            if len(cam_positions) >= 2:
                rr.log("world/trajectory", rr.LineStrips3D(
                    [cam_positions], colors=[[255, 200, 0]], radii=0.005,
                ))

            # --- 3D Object bounding boxes with state change colors ---
            for obj3d in self.objects_3d:
                base_color = type_colors.get(obj3d.obj_type.lower(), [180, 180, 180])

                # Animate color based on state changes
                color = list(base_color)
                label = f"{obj3d.entity}: {obj3d.initial_state}"

                if obj3d.state_changes:
                    t_frac = fd.index / max(len(self.frame_data) - 1, 1)
                    for sc in obj3d.state_changes:
                        time_map = {"early": 0.25, "mid": 0.5, "late": 0.75}
                        sc_t = time_map.get(sc.get("time", "mid"), 0.5)
                        if t_frac >= sc_t:
                            color = [40, 255, 40]  # green = state changed
                            label = f"{obj3d.entity}: {sc.get('to', '?')}"
                elif obj3d.final_state != obj3d.initial_state:
                    t_frac = fd.index / max(len(self.frame_data) - 1, 1)
                    if t_frac > 0.5:
                        color = [40, 255, 40]
                        label = f"{obj3d.entity}: {obj3d.final_state}"

                rr.log(
                    f"world/objects/{self._safe_name(obj3d.entity)}",
                    rr.Boxes3D(
                        centers=[obj3d.center.tolist()],
                        sizes=[obj3d.size.tolist()],
                        labels=[f"{label} (track={obj3d.tracking_confidence:.2f})"],
                        colors=[color],
                    ),
                )

        # --- Reasoning trace text log ---
        rr.set_time("keyframe", sequence=0)
        rr.log("info/pipeline", rr.TextLog(
            f"World2Data: {len(self.frame_data)} frames, "
            f"{sum(fd.pts3d.shape[0] for fd in self.frame_data)} points, "
            f"{len(self.objects_3d)} objects"
        ))
        for obj3d in self.objects_3d:
            rr.log("info/objects", rr.TextLog(
                f"{obj3d.entity} ({obj3d.obj_type}): "
                f"{obj3d.component_type}, "
                f"{obj3d.initial_state} -> {obj3d.final_state}"
            ))
            for sc in obj3d.state_changes:
                rr.log("info/state_changes", rr.TextLog(
                    f"[{sc.get('time', '?')}] {obj3d.entity}: "
                    f"{sc.get('from', '?')} -> {sc.get('to', '?')} "
                    f"({sc.get('cause', 'unknown cause')})"
                ))

        rr.save(rrd_path)
        print(f"PASS: Saved {rrd_path} ({len(self.frame_data)} frames, "
              f"{len(self.objects_3d)} 3D objects)")
        print(f"  View:  uv run rerun --port 0 {rrd_path}")

    # =====================================================================
    # PLY EXPORT (colored point cloud for any 3D viewer)
    # =====================================================================
    def export_ply(self, ply_path):
        """Export the aggregated colored point cloud to PLY format."""
        if self.point_cloud is None or self.point_cloud.shape[0] == 0:
            print("SKIP: No point cloud to export.")
            return

        pts = self.point_cloud
        cols = self.point_colors
        n = pts.shape[0]

        if cols is not None and cols.shape[0] == n:
            cols_u8 = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
        else:
            cols_u8 = np.full((n, 3), 180, dtype=np.uint8)

        with open(ply_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(n):
                f.write(f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} "
                        f"{cols_u8[i,0]} {cols_u8[i,1]} {cols_u8[i,2]}\n")

        print(f"PASS: Exported {ply_path} ({n:,} colored points)")


# =========================================================================
# CLI
# =========================================================================
def _cli_main():
    """Entry point for ``world2data-demo`` console script."""
    import argparse
    parser = argparse.ArgumentParser(description="World2Data Pipeline")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", default="output.usda",
                        help="Output USD file path")
    parser.add_argument("--strategy", default="swin-3",
                        help="MASt3R pair strategy")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Keyframe diff threshold")
    parser.add_argument("--max-keyframes", type=int, default=20,
                        help="Max keyframes to reconstruct (0 = unlimited)")
    parser.add_argument("--target-fps", type=float, default=30.0,
                        help="Sampling FPS for keyframe extraction (0 = source FPS)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic sampling/tracking")
    parser.add_argument("--no-rerun", action="store_true",
                        help="Disable Rerun visualization")
    parser.add_argument("--legacy", action="store_true",
                        help="Force legacy keyframe-based semantic flow")
    args = parser.parse_args()

    pipeline = World2DataPipeline(
        args.video, output_path=args.output,
        rerun_enabled=not args.no_rerun,
        seed=args.seed,
    )
    success = pipeline.run_ralph_loop(
        threshold=args.threshold,
        strategy=args.strategy,
        max_keyframes=args.max_keyframes,
        target_fps=args.target_fps,
        use_multimodel=not args.legacy,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    _cli_main()
