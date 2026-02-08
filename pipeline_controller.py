"""World2Data Pipeline Controller -- The "Ralph Loop"

Converts 2D video into a 3D Dynamic Scene Graph (OpenUSD) using:
  Step 1: Smart keyframe extraction (L1 diff, upgradeable to LPIPS)
  Step 2: Metric 3D reconstruction via MASt3R  (per-frame temporal data)
  Step 3: Semantic 3D object detection via Gemini vision (replaces SAM 3 stub)
  Step 4: Causal reasoning + state change analysis via Gemini
  Step 5: OpenUSD export with physics joint metadata
  Step 6: Save interactive Rerun .rrd recording
"""
import os
import sys
import json
import tempfile
import cv2
import torch
import numpy as np
import rerun as rr
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, Vt

# ---------------------------------------------------------------------------
# Load .env file (GOOGLE_API_KEY, etc.)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# MASt3R: add the local clone to sys.path so imports resolve
# ---------------------------------------------------------------------------
MAST3R_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mast3r")
if MAST3R_ROOT not in sys.path:
    sys.path.insert(0, MAST3R_ROOT)

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
# Config
# ---------------------------------------------------------------------------
MAST3R_CHECKPOINT = os.path.join(
    MAST3R_ROOT, "checkpoints",
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
                 first_seen_frame=0, last_seen_frame=0):
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
        }


# =========================================================================
# Pipeline
# =========================================================================
class World2DataPipeline:
    """Self-correcting pipeline: video -> 3D scene graph -> USD."""

    def __init__(self, video_path, output_path="output.usda",
                 keyframe_dir=None, cache_dir=None, rerun_enabled=True):
        self.video_path = video_path
        self.output_path = output_path
        self.keyframe_dir = keyframe_dir or tempfile.mkdtemp(prefix="w2d_keyframes_")
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="w2d_cache_")

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

        # Rerun
        self.rerun_enabled = rerun_enabled
        if rerun_enabled:
            rr.init("world2data_debug", spawn=False)

    # =====================================================================
    # RALPH LOOP
    # =====================================================================
    def run_ralph_loop(self):
        """Run the full pipeline with self-correction."""
        print(">>> STARTING RALPH LOOP...")

        # STEP 1: KEYFRAME EXTRACTION
        if not self.step_1_smart_extraction():
            print("!!! Step 1 Failed. Retrying with lower threshold...")
            if not self.step_1_smart_extraction(threshold=5.0):
                print("!!! Step 1 still failing. Check your video input.")
                return False

        # STEP 2: METRIC GEOMETRY (MASt3R)
        if not self.step_2_geometric_reconstruction():
            print("!!! Step 2 Failed. Retrying with exhaustive matching...")
            if not self.step_2_geometric_reconstruction(strategy="complete"):
                print("!!! Step 2 still failing. Video may lack parallax.")
                return False

        # STEP 3: SEMANTIC 3D OBJECT DETECTION (Gemini Vision)
        self.step_3_semantic_detection()

        # STEP 4: CAUSAL REASONING (Gemini)
        self.step_4_causal_reasoning()

        # STEP 5: OPENUSD EXPORT (with physics)
        self.step_5_export_usd()

        # STEP 6: TEMPORAL RERUN RECORDING
        rrd_path = self.output_path.replace(".usda", ".rrd")
        self.save_temporal_rrd(rrd_path)

        print(">>> PIPELINE COMPLETE. ARTIFACT GENERATED.")
        return True

    # =====================================================================
    # STEP 1: KEYFRAME EXTRACTION
    # =====================================================================
    def step_1_smart_extraction(self, threshold=15.0, max_keyframes=20):
        """Extract keyframes via L1 pixel diff. Records video timestamps."""
        print(f"--- Step 1: Extracting Keyframes (Threshold: {threshold}) ---")

        if not os.path.isfile(self.video_path):
            print(f"FAIL: Video not found at '{self.video_path}'.")
            return False

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"FAIL: Could not open video '{self.video_path}'.")
            return False

        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        last_frame = None
        self.keyframes = []
        self.keyframe_paths = []
        self.keyframe_timestamps = []
        os.makedirs(self.keyframe_dir, exist_ok=True)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if last_frame is None:
                self.keyframes.append(frame)
                self.keyframe_timestamps.append(frame_idx)
                last_frame = frame
                frame_idx += 1
                continue
            diff = np.mean(np.abs(frame.astype(float) - last_frame.astype(float)))
            if diff > threshold:
                self.keyframes.append(frame)
                self.keyframe_timestamps.append(frame_idx)
                last_frame = frame
                if len(self.keyframes) >= max_keyframes:
                    break
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
        print(f"PASS: Extracted {n} keyframes [{ts_sec[0]:.1f}s .. {ts_sec[-1]:.1f}s]")
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
    # STEP 3: SEMANTIC 3D OBJECT DETECTION (Gemini Vision)
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

        # Send first, middle, and last keyframes for detection
        indices = [0, len(self.keyframe_paths) // 2, len(self.keyframe_paths) - 1]
        indices = sorted(set(i for i in indices if i < len(self.keyframe_paths)))

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

            # Back-project each detection to 3D
            self.objects_3d = []
            for det in detections:
                obj3d = self._backproject_bbox_to_3d(det, indices)
                if obj3d is not None:
                    self.objects_3d.append(obj3d)
                    print(f"    - {obj3d.entity} ({obj3d.obj_type}): "
                          f"3D center={np.round(obj3d.center, 2).tolist()}, "
                          f"size={np.round(obj3d.size, 2).tolist()}")

            print(f"  PASS: {len(self.objects_3d)} objects positioned in 3D")

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
            reasoning_map = {
                obj.get("entity", ""): obj
                for obj in objects_list if isinstance(obj, dict)
            }
            for obj3d in self.objects_3d:
                if obj3d.entity in reasoning_map:
                    r = reasoning_map[obj3d.entity]
                    obj3d.component_type = r.get("component_type", obj3d.component_type)
                    obj3d.initial_state = r.get("initial_state", obj3d.initial_state)
                    obj3d.final_state = r.get("final_state", obj3d.final_state)
                    obj3d.state_changes = r.get("state_changes", [])

            n = len(self.scene_graph.get("objects", []))
            print(f"  PASS: Reasoning complete for {n} objects.")
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

        # --- Point Cloud ---
        if self.point_cloud is not None and self.point_cloud.shape[0] > 0:
            pc_prim = UsdGeom.Points.Define(stage, "/World/PointCloud")
            pts = self.point_cloud
            if pts.shape[0] > 50000:
                idx = np.random.choice(pts.shape[0], 50000, replace=False)
                pts = pts[idx]
            pc_prim.GetPointsAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts.tolist()])
            )
            pc_prim.GetWidthsAttr().Set(Vt.FloatArray([0.005] * len(pts)))

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
        for obj3d in self.objects_3d:
            entity_safe = obj3d.entity.replace(" ", "_").replace("/", "_")
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

            # USD Physics API
            if obj3d.component_type != "FixedJoint":
                UsdPhysics.RigidBodyAPI.Apply(prim)
                UsdPhysics.CollisionAPI.Apply(prim)

        stage.GetRootLayer().Save()
        print(f"PASS: Saved {self.output_path} "
              f"({len(self.objects_3d)} positioned objects with physics)")

        # JSON sidecar
        json_path = self.output_path.replace(".usda", "_scene_graph.json")
        with open(json_path, "w") as f:
            json.dump({
                "scene_graph": self.scene_graph,
                "objects_3d": [o.to_dict() for o in self.objects_3d],
                "num_cameras": len(self.poses),
                "num_points": int(self.point_cloud.shape[0])
                    if self.point_cloud is not None else 0,
                "num_frames": len(self.frame_data),
                "focals": self.focals,
                "video_fps": self.video_fps,
                "keyframe_timestamps": self.keyframe_timestamps,
            }, f, indent=2)
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
                idx = np.random.choice(acc_pts.shape[0], 200000, replace=False)
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
                    f"world/objects/{obj3d.entity}",
                    rr.Boxes3D(
                        centers=[obj3d.center.tolist()],
                        sizes=[obj3d.size.tolist()],
                        labels=[label],
                        colors=[color],
                    ),
                )

        rr.save(rrd_path)
        print(f"PASS: Saved {rrd_path} ({len(self.frame_data)} frames, "
              f"{len(self.objects_3d)} 3D objects)")
        print(f"  View:  uv run rerun {rrd_path}")


# =========================================================================
# CLI
# =========================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="World2Data Pipeline")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", default="output.usda",
                        help="Output USD file path")
    parser.add_argument("--strategy", default="swin-3",
                        help="MASt3R pair strategy")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Keyframe diff threshold")
    parser.add_argument("--no-rerun", action="store_true",
                        help="Disable Rerun visualization")
    args = parser.parse_args()

    pipeline = World2DataPipeline(
        args.video, output_path=args.output,
        rerun_enabled=not args.no_rerun,
    )
    success = pipeline.run_ralph_loop()
    sys.exit(0 if success else 1)
