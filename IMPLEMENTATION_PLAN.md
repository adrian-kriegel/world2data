# World2Data v2 -- Implementation Plan

## 1. Overnight Test: How to Run, View, and Present

### Running
```bash
# Run the overnight test (full pipeline on real video, ~3-5 min)
uv run python -m pytest test_pipeline.py -v --overnight -k overnight

# Outputs saved to overnight_output/
#   overnight.usda          -- OpenUSD scene
#   overnight_scene_graph.json -- Full scene graph
#   overnight.rrd           -- Interactive Rerun recording
#   overnight.ply           -- Colored point cloud
#   overnight/keyframes/    -- Extracted keyframe images
```

### Viewing
```bash
# Interactive 3D timeline in Rerun
uv run rerun overnight_output/overnight.rrd

# Summary report
uv run python generate_demo.py --json overnight_output/overnight_scene_graph.json

# Point cloud in external viewer
# Open overnight_output/overnight.ply in MeshLab / CloudCompare

# USD in Omniverse
# Open overnight_output/overnight.usda in NVIDIA Omniverse
```

### Making it Presentable
1. Open the `.rrd` in Rerun viewer
2. Navigate to a good 3D angle showing the full scene
3. Press Space to play the timeline (point cloud evolves, objects appear)
4. Use OBS Studio or Win+G to screen-record the Rerun viewer
5. Import into a pitch deck video editor

---

## 2. Architecture: 5 Model Interfaces + 4D Fusion

### The Five Interfaces

Each interface is a clean Python class that takes video input and returns
structured 4D data (space + time). All outputs are timestamped per-frame.

```
┌─────────────────────────────────────────────────────────┐
│                    VIDEO INPUT (.mp4)                     │
└────┬──────────┬──────────┬──────────┬──────────┬────────┘
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐
│  Depth  ││  Seg    ││ Detect  ││  Scene  ││Reasoning│
│ MASt3R  ││ SAM3    ││ YOLOv8  ││ Gemini  ││ Gemini/ │
│         ││         ││         ││ Video   ││ GPT-5.2 │
└────┬────┘└────┬────┘└────┬────┘└────┬────┘└────┬────┘
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────┐
│              4D SCENE FUSION LAYER                        │
│  Connects geometry + semantics + detection + reasoning   │
│  in unified spacetime coordinate system                  │
└────┬──────────────────────────────────────┬──────────────┘
     │                                      │
     ▼                                      ▼
┌───────────────┐              ┌────────────────────────┐
│  OpenUSD      │              │  Rerun .rrd            │
│  + physics    │              │  + video player layer  │
│  + confidence │              │  + all model outputs   │
│  + human-loop │              │  + reasoning trace     │
└───────────────┘              └────────────────────────┘
```

### Interface 1: Depth + Geometry (MASt3R) -- ALREADY DONE
- Input: Keyframe images
- Output per frame: `FrameData(pts3d, colors, depth_map, pose, focal, pp)`
- Status: **Fully implemented and tested**

### Interface 2: Segmentation (SAM3)
- Input: Video frames + text prompts (from YOLOv8 class names)
- Output per frame: `SegmentationResult(masks[], labels[], scores[], object_ids[])`
- SAM3 provides pixel-perfect masks + cross-frame object tracking
- Use `Sam3VideoModel` from HuggingFace transformers for video tracking
- Text prompts driven by YOLOv8 detections ("table", "chair", "door")

### Interface 3: Object Detection (YOLOv8)
- Input: Video frames
- Output per frame: `DetectionResult(boxes[], classes[], scores[], class_names[])`
- Fast (< 5ms/frame on RTX 4090), provides class vocabulary for SAM3
- Use `yolov8x-seg.pt` for detection + instance segmentation
- Runs on ALL frames (not just keyframes) for temporal continuity

### Interface 4: Scene Description (Gemini Video Analysis)
- Input: Full video file (up to 100MB / ~11 min)
- Output: `SceneDescription(objects[], spatial_relations[], events[], narrative)`
- Upload entire video to Gemini instead of keyframe images
- Get holistic scene understanding + temporal event detection
- Much richer than per-frame analysis

### Interface 5: Reasoning + Ground Truth (Gemini Pro / GPT-5.2)
- Input: All outputs from interfaces 1-4 + scene description
- Output: `ReasoningResult(verified_objects[], state_changes[], confidence_scores[], human_review_flags[])`
- Cross-validates detections across models
- Assigns confidence scores (how many models agree on this object?)
- Flags ambiguous items for human-in-the-loop review
- Builds symbolic reasoning chain for each object

---

## 3. Execution Plan

### Phase 1: Install Dependencies (15 min)
```bash
uv add ultralytics transformers accelerate
```
- `ultralytics` provides YOLOv8 (auto-downloads weights)
- `transformers` provides SAM3 (facebook/sam3 from HuggingFace)
- `accelerate` needed by SAM3 for device management

### Phase 2: Build Individual Interfaces (model_interfaces.py)
Each interface is a class with:
- `__init__(device="cuda")` -- load model once
- `process_frame(frame, frame_idx) -> FrameResult` -- per-frame
- `process_video(video_path) -> list[FrameResult]` -- full video
- `to_rerun(result, timestamp)` -- log to Rerun
- `to_dict(result)` -- JSON-serializable

### Phase 3: Build 4D Fusion Layer (scene_fusion.py)
- Takes ALL interface outputs
- Aligns them in shared coordinate system (using MASt3R poses)
- Projects 2D detections/masks to 3D (using MASt3R depth)
- Merges overlapping detections across models
- Computes confidence scores per object
- Produces unified `SceneObject4D` with full lifecycle

### Phase 4: Upgrade Pipeline Controller
- Replace Step 3 with multi-model detection pipeline
- Add Gemini video analysis (full video upload)
- Add reasoning/validation step with confidence + human flags
- Export enriched USD with confidence attributes

### Phase 5: Visualization Layer
- Rerun: all 5 interface outputs on shared timeline
- Video player mode: original video + overlay annotations
- Confidence heatmap: color objects by certainty
- Human-in-the-loop: flag uncertain objects in the viewer

---

## 4. Connection Plan: How the Interfaces Link Together

```
YOLOv8 (fast, per-frame)
  │ class names + boxes
  ├──────────────────────► SAM3 (text prompts from YOLO classes)
  │                          │ pixel-perfect masks
  │                          │
  ├──► MASt3R (depth + 3D)◄─┘ masks projected to 3D
  │       │ 3D point clouds
  │       │ camera poses
  │       │
  ├──► Gemini Video (holistic scene understanding)
  │       │ narrative + events + spatial relations
  │       │
  └──► Reasoning Model (Gemini Pro / GPT-5.2)
          │ INPUT: all of the above
          │ OUTPUT: verified scene graph with confidence
          │
          ▼
     UNIFIED 4D SCENE GRAPH
       - Each object has: 3D bbox, mask, class, confidence
       - Each object has: lifecycle (first_seen, last_seen, state_changes)
       - Each object has: human_review_flag if uncertain
```

### Data Flow Per Frame:
1. YOLOv8 detects objects (boxes + classes) -- ~5ms
2. SAM3 segments each detected class (pixel masks) -- ~30ms
3. MASt3R provides depth + 3D for keyframes -- already computed
4. Fusion: project YOLOv8 boxes + SAM3 masks to 3D using MASt3R depth
5. Gemini Video: provides scene-level context (run once for whole video)
6. Reasoning: validates everything, assigns confidence, flags for human review

### Similarity Matching Across Models:
- **Geometric**: 3D IoU between MASt3R-projected objects
- **Semantic**: Name matching (YOLO "dining table" = Gemini "table")
- **Visual**: SAM3 mask overlap with YOLO bbox

---

## 5. API Key Strategy

### Current: Gemini Free Tier (Strict Limits)
- 15 RPM, 1M tokens/day
- Video analysis: ~1 min video per request
- Sufficient for: development and testing on short videos

### Recommended: Gemini Paid Tier
- For production/demo: upgrade to pay-as-you-go
- Allows: longer videos, more requests, faster responses
- Cost: ~$0.075/min for video input

### Alternative: GPT-5.2 via OpenAI API
- Use for the reasoning/validation step (Step 5)
- Better at structured JSON output and logical reasoning
- Use Gemini for video analysis (native video support)
- Use GPT-5.2 for the final reasoning synthesis

### Recommendation:
- **YES, make a paid Gemini key** for video analysis (Step 4)
- Keep GPT-5.2 for reasoning step (Step 5) as backup/complement
- Total cost estimate: ~$1-5 per full pipeline run

---

## 6. Visualization Output Plan

### A. Comprehensive OpenUSD File
```
/World
  /PointCloud                    -- colored, per-frame animated
  /DepthMaps/Frame_00..N         -- per-keyframe depth images
  /Cameras/Cam_00..N             -- camera transforms + intrinsics
  /Objects/Table_01              -- positioned at 3D center
    world2data:type              = "table"
    world2data:component_type    = "FixedJoint"
    world2data:confidence        = 0.92        ← NEW
    world2data:detected_by       = "yolo,sam3,gemini"  ← NEW
    world2data:human_review      = false       ← NEW
    world2data:initial_state     = "stationary"
    world2data:final_state       = "stationary"
    world2data:reasoning_trace   = "Detected by YOLO (0.95), confirmed by SAM3 mask, Gemini identifies as dining table"  ← NEW
    UsdPhysicsRigidBodyAPI
    UsdPhysicsCollisionAPI
  /Uncertain/Unknown_03          ← NEW: objects needing human review
    world2data:confidence        = 0.35
    world2data:human_review      = true
    world2data:review_reason     = "YOLO detected, SAM3 mask poor, Gemini unsure"
```

### B. Symbolic Reasoning / Evaluator Layer
This is the "cost function" that improves over time:

```python
class GroundTruthEvaluator:
    """Evaluates and improves scene graph quality over time."""

    def evaluate(self, scene_graph, video_path) -> EvaluationResult:
        """Score the scene graph quality."""
        # Metrics:
        # - cross_model_agreement: do YOLO, SAM3, Gemini agree?
        # - temporal_consistency: does object tracking hold over time?
        # - physics_plausibility: do state changes make physical sense?
        # - coverage: what % of video frames have detections?

    def flag_for_review(self, scene_graph) -> list[ReviewItem]:
        """Identify items needing human correction."""
        # Flag when: confidence < 0.5, models disagree, physics violated

    def incorporate_feedback(self, feedback: HumanFeedback):
        """Learn from human corrections to improve future runs."""
        # Store corrections as training signal
        # Adjust confidence thresholds
        # Build "known objects" vocabulary for this scene type
```

### C. Video Player Visualization Layer
For investor presentation, we need a synchronized video player:

```
┌──────────────────────────────────────────────────────┐
│  ┌─────────────────┐  ┌─────────────────────────────┐│
│  │  Original Video  │  │  3D Rerun View              ││
│  │  with overlays:  │  │  - Point cloud evolving     ││
│  │  - YOLO boxes    │  │  - 3D bounding boxes        ││
│  │  - SAM3 masks    │  │  - Camera trajectory        ││
│  │  - Labels        │  │  - Object labels            ││
│  └─────────────────┘  └─────────────────────────────┘│
│  ┌───────────────────────────────────────────────────┐│
│  │  Timeline scrubber  ▶ ││══════════●═══════════││  ││
│  │  Frame 142/1984     0:04.7 / 0:33.0               ││
│  └───────────────────────────────────────────────────┘│
│  ┌───────────────────────────────────────────────────┐│
│  │  SCENE GRAPH: Table_01 (stationary, conf=0.92)    ││
│  │               Chair_03 (moved, conf=0.78)         ││
│  │               ⚠ Unknown_03 (needs review, 0.35)  ││
│  └───────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

Implementation: Use Rerun's multi-panel layout:
- Panel 1: `input/video` -- log original frames as `rr.Image`
- Panel 2: `world/*` -- 3D view with point clouds + boxes
- Panel 3: `info/*` -- text logs with reasoning trace
- Timeline: synchronized via `rr.set_time`

For the annotated video overlay, render YOLOv8 boxes + SAM3 masks
directly onto each frame before logging to Rerun.

---

## 7. File Structure After Implementation

```
world2data/
  pipeline_controller.py      -- Main pipeline (updated)
  model_interfaces.py          -- Clean interfaces for all 5 models
  scene_fusion.py             -- 4D fusion + confidence + human-loop
  ground_truth_evaluator.py   -- Symbolic reasoning + evaluation
  video_annotator.py          -- Render overlays on video frames
  generate_demo.py            -- Demo generator (updated)
  visualize_impact.py         -- Pitch visualization (updated)
  test_pipeline.py            -- Tests (updated)
  test_interfaces.py          -- Tests for individual model interfaces
  conftest.py                 -- Pytest config
  generate_test_video.py      -- Synthetic test video
  pyproject.toml              -- Dependencies (updated)
  .env                        -- API keys
  IMPLEMENTATION_PLAN.md      -- This file
  AGENT_PROMPT.md             -- Instructions for implementation agent
  World2Data.txt              -- Original project plan
  README.md                   -- Updated docs
```
