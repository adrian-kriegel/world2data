# World2Data

**Video -> 4D Dynamic Scene Graph -> OpenUSD**

A deep tech pipeline that converts 2D video into a **4D Dynamic Scene Graph**
(space + time) with multi-model object detection, pixel-perfect segmentation,
physics joint metadata, confidence scoring, human-in-the-loop review flags,
and causal reasoning -- exported as industry-standard OpenUSD.

## What Actually Works

| Feature | Status | Details |
|---------|--------|---------|
| Keyframe extraction | Working | L1 pixel diff with adaptive threshold + retry |
| 3D reconstruction | Working | MASt3R (SOTA multi-view stereo), per-frame temporal point clouds |
| YOLOv8 detection | **NEW** | Real-time object detection + instance segmentation (80 classes) |
| SAM3 segmentation | **NEW** | Pixel-perfect masks + video tracking via text prompts |
| Gemini video analysis | **NEW** | Full video upload for holistic scene understanding |
| 4D scene fusion | **NEW** | Cross-model validation, confidence scoring, human review flags |
| Reasoning engine | **NEW** | Gemini/GPT-5.2 cross-validates all model outputs |
| Ground truth evaluator | **NEW** | Quality metrics + human feedback learning |
| Legacy Gemini detection | Working | Fallback: Gemini Vision detects objects, back-projects to 3D |
| Causal reasoning | Working | Gemini analyzes state changes, assigns physics joint types |
| OpenUSD export | Working | Point cloud + cameras + objects + physics + confidence + human flags |
| Rerun recording | Working | Interactive temporal 3D with animated object boxes |
| Colored PLY export | Working | For MeshLab / CloudCompare / Blender viewing |
| Self-correction loop | Working | Each step verifies + retries with different params on failure |
| Annotated video export | **NEW** | YOLO boxes + SAM3 masks overlaid on video frames |

### Proven on Real Video

Last verified run on a 33-second 720p video:
- **737,486** 3D points reconstructed across **20** temporal frames
- **6** objects detected by YOLOv8 (chair, person, dining table) with instance masks
- Objects positioned in 3D with bounding boxes and confidence scores
- Physics joint metadata (FixedJoint, RevoluteJoint) assigned via Gemini
- Full pipeline completes in ~2.5 minutes (GPU + API calls)

## Quick Start

```bash
# 1. Install dependencies (uv required)
uv sync

# 2. Set your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Run on a video (multi-model v2 pipeline)
uv run python pipeline_controller.py your_video.mp4 --output scene.usda

# 4. View the interactive 3D recording
uv run rerun scene.rrd
```

## Pipeline Architecture (The "Ralph Loop" v2)

```
Video (.mp4, up to 100MB / ~11 min)
    |
    v
[Step 1: Keyframe Extraction]
    |
    v
[Step 2: MASt3R 3D Reconstruction]
    |  per-frame point clouds + depth + camera poses
    |
    +---> [Step 3a: YOLOv8 Detection]    -- boxes + classes + masks (~5ms/frame)
    |         |
    +---> [Step 3b: SAM3 Segmentation]   -- pixel-perfect masks + tracking IDs
    |         |  (text prompts from YOLO)
    +---> [Step 3c: Gemini Video]         -- full video upload, narrative + events
              |
              v
         [Step 4: 4D Scene Fusion + Reasoning]
              |  cross-model validation + confidence + human flags
              v
         [Step 5: OpenUSD Export]    [Step 6: Rerun .rrd]    [Step 7: PLY]
              |                           |                       |
         scene.usda                  scene.rrd               scene.ply
         + physics joints            + temporal 3D            + colored points
         + confidence scores         + YOLO overlay
         + human review flags        + SAM3 masks
         + reasoning trace           + reasoning trace
```

Each step self-verifies and retries with different parameters on failure.
Models degrade gracefully: if SAM3 is unavailable, YOLO + Gemini still work.
If no models are available, falls back to legacy Gemini-only pipeline.

## Five Model Interfaces

| # | Interface | Model | Input | Output |
|---|-----------|-------|-------|--------|
| 1 | Depth + Geometry | **MASt3R** | Keyframe images | 3D points, depth, poses, intrinsics |
| 2 | Object Detection | **YOLOv8x-seg** | Video frames | Boxes, classes, scores, instance masks |
| 3 | Segmentation | **SAM3** | Frames + text prompts | Pixel-perfect masks, tracking IDs |
| 4 | Scene Description | **Gemini Video** | Full video file | Objects, events, spatial relations, narrative |
| 5 | Reasoning | **Gemini/GPT-5.2** | All model outputs | Verified objects, confidence, human flags |

## Output Files

| File | Description |
|------|-------------|
| `*.usda` | OpenUSD scene: colored point cloud, cameras, physics objects, confidence |
| `*_scene_graph.json` | Full 4D scene graph: objects, reasoning, YOLO summary, evaluation |
| `*.rrd` | Interactive Rerun recording (temporal 3D + all model outputs) |
| `*.ply` | Colored point cloud (viewable in any 3D tool) |

### USD Scene Hierarchy (v2)

```
/World
  /PointCloud                    -- colored point cloud (displayColor)
  /Cameras/Cam_00..N             -- camera transforms + focal length
  /Objects/Chair_01              -- positioned at real 3D coordinates
    world2data:type              = "chair"
    world2data:component_type    = "FixedJoint"
    world2data:confidence        = 0.92           <-- cross-model agreement
    world2data:detected_by       = "yolo,sam3,gemini"
    world2data:human_review      = false
    world2data:initial_state     = "stationary"
    world2data:final_state       = "stationary"
    world2data:reasoning_trace   = "Detected by YOLO (0.93), confirmed by SAM3 mask..."
    UsdPhysicsRigidBodyAPI       (for movable objects)
    UsdPhysicsCollisionAPI
```

## Viewing the 3D Recording

### Rerun Viewer (Interactive Timeline)

```bash
uv run rerun scene.rrd
```

What you see:
- **world/current_view** -- point cloud from current camera (changes per frame)
- **world/accumulated** -- all points seen up to this moment (grows over time)
- **world/camera** -- camera frustum showing viewpoint
- **world/camera/image** -- 2D keyframe at that viewpoint
- **world/camera/depth** -- depth map from MASt3R
- **world/trajectory** -- yellow camera path line
- **world/objects/\*** -- labeled 3D bounding boxes (color = type, green = state changed)
- **input/annotated** -- video frame with YOLO boxes + SAM3 mask overlay
- **info/\*** -- reasoning trace text logs

Keyboard: `Space` play/pause, `Left/Right` step, `Shift+arrows` jump, scroll to zoom.

### Point Cloud Viewers

The `.ply` file can be opened in:
- **MeshLab** (free, cross-platform)
- **CloudCompare** (free, point cloud specialist)
- **Blender** (import as mesh)

### USD Viewers

The `.usda` file can be opened in:
- **NVIDIA Omniverse** (full physics simulation)
- **usdview** (comes with USD toolkit)
- **Blender** (with USD add-on)

## Running Tests

```bash
# Fast tests (mock geometry, no GPU, ~15s)
uv run python -m pytest test_pipeline.py -v

# Excluding API-dependent tests
uv run python -m pytest test_pipeline.py -v -k "not gemini"

# OVERNIGHT: Full pipeline on real video with MASt3R + Gemini (~3-5 min)
uv run python -m pytest test_pipeline.py -v --overnight

# Overnight test only
uv run python -m pytest test_pipeline.py -v --overnight -k overnight
```

The overnight test saves outputs to `overnight_output/` for manual inspection.

## Overnight Test Results

To view the overnight test output:

```bash
# View interactive 3D recording
uv run rerun overnight_output/overnight.rrd

# View pipeline summary
uv run python generate_demo.py --json overnight_output/overnight_scene_graph.json

# View point cloud externally
# Open overnight_output/overnight.ply in MeshLab or CloudCompare

# View USD in Omniverse
# Open overnight_output/overnight.usda
```

## Demo Generator

```bash
# Show summary of pipeline outputs
uv run python generate_demo.py --json real_scene_scene_graph.json

# Run pipeline and generate demo from video
uv run python generate_demo.py --video your_video.mp4 --output demo

# Open in Rerun viewer
uv run python generate_demo.py --json real_scene_scene_graph.json --open
```

## Core Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Geometry | **MASt3R** (Naver) | SOTA multi-view stereo, metric 3D reconstruction |
| Detection | **YOLOv8x-seg** (Ultralytics) | Real-time object detection + instance segmentation |
| Segmentation | **SAM3** (Meta) | Pixel-perfect masks, video tracking, text prompts |
| Scene Analysis | **Gemini 2.5** (Google) | Full video analysis, narrative, events, relations |
| Reasoning | **Gemini/GPT-5.2** | Cross-model validation, confidence, human flags |
| 3D Format | **OpenUSD** (Pixar) | Industry-standard scene format with physics APIs |
| Visualization | **Rerun.io** | Interactive temporal 3D viewer |
| Fusion | Custom **SceneFusion4D** | 4D object tracking, confidence, human-in-the-loop |
| Evaluation | Custom **GroundTruthEvaluator** | Quality metrics, human feedback learning |

## API Key Strategy

| Provider | Purpose | Tier Needed |
|----------|---------|-------------|
| Google Gemini | Video analysis + reasoning | Free (15 RPM) or Paid |
| OpenAI | Alternative reasoning (GPT-5.2) | Optional, paid |
| HuggingFace | SAM3 model download | Free (accept license) |

**Recommended:** Upgrade Gemini to paid tier for video analysis (100MB/request).
Cost: ~$0.075/min of video. Total per run: ~$1-5.

## Presentation Plan

### Recording a Demo Video

1. Run the pipeline on your video:
   ```bash
   uv run python pipeline_controller.py your_video.mp4 --output demo.usda
   ```

2. Open the recording in Rerun:
   ```bash
   uv run rerun demo.rrd
   ```

3. In the Rerun viewer:
   - Navigate to a good 3D viewpoint
   - Press `Space` to play through the timeline
   - Point cloud evolves, camera moves, objects appear with labels
   - Use OBS Studio or Windows Game Bar (`Win+G`) to screen-record

4. For the pitch:
   - Show original video side-by-side with Rerun 3D view
   - Highlight temporal evolution (point cloud growing)
   - Zoom into detected objects and their labeled bounding boxes
   - Show confidence scores and human review flags
   - Demonstrate multi-model agreement (YOLO + SAM3 + Gemini)
   - Show the USD file structure with physics metadata

### Key Talking Points for Judges

1. **Input:** A single 2D video (phone camera, no special hardware)
2. **Output:** Industry-standard OpenUSD with physics-ready 3D objects
3. **Multi-model:** YOLOv8 + SAM3 + Gemini + MASt3R cross-validate each other
4. **4D Tracking:** Objects tracked through space AND time
5. **Confidence scoring:** Cross-model agreement quantified per object
6. **Human-in-the-loop:** Uncertain objects flagged for review
7. **Self-correcting:** Pipeline retries each step automatically
8. **Temporal:** Point clouds evolve through time (not just static)
9. **Semantic:** Objects detected, segmented, labeled, and positioned in 3D
10. **Physics-aware:** Doors get RevoluteJoint, drawers get PrismaticJoint
11. **Causal:** State changes tracked (door opened, cup moved)
12. **Product foundation:** GroundTruthEvaluator learns from human feedback

## Project Structure

```
world2data/
  pipeline_controller.py   -- Main pipeline (The "Ralph Loop" v2)
  model_interfaces.py       -- Clean interfaces: YOLO, SAM3, Gemini, Reasoning
  scene_fusion.py           -- 4D fusion, confidence, human-loop, evaluator
  test_pipeline.py          -- Test suite (29 fast + 1 overnight)
  conftest.py              -- Pytest config for --overnight marker
  generate_test_video.py   -- Synthetic test video generator
  generate_demo.py         -- Demo summary + viewer launcher
  visualize_impact.py      -- Pitch visualization (demo + live modes)
  pyproject.toml           -- Dependencies (uv)
  IMPLEMENTATION_PLAN.md   -- Full architecture plan
  AGENT_PROMPT.md          -- Implementation instructions for agents
  World2Data.txt           -- Original project vision
  .env                     -- API keys (not committed)
  .gitignore               -- Excludes secrets, outputs, mast3r/
  mast3r/                  -- MASt3R clone (not committed)
```

## Resources

- https://github.com/naver/mast3r
- https://huggingface.co/facebook/sam3
- https://huggingface.co/Ultralytics/YOLOv8
- https://rerun.io/
- https://openusd.org/
- https://ai.google.dev/ (Gemini API)
