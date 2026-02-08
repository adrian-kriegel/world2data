# World2Data

# world2data

## Current Implementation Plan

For the presentation-ready, code-accurate plan of what is implemented now, see:

- [WORLD2DATA_REVISED_PLAN.md](WORLD2DATA_REVISED_PLAN.md)

Generate ground truth data from videos. 

Creates a 3D map of containing 

- coordinates with item labels
- action hints like "graspable" for "doorhandle"
- state changes like "door opened" "cup vanished" 

Reflects on improbable state changes that cannot be explained by an items motion model:

"Door opened" -> OK
"Cup vanished" -> a reasoning model is asked to understand why -> "the person who just drank from the cup was seen leaving the room" 
"

## Interfaces 

- Camera motion from video
- Object detection (bounding boxes or even segmention) 
- Depth estimation from video 
- Cost function for particles 
    - This can include the output from the depth estimation, camera from motion, object detection 
    - Should be agnostic to what exactly the interfaces give. e.g. object detection gives bounding boxes or segmentation - works with both 
- Motion model for each object: Maps current particle position in state space to new position + spread. The motion model may change based on the context: 
    - An object in someones hand is expected to move 
    - Objects on the table are not expected to move (they move with the table top)

- State change feedback
    - If a motion happens that is not explained adequatly by the model (things vanish, things that are not supposed to move, move), this information is fed into a reasoning model which can explain why 
    - The output from the reasoning model is used to adjust the motion model

All tied together with a particle filter:

Each *new* object detection spawns a new particle. Motion model and particle evolution is applied, then particles are sampled using cost function. 

## Idea


LFM2.5‑VL generates scene description (structured)
    - what objects this scene contains 
    - relationships "person drinks from mug" 

SLAM first to get camera motion from images and set a coordinate frame
Use Yolo + 3D estimation to find objects 
map them to 3D world (SLAM) coordinates 
find the objects mentioned by LFM before 

track each item using a particle filter 
LFM suggests a motion model for each particle 

Correlate the things found by Yolo with the object descriptions. 

The output should be
    - list of items that exist in the entire video and where in 3D space (initial condition)
    - list of actions that happen e.g.
        - "door opens at time t" (e.g. from LFM)
        - "cup vanishes at time t" (should be there according to PF but isn't found)
        - "person leaves room at time t" (e.g. from LFM)

We now have a graph and can look up items which are associated with unexplainable things, like the cup vanishing. The cup node is linked to the person node, we can collect all the cups actions and persons actions from the graph and get 


"cup seen first at x,y,z" 
"person drinks from cup at time t" 
"person leaves the room"
"cup expected at x,y,z but not found" 


## Resources 

https://arxiv.org/html/2602.04517v1
https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B?utm_source=chatgpt.com
https://rerun.io/

## Interface

All compontents read/write openusd. In the initial pipeline stage, the usd contains only reference to the video. 
Second stage produces stamped point clouds an camera poses and places them in the usd. 

Yolo places stamped 2d detections in the usd. 

A semantic model places relationships between entities. E.g. man drinks from cup at time t 

Particle filter places only the centroids and mean bounding box of all partices (the mean of the bounding_box state, not the bounding box of the cloud of particles) in the usd. 


## Particle filter 

A particle filter is used to fuse output from 2d object detection, 3d point cloud generation, camera motion, and semantic scene understanding. Each detected item in the scene is tracked by its own set of particles (effectively its own filter). This is important for the selection process as when selecting the top N particles, this is done on a per-object basis. You cannot compare the weight of "cup 01" and "chair 02", weights are only meaningful in relation to each other. You can compare "particle 170 of cup 01" to "particle 9 of cup 01". 

Each particle describes a hypothesis for an objects state:

x,y,z,bouding_box,mass,velocity3d

For each yolo detection, we try to find the corresponding set of particles (track_id) or multiple if there are multiple matches (e.g. "cup 01" and "cup 02" as yolo outputs only classses).

We can check which "cup" from yolo is "cup 01" by overlaying the back-projected bounding boxes of all particles and seeing where the final score is highest. 

If there is no match, we create a new set of particles, e.g. "cup 03". It is initialized by naively finding the points in the point cloud which are underneath the detection rect. Take the mean of all these points and scatter particles. The bounding box parameter can be deduced by the size of the 2d rect when projected to the given distance with the depth being guessed to roughly match the width/height. 

Selection: When selecting particles, their score comes from the back-projection of the 3d bounding box which is compared to the 2d yolo box, e.g. union over difference. 

Label flicker handling: assignment is done in two phases:
- label-consistent matching first (same class + IoU threshold)
- cross-label fallback only when geometric agreement is very strong and most particles support the match

Each track stores a per-label count map (`label -> count`) that is updated from particle support, so a track can accumulate class history such as `{"mug": 29, "remote": 5}` without losing track identity.

Point-cloud logic in the particle filter uses a **PCL-first backend** (`pclpy`/`pcl`) for:
- cropping observed points by a particle 3D AABB
- stitching per-track accumulated clouds
- voxel downsampling and nearest-neighbor alignment

If PCL bindings are unavailable, the runtime currently falls back to a numpy backend unless `require_point_cloud_backend=true` is configured.

## Camera calibration (quick app)

For ChArUco videos, calibrate intrinsics + distortion and export JSON + USDA camera prim:

```bash
world2data-calibration \
  --video data/inputs/camera_calibration.mp4 \
  --board-squares-x 11 \
  --board-squares-y 8 \
  --square-mm 15 \
  --marker-mm 11 \
  --dictionary DICT_4X4_50 \
  --output-json data/outputs/camera_calibration.json \
  --output-usda data/outputs/camera_calibration.usda
```

The USDA layer is protocol-aligned:
- Camera prim path: `/World/W2D/Sensors/CalibrationCamera`
- No authored camera pose/extrinsics (intrinsics only)
- Custom attrs:
  - `w2d:intrinsicMatrix`
  - `w2d:distortionModel`
  - `w2d:distortionCoeffs`
  - `w2d:imageWidth`
  - `w2d:imageHeight`
  - `w2d:producedByRunId`
- Provenance record at `/World/W2D/Provenance/runs/<RUNID>`

## YOLO observations adapter

Run YOLO on a video and write a protocol-aligned observations layer with raw detections:

```bash
world2data-yolo-adapter \
  --video res/video_360p_30fps.mp4 \
  --model yolov8n.pt \
  --output-usd outputs/yolo_observations.usda \
  --frame-step 1
```

Output paths are authored under `/World/W2D/Observations/YOLO/**` and include provenance at `/World/W2D/Provenance/runs/<RUNID>`.

## Particle filter adapter

Run particle tracking from protocol layers (calibration + camera poses + YOLO + stamped point cloud):

```bash
world2data-particle-filter-adapter \
  --calibration-layer outputs/camera_calibration.usda \
  --camera-poses-layer outputs/camera_poses.usda \
  --yolo-layer outputs/yolo_observations.usda \
  --point-cloud-layer outputs/point_cloud_frames.usda \
  --output-usd outputs/particle_tracks.usda
```

Tracks are written under `/World/W2D/Tracks/ParticleFilter/**` and entities under `/World/W2D/Entities/Objects/**`.

## Open3D mesh adapter

Generate colored, high-fidelity meshes from stitched PF point clouds and write a protocol-aligned mesh layer:

```bash
world2data-mesh-adapter \
  --tracks-layer outputs/particle_tracks.usda \
  --point-cloud-layer outputs/point_cloud_frames.usda \
  --output-usd outputs/mesh_reconstruction.usda \
  --poisson-depth 11
```

Outputs:
- mesh index prims under `/World/W2D/Reconstruction/Meshes/**`
- stitched colored cloud index prims under `/World/W2D/Reconstruction/StitchedTrackPointClouds/**`
- external assets in `external/recon/stitched/*_points_colored.ply` and `external/recon/meshes/*_mesh_colored.ply`



---- AI SLOP INCOMING

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

# 3. Run the full pipeline on a video (10fps keyframes)
uv run world2data-demo --video data/inputs/video_2026-02-08_12-12-15.mp4 --output data/outputs/investor_demo --fps 10

# 4. View the interactive 3D temporal recording
uv run rerun data/outputs/investor_demo/investor_demo.rrd
```

## Layered OpenUSD Output (Protocol v0.2)

Every pipeline run now produces a **protocol-compliant layered scene bundle**:

```
data/outputs/investor_demo/scene/
  scene.usda                           # Assembly -- open this file
  layers/
    00_base.usda                       # Conventions + namespace skeleton
    10_inputs_run_<RUN>.usda           # Video/cloud refs
    20_recon_run_<RUN>.usda            # Cameras + per-frame cloud index
    25_yolo_run_<RUN>.usda             # YOLO per-frame detections
    30_tracks_run_<RUN>.usda           # Entities + time-sampled tracks
    40_events_run_<RUN>.usda           # Inferred events/relations
    90_overrides.usda                  # Human QA corrections (empty template)
    99_session.usda                    # Per-user local edits
  external/
    inputs/                            # Copied video file
    recon/
      frame_000000.ply                 # Per-frame point clouds
      frame_000001.ply
      ...
      point_lineage_<RUN>.parquet      # Point provenance table
```

Key properties:
- **No dense point arrays in protocol layers** -- heavy geometry is externalized as PLY files
- **Every point is traceable** to its source frame via the lineage parquet
- **Temporal point clouds** -- the Rerun viewer shows active-map (sliding window) and full-history views
- **Each entity has a stable `w2d:uid`** and provenance linking to the exact run + model version

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
| `mesh_reconstruction.usda` | OpenUSD layer indexing colored stitched PF clouds + colored Open3D meshes |
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
uv run python -m world2data.pipeline.generate_demo --json overnight_output/overnight_scene_graph.json

# View point cloud externally
# Open overnight_output/overnight.ply in MeshLab or CloudCompare

# View USD in Omniverse
# Open overnight_output/overnight.usda
```

## Demo Generator

```bash
# Show summary of pipeline outputs
uv run python -m world2data.pipeline.generate_demo --json real_scene_scene_graph.json

# Run pipeline and generate demo from video
uv run python -m world2data.pipeline.generate_demo --video data/inputs/your_video.mp4 --output demo

# Open in Rerun viewer
uv run python -m world2data.pipeline.generate_demo --json real_scene_scene_graph.json --open
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
   uv run world2data-demo --video data/inputs/your_video.mp4
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
  src/world2data/               -- Python package (src layout)
    __init__.py                 -- Core exports: particle filter, vision, calibration
    __main__.py                 -- CLI entry point (uv run world2data)
    model.py                    -- Data model (AABB2D, Detection2D, CameraPose, etc.)
    particle_filter.py          -- Multi-object particle filter (production)
    vision.py                   -- 3D projection / back-projection helpers
    calibration.py              -- ChArUco camera calibration
    openusd.py                  -- USD export for particle-filter estimates
    usd_layers.py               -- OpenUSD Layering Protocol (multi-layer composition)
    pipeline/
      __init__.py               -- Pipeline exports
      controller.py             -- World2DataPipeline (The "Ralph Loop" v2)
      model_interfaces.py       -- Clean interfaces: YOLO, SAM3, Gemini, Reasoning
      scene_fusion.py           -- 4D fusion, confidence, human-loop, evaluator
      demo_run.py               -- Full demo runner for investor presentations
      generate_demo.py          -- Presentation-ready demo generator
      human_review_ui.py        -- Human-in-the-loop review UI (Gradio)
      visualize_impact.py       -- Rerun pitch visualization
  data/
    inputs/                     -- Video files for pipeline input (gitignored .mp4s)
    outputs/                    -- Generated calibration + pipeline artifacts
  tests/
    generate_test_video.py      -- Synthetic test video generator
    test_calibration.py         -- Camera calibration tests
    test_openusd_compat.py      -- USD export tests
    test_particle_filter.py     -- Multi-object particle filter tests
    test_pipeline.py            -- Pipeline tests (29 fast + 1 overnight)
    test_demo_components.py     -- Demo component tests (YOLO, fusion, etc.)
    test_usd_layers.py          -- OpenUSD Layering Protocol tests (12 tests)
  scripts/
    check_sam3_hf_access.py     -- Verify Hugging Face SAM3 access
    prepare_hackathon_submission.py -- Build submission archive
  submission/                   -- Hackathon submission materials
  testenvironment/              -- LFM2.5-VL model experiments (standalone)
  mast3r/                       -- MASt3R clone (not committed)
  conftest.py                   -- Pytest config for --overnight marker
  pyproject.toml                -- Dependencies (uv + hatchling)
  W2D_OpenUSD_Layering_Protocol.md -- OpenUSD multi-layer spec
  .env                          -- API keys (not committed)
```

### OpenUSD Layered Output Structure

The pipeline now produces a **multi-layer composed USD scene** following the W2D protocol:

```
scene/
  scene.usda                      # Assembly entrypoint (open this file)
  layers/
    00_base.usda                  # Conventions + namespace skeleton
    10_inputs_run_<RUNID>.usda    # Input refs (video, point cloud)
    20_recon_run_<RUNID>.usda     # Cameras + reconstruction
    30_tracks_run_<RUNID>.usda    # Entity tracks + bounding boxes
    40_events_run_<RUNID>.usda    # Events/relations graph
    90_overrides.usda             # Human QA corrections (strong layer)
    99_session.usda               # Per-user local edits (gitignored)
  external/
    inputs/                       # Video files
    recon/                        # Point cloud caches (.ply)
```

**Key principles:** No physical merging (composition only), namespace ownership,
provenance on every prim, deterministic layer ordering.

## Resources

- https://github.com/naver/mast3r
- https://huggingface.co/facebook/sam3
- https://huggingface.co/Ultralytics/YOLOv8
- https://rerun.io/
- https://openusd.org/
- https://ai.google.dev/ (Gemini API)
  - `w2d:imageWidth`
  - `w2d:imageHeight`
  - `w2d:producedByRunId`
- Provenance record at `/World/W2D/Provenance/runs/<RUNID>`
