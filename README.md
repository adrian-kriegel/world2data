# World2Data

Generate ground truth data from videos.

Creates a 3D Dynamic Scene Graph from 2D video containing:
- **3D point clouds** that evolve through time (per-frame reconstruction)
- **Object detection** with labels (door, table, chair, cup, ...)
- **State changes** (door opened, cup vanished, ...)
- **Physics hints** (RevoluteJoint, FixedJoint, PrismaticJoint)
- **Causal reasoning** about unexplainable events

## Quick Start

```bash
# 1. Install dependencies (uv required)
uv sync

# 2. Set your API key in .env
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Run on a video
uv run python pipeline_controller.py your_video.mp4 --output scene.usda
```

## Pipeline Steps (The "Ralph Loop")

| Step | Model | What it does |
|------|-------|-------------|
| 1. Keyframe Extraction | L1 pixel diff | Selects informative frames from video |
| 2. Geometry | **MASt3R** | Metric 3D reconstruction with per-frame point clouds |
| 3. Semantics | SAM 3 (stub) | Object segmentation projected to 3D |
| 4. Reasoning | **Gemini** | Identifies objects, state changes, and causality |
| 5. Export | **OpenUSD** + Rerun | `.usda` scene + interactive `.rrd` recording |

Each step self-verifies and retries with different parameters on failure.

## Viewing the Interactive 3D Timeline

The pipeline generates a **Rerun `.rrd` recording** with temporal 3D data.
This lets you scrub through the video timeline and see the point cloud evolve.

### Open in Rerun Viewer

```bash
# Install the Rerun viewer (first time only)
uv run pip install rerun-sdk

# Open the recording
uv run rerun scene.rrd
```

### What you see in the viewer

- **world/current_view** -- Point cloud from the current camera (changes per frame)
- **world/accumulated** -- All points seen up to this moment (grows over time)
- **world/camera** -- Camera frustum showing where the camera is looking
- **world/camera/image** -- The 2D keyframe image from that viewpoint
- **world/trajectory** -- Yellow line showing the camera path through space

Use the **timeline scrubber** at the bottom of the Rerun viewer to play through
the reconstruction. You can drag it, play/pause, or step frame-by-frame.

### Keyboard shortcuts in Rerun

- `Space` -- Play/Pause
- `Left/Right` -- Step one frame
- `Shift+Left/Right` -- Jump 10 frames
- Mouse drag in 3D view -- Rotate camera
- Scroll -- Zoom

## Output Files

| File | Description |
|------|-------------|
| `*.usda` | OpenUSD scene with point cloud, cameras, detected objects |
| `*_scene_graph.json` | Object graph from Gemini + camera metadata |
| `*.rrd` | Interactive Rerun recording (temporal 3D) |

## Running Tests

```bash
# All tests (includes real MASt3R + Gemini if available)
uv run python -m pytest test_pipeline.py -v

# Fast tests only (mock geometry, no GPU)
uv run python -m pytest test_pipeline.py -v -k "not real_mast3r and not real_video"
```

## Core Stack

1. **MASt3R** (Naver) -- SOTA multi-view stereo reconstruction
2. **Gemini** (Google) -- Multimodal reasoning for object state analysis
3. **SAM 3** (Meta) -- Segment Anything (stub, waiting for public release)
4. **OpenUSD** (Pixar) -- Industry-standard 3D scene format
5. **Rerun.io** -- Interactive temporal 3D visualization

## Architecture

```
Video --> [Keyframe Extraction] --> [MASt3R 3D] --> [SAM 3 Semantics] --> [Gemini Reasoning]
                                         |                  |                     |
                                   per-frame 3D        object masks         state changes
                                   point clouds        in 3D space          + causality
                                         |                  |                     |
                                         +------------------+---------------------+
                                                            |
                                                    [OpenUSD Export]
                                                    [Rerun .rrd Recording]
```

## Interfaces

- Camera motion from video (MASt3R camera poses)
- Object detection (Gemini + future SAM 3)
- Depth estimation from video (MASt3R dense depth maps)
- State change detection (Gemini causal reasoning)
- Motion models per object (planned: particle filter)

## Resources

- https://github.com/naver/mast3r
- https://rerun.io/
- https://openusd.org/
