# World2Data Revised Plan

Last updated: 2026-02-08  
Purpose: implementation overview of the current World2Data system, focused on how it runs end-to-end and how data moves through it.

## 1) System Goal

World2Data converts monocular video into inspectable 3D + temporal scene data.

For each run, the system produces:

- reconstructed 3D geometry and camera trajectory
- object-level scene understanding from multiple models
- reasoning-enriched scene metadata
- review-ready artifacts for USD, Rerun, and JSON workflows

## 2) Runtime Architecture

### 2.1 Package Layout

- Core package: `src/world2data/`
- Pipeline package: `src/world2data/pipeline/`
- Tests: `tests/`
- Data roots:
  - `data/inputs/`
  - `data/models/`
  - `data/outputs/`

### 2.2 Entrypoints

The project exposes these CLI commands via `pyproject.toml`:

- `world2data` -> lightweight core particle-filter smoke entrypoint
- `world2data-calibration` -> ChArUco camera calibration tool
- `world2data-demo` -> full multi-model demo runner
- `world2data-generate-demo` -> summary/report helper for generated outputs
- `world2data-review` -> Gradio human-review UI

## 3) End-to-End Processing Flow

```text
+-----------------------------+
| Input Video (.mp4)          |
| data/inputs/... or any path |
+--------------+--------------+
               |
               v
+-----------------------------+
| Optional Camera Calibration |
| world2data-calibration      |
+--------------+--------------+
               |
               v
+-----------------------------+
| World2DataPipeline          |
| (Ralph Loop)               |
+--------------+--------------+
               |
               v
+-----------------------------+
| Step 1: Keyframe extraction |
| L1 diff + FPS sampling      |
+--------------+--------------+
               |
               v
+-----------------------------+
| Step 2: MASt3R recon        |
| depth, poses, point clouds  |
+--------------+--------------+
               |
     +---------+----------+--------------------+
     |                    |                    |
     v                    v                    v
+-----------+     +---------------+     +----------------+
| YOLOv8x   |     | SAM3          |     | Gemini Video   |
| detection |     | segmentation  |     | scene analysis |
+-----------+     +---------------+     +----------------+
     \                |                    /
      \               |                   /
       +--------------+------------------+
                      |
                      v
         +------------------------------+
         | Scene fusion + reasoning     |
         | confidence + review flags    |
         +--------------+---------------+
                        |
                        v
         +------------------------------+
         | OpenUSD + sidecar export     |
         | + Rerun + PLY export         |
         +--------------+---------------+
                        |
         +--------------+------------------------------+
         |                                             |
         v                                             v
+------------------------+                  +------------------------+
| Demo summary / scripts |                  | Human review workflow  |
| world2data-generate-demo|                 | world2data-review      |
+------------------------+                  +------------------------+
```

## 4) Model Stack and Roles

### 4.1 Geometry

- MASt3R handles geometric reconstruction from keyframes.
- Output includes per-frame geometry, camera poses, and accumulated point cloud.

### 4.2 Detection and Segmentation

- YOLOv8 (`yolov8x-seg`) provides 2D detections and masks.
- SAM3 adds prompt-driven segmentation/tracking when available.

### 4.3 Semantic Analysis and Reasoning

- Gemini video analysis contributes scene-level objects/events.
- Scene fusion combines multi-model evidence into 4D object records.
- Reasoning output contributes confidence, traceability, and review flags.

## 5) Calibration Workflow

Camera calibration is a first-class workflow via `world2data-calibration`.

- Input: ChArUco calibration video
- Output JSON (default): `data/outputs/camera_calibration.json`
- Output USDA (default): `data/outputs/camera_calibration.usda`
- USDA includes a calibration camera prim and custom `w2d:*` calibration attributes.

Recommended command:

```bash
uv run world2data-calibration \
  --video data/inputs/camera_calibration.mp4 \
  --output-json data/outputs/camera_calibration.json \
  --output-usda data/outputs/camera_calibration.usda
```

## 6) Output Artifacts and Data Handling

### 6.1 Output Location

Default demo outputs are written under:

- `data/outputs/demo_output/`

The demo runner writes a complete bundle per run:

- `demo_scene.usda`
- `demo_scene_scene_graph.json`
- `demo_scene.rrd`
- `demo_scene.ply`
- `demo_annotated.mp4`
- `demo_human_review.json`
- `keyframes/`
- `cache/`

### 6.2 Intermediate Data

During processing, the pipeline uses:

- `keyframe_dir` for extracted keyframes
- `cache_dir` for MASt3R artifacts and sliding-window reconstruction caches

The demo runner configures these as persistent subfolders inside the chosen output directory, so runs are fully inspectable and portable.

### 6.3 Point Cloud Representation

Point cloud data is persisted as:

- `.ply` for external 3D tools
- USD `Points` prims for scene-level visualization and composition

## 7) OpenUSD Authoring

World2Data includes two USD authoring paths:

- Pipeline scene export (`controller.py`) for direct end-to-end runs
- Layered USD authoring utility (`usd_layers.py`) for protocol-driven composed stages

Layered writer supports:

- base/input/recon/tracks/events/overrides/session layers
- deterministic assembly stage generation
- provenance records under `/World/W2D/Provenance/runs/...`

## 8) Run Commands

### 8.1 Full demo pipeline

```bash
uv run world2data-demo \
  --video data/inputs/video_2026-02-08_09-36-42.mp4 \
  --output data/outputs/demo_output \
  --fps 5
```

### 8.2 Generate demo summary from JSON

```bash
uv run world2data-generate-demo \
  --json data/outputs/demo_output/demo_scene_scene_graph.json
```

### 8.3 Launch human review UI

```bash
uv run world2data-review \
  --json data/outputs/demo_output/demo_human_review.json
```

### 8.4 Overnight integration validation

```bash
uv run python -m pytest tests/test_pipeline.py -v --overnight -k overnight
```

## 9) Validation and Test Coverage

The test suite validates:

- calibration outputs (`tests/test_calibration.py`)
- pipeline stages and overnight end-to-end coverage (`tests/test_pipeline.py`)
- demo runner components (`tests/test_demo_components.py`)
- OpenUSD compatibility (`tests/test_openusd_compat.py`)
- layered USD protocol authoring/validation (`tests/test_usd_layers.py`)
- particle filter behavior (`tests/test_particle_filter.py`)
