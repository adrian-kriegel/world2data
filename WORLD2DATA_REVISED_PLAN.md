# World2Data Revised Plan

Last updated: 2026-02-08 (v2 -- WP1-5 implemented)  
Purpose: implementation plan for delivering the World2Data pipeline with `W2D_OpenUSD_Layering_Protocol.md` as protocol authority and the current `src/world2data` runtime as execution base.

## 1) Ground Truths and Goal

World2Data has two authoritative references:

- Runtime implementation in `src/world2data/` and `src/world2data/pipeline/`
- OpenUSD composition contract in `W2D_OpenUSD_Layering_Protocol.md`

Primary goal:

- produce protocol-compliant, layered OpenUSD outputs
- keep heavy geometry external
- preserve temporal point provenance so the point cloud is time-aware, auditable, and animatable

## 2) Current Runtime Baseline

### 2.1 Active Entrypoints

- `world2data-demo`: full multi-model run (`src/world2data/pipeline/demo_run.py`)
- `world2data-calibration`: ChArUco calibration (`src/world2data/calibration.py`)
- `world2data-generate-demo`: demo summary utilities (`src/world2data/pipeline/generate_demo.py`)
- `world2data-review`: human review UI (`src/world2data/pipeline/human_review_ui.py`)

### 2.2 Data Roots

- inputs: `data/inputs/`
- model weights: `data/models/`
- generated artifacts: `data/outputs/`

### 2.3 Active Pipeline Flow

- keyframe extraction
- MASt3R reconstruction
- YOLO detection
- SAM3 segmentation (if available)
- Gemini scene analysis + reasoning
- USD export + scene graph JSON + Rerun recording + PLY

## 3) Protocol-Driven Target Output Shape

Every production run writes a run-scoped scene bundle with OpenUSD composition:

```text
data/outputs/runs/<RUNID>/
  scene/
    scene.usda
    layers/
      00_base.usda
      10_inputs_run_<RUNID>.usda
      20_recon_run_<RUNID>.usda
      25_yolo_run_<RUNID>.usda
      30_tracks_run_<RUNID>.usda
      40_events_run_<RUNID>.usda
      90_overrides.usda
      99_session.usda
    external/
      inputs/
      recon/
      observations/
      tracking/
```

Composition rule:

- `scene/scene.usda` contains stage metadata + ordered sublayers only
- heavy assets are referenced via relative `asset` paths

## 4) Temporal Point-Cloud Contract (Non-Static by Design)

The point cloud is not a single static object in system-of-record form.

### 4.1 Source of Truth for Geometry

Per-frame stamped point assets are authoritative:

- `/World/W2D/Reconstruction/PointCloudFrames/f_<FRAME>`
- required attrs:
  - `w2d:frameIndex`
  - `w2d:timestampSec`
  - `w2d:pointsAsset`
  - `w2d:pointCount`
  - `w2d:pointsFormat`
  - `w2d:producedByRunId`

### 4.2 Point Lineage and Update Semantics

Each point observation is traceable to its origin frame and source asset.

Required external lineage table per run:

- `scene/external/recon/point_lineage_<RUNID>.parquet`
- columns:
  - `point_uid`
  - `frame_index_origin`
  - `timestamp_sec_origin`
  - `source_points_asset`
  - `x`, `y`, `z`
  - `r`, `g`, `b`
  - `confidence`
  - `state` (`active`, `superseded`, `retired`)
  - `superseded_by_point_uid` (nullable)

`point_uid` generation rule:

- deterministic hash of `(run_id, frame_index_origin, source_points_asset, local_point_index)`

### 4.3 Animation and Map Refresh Behavior

Rerun and demo views must support temporal map updates:

- current frame view shows only frame-local points (`world/current_view`)
- active map view shows points with `state=active` at time `t`
- superseded/retired points are excluded from active map at time `t`

This guarantees old geometry can be ignored and replaced when newer observations improve map consistency.

## 5) Implementation Work Packages

## WP1: Make Layered OpenUSD the Default Pipeline Output -- DONE

Implemented in `controller.py:step_5b_export_layered_usd()`.

- Integrated `usd_layers.py` into `World2DataPipeline` export path
- Writes run-scoped layered scene bundle under `<output_dir>/scene/`
- Called automatically in `run_ralph_loop()` after `step_5_export_usd()`
- Graceful fallback: if layered export fails, monolithic USD still produced

Acceptance: 14 tests in `tests/test_layered_pipeline.py` all pass.

## WP2: Externalize Reconstruction Payloads per Protocol -- DONE

Implemented via `USDLayerWriter.write_per_frame_point_clouds()` and `write_recon_layer_with_frames()`.

- Per-frame PLY files written to `scene/external/recon/frame_NNNNNN.ply`
- Compact metadata prims at `/World/W2D/Reconstruction/PointCloudFrames/f_NNNNNN`
- No dense point arrays in protocol layers (verified by test)
- All asset paths are relative

Acceptance: Tests `test_write_per_frame_ply_files`, `test_recon_layer_with_frame_index`, `test_no_dense_point_arrays_in_recon_with_frames` all pass.

## WP3: Perception and Tracking Layer Contracts -- DONE

Implemented via `USDLayerWriter.write_yolo_observations_layer()`.

- YOLO observations layer (`25_yolo_run_<RUNID>.usda`) with per-frame prims
- Per-frame attributes: `w2d:labels`, `w2d:classIds`, `w2d:scores`, `w2d:boxesXYXY`, `w2d:detectionCount`
- Tracks layer already existed; entities have `w2d:uid` and `w2d:producedByRunId`
- Namespace: `/World/W2D/Observations/YOLO/Frames/f_NNNNNN`

Acceptance: Tests `test_writes_yolo_layer`, `test_yolo_layer_in_assembly` pass.

## WP4: Temporal Point Lineage + Rerun Adapter -- DONE

Implemented via `write_point_lineage()` in `usd_layers.py` and updated `save_temporal_rrd()` in `controller.py`.

- Point lineage parquet written to `scene/external/recon/point_lineage_<RUNID>.parquet`
- Schema: `point_uid, frame_index_origin, timestamp_sec_origin, source_points_asset, x, y, z, r, g, b, confidence, state`
- Deterministic `point_uid` via SHA-256 of `(run_id, frame_index, asset, local_index)`
- Rerun now has three views: `world/current_view` (frame-local), `world/active_map` (sliding window with recency fading), `world/full_history` (complete accumulation)
- Older points fade in the active map; full history preserved separately

Acceptance: Tests `test_writes_lineage_parquet`, `test_lineage_schema`, `test_lineage_row_count`, `test_lineage_all_active` pass.

## WP5: Camera Poses and Calibration in Protocol Layers -- DONE

Implemented as part of `write_recon_layer_with_frames()`.

- Per-frame camera pose prims at `/World/W2D/Sensors/CameraPoses/Frames/f_NNNNNN`
- Attributes: `w2d:frameIndex`, `w2d:timestampSec`, `w2d:translation`, `w2d:poseConvention`, `w2d:producedByRunId`
- Camera intrinsics via standard `UsdGeom.Camera` prims under `/World/W2D/Sensors/Rig_01/`
- Calibration JSON consumption available via `world2data-calibration` CLI

Acceptance: Test `test_recon_layer_with_frame_index` verifies pose prims.

## WP6: Run Manifests, Checkpoints, and Resume

Scope:

- create run manifest (`run_manifest_<RUNID>.json`)
- stage-wise checkpoint records and resume support
- deterministic restart semantics

Deliverables:

- manifest with stage status, artifacts, hashes
- resume command that skips validated completed stages

Acceptance criteria:

- interrupted run restarts from last valid stage without recomputing completed stages
- manifest captures full artifact graph and provenance references

## WP7: Protocol CI Gates

Scope:

- enforce protocol validations in automated tests/CI

Checks:

- composition opens cleanly
- namespace ownership rules
- `w2d:uid` presence for entities
- relative `asset` path resolution
- provenance run records
- no dense point arrays in protocol layers
- temporal point lineage file exists and validates schema

Acceptance criteria:

- CI fails on first protocol violation with actionable diagnostics

## 6) Execution Sequence

### Phase 1 (Foundation)

- WP1, WP2, WP5
- output structure + calibration + external recon indexing

### Phase 2 (Temporal Fidelity) -- DONE

- WP3, WP4
- full observation/tracking contracts + lineage-driven dynamic map behavior
- **WP3.5 (NEW)**: MultiObjectParticleFilter integrated as Step 3d
  - Converts MASt3R poses + YOLO 2D detections -> 3D track estimates
  - Runs BEFORE Gemini/reasoning (spatial anchoring)
  - Feeds into: layered USD tracks, Rerun bounding boxes, point lineage
  - 7 unit tests + 5 E2E tests, all passing
- **WP4.5 (NEW)**: PF-enriched point lineage with lifecycle
  - `track_id` column associates points to PF tracks
  - Lifecycle states: active (recent) / stale (tracked, older) / retired

### Phase 2b (Model Integration) -- IN PROGRESS

- SAM3: Code wired, needs HuggingFace login (`uv run huggingface-cli login`)
- Gemini: Enriched with PF 3D context (tracked_objects_3d in mast3r_summary)
- Protocol sections 12.8-12.10 added for Gemini + SAM3 producer contracts

### Phase 3 (Operational Hardening)

- WP6, WP7
- resumeability + CI protocol enforcement

## 7) Demo and Validation Commands

### 7.1 Full pipeline demo run (investor demo)

```bash
uv run world2data-demo --video data/inputs/video_2026-02-08_12-12-15.mp4 --output data/outputs/investor_demo --fps 10
```

### 7.1b View demo outputs

```bash
# Interactive 3D temporal viewer (Rerun)
uv run rerun data/outputs/investor_demo/investor_demo.rrd

# USD scene (usdview -- requires USD tools)
uv run python -c "from pxr import Usd; print(Usd.Stage.Open('data/outputs/investor_demo/scene/scene.usda').ExportToString()[:2000])"

# Or open in any USD viewer:
# usdview data/outputs/investor_demo/scene/scene.usda
```

### 7.2 Calibration run

```bash
uv run world2data-calibration --video data/inputs/camera_calibration.mp4 --output-json data/outputs/camera_calibration.json --output-usda data/outputs/camera_calibration.usda
```

### 7.3 Overnight integration validation

```bash
uv run python -m pytest tests/test_pipeline.py -v --overnight -k overnight
```

### 7.4 Layered USD protocol validation target

- run layered-writer tests and scene validation in `tests/test_usd_layers.py`
- enforce added CI gates for temporal lineage and no-dense-array rule

## 8) Definition of Done

A run is done when:

- `scene/scene.usda` composes all protocol layers successfully
- reconstruction point data is externalized and frame-indexed
- point lineage is persisted and queryable
- Rerun demonstrates temporal point updates (active vs superseded/retired)
- outputs are fully provenance-linked to run and model configuration
- CI protocol gates pass without exemptions
