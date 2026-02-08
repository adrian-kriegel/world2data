# World2Data Revised Plan

Last updated: 2026-02-08  
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

## WP1: Make Layered OpenUSD the Default Pipeline Output

Scope:

- integrate `usd_layers.py` into `World2DataPipeline` export path
- write run-scoped layered scene bundle under `data/outputs/runs/<RUNID>/scene/`

Deliverables:

- default output is `scene/scene.usda` + layer stack
- no direct dependency on monolithic `*.usda` for primary flow

Acceptance criteria:

- opening `scene/scene.usda` composes all authored layers
- run is reproducible with stable layer ordering and run IDs

## WP2: Externalize Reconstruction Payloads per Protocol

Scope:

- write per-frame point assets to `scene/external/recon/`
- stop authoring dense cloud arrays in protocol layers
- keep only compact metadata in `20_recon_run_<RUNID>.usda`

Deliverables:

- point assets indexed by frame
- camera poses/intrinsics remain in recon layer

Acceptance criteria:

- protocol layers contain no dense point arrays
- all frame point assets resolve via relative `asset` paths

## WP3: Perception and Tracking Layer Contracts

Scope:

- add YOLO observations layer writer (`25_yolo_run_<RUNID>.usda`)
- add particle-filter tracks writer (`30_tracks_run_<RUNID>.usda`)
- align namespaces and ownership with protocol

Deliverables:

- `/World/W2D/Observations/YOLO/**`
- `/World/W2D/Entities/**`, `/World/W2D/Tracks/**`

Acceptance criteria:

- all entities have stable `w2d:uid`
- `w2d:producedByRunId` exists on authored output prims

## WP4: Temporal Point Lineage + Rerun Adapter

Scope:

- generate point lineage parquet per run
- update Rerun emission to use lifecycle states
- support active-map rendering and point retirement/supersession

Deliverables:

- `point_lineage_<RUNID>.parquet`
- deterministic active map at any timeline position

Acceptance criteria:

- every rendered point has traceable lineage back to source frame asset
- Rerun active map can drop old/superseded points as timeline advances

## WP5: Calibration Consumption in Main Pipeline

Scope:

- add pipeline inputs for calibration artifacts
- inject calibration intrinsics/distortion into recon/tracking contracts

Deliverables:

- calibration-aware run configuration
- calibration provenance in run metadata

Acceptance criteria:

- run can consume `camera_calibration.json` and/or calibration USDA
- intrinsics contract is available in composed stage for downstream consumers

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

### Phase 2 (Temporal Fidelity)

- WP3, WP4
- full observation/tracking contracts + lineage-driven dynamic map behavior

### Phase 3 (Operational Hardening)

- WP6, WP7
- resumeability + CI protocol enforcement

## 7) Demo and Validation Commands

### 7.1 Full pipeline demo run

```bash
uv run world2data-demo --video data/inputs/video_2026-02-08_09-36-42.mp4 --output data/outputs/demo_output --fps 5
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
