# World2Data — OpenUSD Layering Protocol (Prototype‑Ready Spec)

**Status:** Prototype v0.1  
**Scope:** Defines how multiple pipeline components author and compose OpenUSD outputs deterministically, merge‑safely, and with provenance.

This protocol assumes a pipeline that ingests **video**, estimates **camera motion**, produces **point clouds**, detects and tracks **entities**, and infers **events/relations** (e.g., *“man drinks from mug”*) as a queryable scene graph.

---

## 1. Objectives (Non‑Negotiables)

1. **No physical merging by default.** Use OpenUSD composition (layer stacks) as the standard “automatic merge” mechanism.
2. **Deterministic composition.** Given the same set of layers, the composed stage must be identical.
3. **Merge‑safe collaboration.** Multiple producers can write outputs without overwriting each other.
4. **Provenance‑first.** Every authored result must be traceable to a pipeline run + model version + parameters.
5. **Data scalability.** Heavy payloads (video, dense point clouds, meshes) must remain external and be referenced by `asset` paths (never inlined arrays in protocol layers).

---

## 2. Key Concepts

### 2.1 “Merge” in OpenUSD
**Automatic merging** is achieved via **composition**:
- A small **assembly stage** (`scene.usda`) sublayers multiple output layers.
- **Stronger** layers override **weaker** layers deterministically by sublayer order.

Physical merge/flatten is reserved for:
- Export to consumers that require a single file
- Snapshotting for archival delivery

### 2.2 Layer roles
We standardize layers into roles:
- **Base**: conventions and skeleton namespaces
- **Inputs**: references to raw inputs (video, point cloud URIs)
- **Reconstruction**: cameras, calibration, optional mesh/points caches
- **Tracking**: entities + tracks + associations
- **Events/Graph**: inferred relations/events
- **Overrides/QA**: human corrections / approvals
- **Session**: local user tweaks (not checked in)

### 2.3 “Namespace Ownership”
A producer **owns** specific prim path prefixes and must not author outside them (except via explicit, documented extension points).

---

## 3. Directory Layout (Canonical)

```
scene/
  scene.usda                      # assembly entrypoint (open this file)
  layers/
    00_base.usda                  # conventions + empty scopes
    10_inputs_run_<RUNID>.usda    # input refs + metadata
    20_recon_run_<RUNID>.usdc     # cameras + recon outputs
    30_tracks_run_<RUNID>.usdc    # entities + tracks
    40_events_run_<RUNID>.usda    # events/relations graph
    90_overrides.usda             # human QA / final decisions (strong)
    99_session.usda               # per-user local edits (strongest; ignored by VCS)
  external/
    inputs/office.mp4
    recon/cloud_01.ply
    recon/cloud_01_cache.usdc     # optional USD cache payload
```

**Rule:** Everything under `layers/` is composable. Everything under `external/` is data referenced by `asset` paths.

---

## 4. Assembly Stage (`scene.usda`) Specification

### 4.1 Responsibilities
`scene.usda` must:
- Declare stage‑level metadata: `upAxis`, `metersPerUnit`, `timeCodesPerSecond`, `startTimeCode`, `endTimeCode`
- Define `defaultPrim`
- Provide **only** the **subLayer stack**
- Contain no authored geometry/semantics (keep it stable and small)

### 4.2 Sublayer order (Weak → Strong)
Sublayers listed earlier are weaker. Later layers override earlier ones.

**Standard order:**
1. `00_base.usda`
2. `10_inputs_run_<RUNID>.usda`
3. `20_recon_run_<RUNID>.usdc`
4. `30_tracks_run_<RUNID>.usdc`
5. `40_events_run_<RUNID>.usda`
6. `90_overrides.usda`
7. `99_session.usda` (optional, local only)

### 4.3 Minimal assembly example
```usda
#usda 1.0
(
  defaultPrim = "World"
  metersPerUnit = 1
  upAxis = "Y"
  timeCodesPerSecond = 30
  startTimeCode = 0
  endTimeCode = 1000
  subLayers = [
    @./layers/00_base.usda@,
    @./layers/10_inputs_run_01JX...usda@,
    @./layers/20_recon_run_01JX...usdc@,
    @./layers/30_tracks_run_01JX...usdc@,
    @./layers/40_events_run_01JX...usda@,
    @./layers/90_overrides.usda@,
    @./layers/99_session.usda@
  ]
)
def Xform "World" {}
```

---

## 5. Canonical Namespace Map (Ownership Table)

All pipeline artifacts live under:
- `/World/W2D/**`

| Namespace Prefix | Owner | Content |
|---|---|---|
| `/World/W2D/Inputs/**` | Ingest | Video refs, point cloud refs, calibration blobs |
| `/World/W2D/Sensors/**` | Recon | Cameras, rigs, intrinsics/extrinsics |
| `/World/W2D/Reconstruction/**` | Recon | Point cloud caches, meshes, depth assets |
| `/World/W2D/Entities/**` | Tracking | Entity instances (people/objects) |
| `/World/W2D/Tracks/**` | Tracking | Track prims + time‑varying poses |
| `/World/W2D/Observations/**` | Perception | Per‑frame detections/segmentations (often as external refs) |
| `/World/W2D/Events/**` | Reasoning | Actions/relations with time intervals |
| `/World/W2D/Graph/**` | Reasoning | Optional indexes / aggregations |
| `/World/W2D/Provenance/**` | All | Run metadata, model versions, hashes |
| `/World/W2D/Overrides/**` | QA/Human | Corrections and approvals (strong layer) |

**Hard rule:** A component must not author properties outside its owned prefixes unless the spec defines an extension point.

---

## 6. Identity & Referencing Rules

### 6.1 Stable IDs
Every real‑world entity must have:
- Stable prim path (preferred), and
- `string w2d:uid` (required, immutable)

**Example:**
- `/World/W2D/Entities/Objects/mug_017`
- `w2d:uid = "01JX...ULID"`

### 6.2 Referencing external files
Videos and point clouds must be stored as external files and referenced via `asset` attributes:
- `asset w2d:uri = @../external/inputs/office.mp4@`
- `asset w2d:uri = @../external/recon/cloud_01.ply@`

**Optional cache:** For performance, provide a USD payload cache (e.g., `cloud_01_cache.usdc`) but keep the external file as authoritative.

### 6.3 Paths and portability
- Prefer **relative asset paths** from the layer location.
- Never write absolute machine paths into committed layers.

### 6.4 Large-data authoring ban
- Protocol layers must not author large binary-like arrays (for example `float3[] w2d:points` for dense clouds).
- Large geometry/depth content must be externalized and referenced via `asset` attributes.
- Layers may store compact metadata only (timestamps, frame indices, counts, hashes, bounds, provenance).

---

## 7. Conflict Policy (What happens if two layers disagree?)

OpenUSD resolves conflicts by **strength**:
- Later sublayers override earlier sublayers for the same property.

To keep behavior predictable, we define:

### 7.1 “No shared property” rule
Two producers must not author the **same property** on the **same prim** unless explicitly coordinated.

Example conflict:
- Both recon and tracking author `xformOp:transform` on `/World/W2D/Sensors/Rig_01/Cam_01`

**Fix:** Define ownership (recon owns camera transforms) and enforce it.

### 7.2 Approved overrides
Human QA overrides must live in `90_overrides.usda` (or `/World/W2D/Overrides/**`) and can override any upstream output, but must include provenance tags (see §8).

### 7.3 Competing hypotheses pattern (recommended)
If multiple models produce alternative results, do **not** fight in the same properties. Use one of:
1. **Separate namespaces**
   - `/World/W2D/Tracks/modelA/...`
   - `/World/W2D/Tracks/modelB/...`
   - Resolver writes final `/World/W2D/Tracks/...`
2. **VariantSets**
   - `variantSet "trackingHypothesis" = { "modelA", "modelB" }`
   - Resolver selects active variant

---

## 8. Provenance (Required on every producer layer)

Each layer must author a run record under:
- `/World/W2D/Provenance/runs/<RUNID>`

Minimum fields (recommended):
- `string w2d:runId`
- `string w2d:gitCommit`
- `string w2d:component` (e.g., recon, tracking, reasoning)
- `string w2d:modelName`
- `string w2d:modelVersion`
- `dictionary w2d:params`
- `string w2d:timestampIso8601`

Each output prim must include:
- `string w2d:producedByRunId = "<RUNID>"`

**Goal:** Any event edge (“drinksFrom”) is auditable to exact code+params.

---

## 9. Timebase & Alignment

Pick **one** canonical timebase for the entire stage:

### Option A (recommended): Stage time = video frames
- `timeCodesPerSecond = video_fps`
- timeSamples keyed by integer frame indices
- Events use `w2d:tStart` / `w2d:tEnd` in frames

### Option B: Stage time = seconds
- `timeCodesPerSecond = 1` (or 1000)
- store seconds as timeCodes

**Rule:** All producers must respect the stage timebase declared in `scene.usda`.

---

## 10. When to Flatten (Physical Merge)

Flattening is allowed only for:
- export to a consumer requiring one file
- archival snapshot of a composed “truth” stage

Flattening is **not** the collaboration mechanism because it:
- destroys layer provenance and editability
- makes diff/merge harder in VCS
- encourages overwriting rather than composing

Deliverables:
- `final_composed.usdc` (flattened)
- `final_composed.usda` (optional ASCII for debugging)

---

## 11. CI Validation (Prototype‑Ready Checks)

A minimal CI step should verify:
1. `scene.usda` opens and composes cleanly
2. No authoring outside namespace ownership (lint)
3. All entity prims have `w2d:uid`
4. All external `asset` paths resolve (relative) in the repo layout
5. Cameras have `xformOpOrder` when using xformOps
6. Provenance run record exists for each producer layer
7. No dense point-cloud arrays are authored inside protocol layers (asset refs only)

---

## 12. Producer Contracts (What each component must output)

### 12.1 Ingest (`10_inputs_run_<RUNID>.usda`)
- `/World/W2D/Inputs/Video_*` with `asset w2d:uri`
- `/World/W2D/Inputs/PointCloud_*` with `asset w2d:uri`
- `/World/W2D/Provenance/runs/<RUNID>`

### 12.2 Reconstruction (`20_recon_run_<RUNID>.usdc`)
- `/World/W2D/Sensors/**` cameras with time‑sampled poses
- point-cloud/depth data referenced via `asset` attributes (no inlined dense arrays)
- optional cached USD payload under `/World/W2D/Reconstruction/**`
- provenance

### 12.3 Tracking (`30_tracks_run_<RUNID>.usdc`)
- `/World/W2D/Entities/**` entity prims with `w2d:uid`, `w2d:class`
- `/World/W2D/Tracks/**` track prims with time‑sampled transforms
- provenance

### 12.4 Reasoning (`40_events_run_<RUNID>.usda`)
- `/World/W2D/Events/**` event prims with:
  - `w2d:predicate`, `w2d:tStart`, `w2d:tEnd`, `w2d:confidence`
  - `rel w2d:subject`, `rel w2d:object`
- provenance

### 12.5 QA Overrides (`90_overrides.usda`)
- Only edits/corrections
- Must preserve upstream references and add provenance tags like:
  - `string w2d:overrideReason`
  - `string w2d:approvedBy` (optional)
  - `string w2d:approvedAt` (optional)

### 12.6 Perception YOLO (`25_yolo_run_<RUNID>.usda/.usdc`)
- `/World/W2D/Observations/YOLO/**` raw 2D detections per frame
- provenance

### 12.7 Particle Filter Tracking (`30_tracks_run_<RUNID>.usda/.usdc`)
- consumes calibration + camera poses + YOLO + stamped point-cloud asset references
- writes only centroid trajectory + mean bounding box dimensions per track
- provenance

---

## 13. Operational Workflow (How to “merge automatically”)

1. Each component writes its own output layer file (immutable naming).
2. A coordinator updates `scene.usda` subLayers list (or generates it from a manifest).
3. Users open `scene.usda` in their USD viewer or application.
4. Composition yields the full stage. No file merging needed.
5. Optional: flatten for export.

---

## 14. Notes for Prototype Implementation

- Start with **vanilla prims** (`Scope`, `Xform`, `Camera`) plus `w2d:*` attributes.
- Add typed schemas later (USD plugin) once the contract stabilizes.
- Keep point clouds/depth external by default:
  - authoritative files (`.ply/.pcd/.las`) referenced via `asset`
  - optional USD cache payload for visualization/performance.

---

## 15. Concrete Schemas (Current Prototype)

### 15.1 Calibration Intrinsics Schema (Input to Particle Filter)
- Layer role: reconstruction/calibration
- Prim path: `/World/W2D/Sensors/CalibrationCamera` (`Camera`)
- Required attributes:
  - `matrix3d w2d:intrinsicMatrix`
  - `int w2d:imageWidth`
  - `int w2d:imageHeight`
  - `string w2d:distortionModel`
  - `float[] w2d:distortionCoeffs`
  - `string w2d:producedByRunId`
- Pose requirement:
  - calibration layer must **not** author camera world pose; intrinsics only.

### 15.2 Camera Pose Frames Schema (Input to Particle Filter)
- Layer role: reconstruction/camera motion
- Scope root: `/World/W2D/Sensors/CameraPoses/Frames`
- Per-frame prim: `/World/W2D/Sensors/CameraPoses/Frames/f_<FRAME>`
- Required attributes per frame:
  - `int w2d:frameIndex`
  - `double w2d:timestampSec`
  - `matrix3d w2d:rotationMatrix`
  - `float3 w2d:translation`
  - `string w2d:poseConvention = "world_to_camera"`
  - `string w2d:producedByRunId`
- Semantics:
  - `rotationMatrix` and `translation` encode `x_cam = R * x_world + t`.

### 15.3 YOLO Raw Observations Schema (Input to Particle Filter)
- Layer role: perception
- Scope roots:
  - `/World/W2D/Observations/YOLO`
  - `/World/W2D/Observations/YOLO/Frames`
- Per-frame prim: `/World/W2D/Observations/YOLO/Frames/f_<FRAME>`
- Required per-frame attributes:
  - `int w2d:frameIndex`
  - `double w2d:timestampSec`
  - `int w2d:imageWidth`
  - `int w2d:imageHeight`
  - `string[] w2d:labels`
  - `int[] w2d:classIds`
  - `float[] w2d:scores`
  - `float4[] w2d:boxesXYXY` (`[xMin, yMin, xMax, yMax]` pixels)
  - `int w2d:detectionCount`
  - `string w2d:producedByRunId`
- Optional detection child prims (equivalent source of truth):
  - `/World/W2D/Observations/YOLO/Frames/f_<FRAME>/det_<IDX>`
  - `int w2d:classId`, `string w2d:class`, `float w2d:confidence`, `float4 w2d:bboxXYXY`

### 15.4 Stamped Point Cloud Frames Index Schema (Input to Particle Filter)
- Layer role: reconstruction/depth
- Scope root: `/World/W2D/Reconstruction/PointCloudFrames`
- Per-frame prim: `/World/W2D/Reconstruction/PointCloudFrames/f_<FRAME>`
- Required attributes per frame:
  - `int w2d:frameIndex`
  - `double w2d:timestampSec`
  - `asset w2d:pointsAsset` (external file URI/path, world coordinates in meters)
  - `string w2d:producedByRunId`
- Recommended compact metadata:
  - `string w2d:pointsFormat` (e.g., `ply`, `pcd`, `las`)
  - `int w2d:pointCount`
  - `string w2d:pointsSha256` (optional integrity check)
- Forbidden:
  - `float3[] w2d:points` or other dense in-layer point payloads.

### 15.5 Particle Filter Adapter Input Contract
- Required composed inputs:
  - calibration intrinsics layer (15.1)
  - camera pose frames layer (15.2)
  - YOLO observations layer (15.3)
  - stamped point cloud index layer with `w2d:pointsAsset` (15.4)
- Join key:
  - `w2d:frameIndex` across camera poses, YOLO frames, and point-cloud frames.
- Time:
  - `w2d:timestampSec` used for `dt`; stage `timeCodesPerSecond` provides fallback.

### 15.6 Particle Filter Tracks Schema (Output)
- Layer role: tracking
- Scope roots:
  - `/World/W2D/Entities/Objects`
  - `/World/W2D/Tracks/ParticleFilter`
- Per-entity prim: `/World/W2D/Entities/Objects/<TRACK_ID_SAFE>`
  - `string w2d:uid` (stable track id)
  - `string w2d:class`
  - `rel w2d:track -> /World/W2D/Tracks/ParticleFilter/<TRACK_ID_SAFE>`
  - `string w2d:producedByRunId`
- Per-track prim: `/World/W2D/Tracks/ParticleFilter/<TRACK_ID_SAFE>` (`Xform`)
  - `xformOp:translate` (time-sampled centroid in world coordinates)
  - `float3 w2d:meanBoundingBox` (time-sampled; mean PF box dimensions in meters)
  - `int w2d:particleCount` (time-sampled)
  - `string w2d:trackId`
  - `string w2d:class`
  - `string w2d:classHistoryJson` (label-count map)
  - `string w2d:producedByRunId`
- Tracking scope summary:
  - `/World/W2D/Tracks/ParticleFilter`
  - `int w2d:trackCount`
  - `int w2d:processedFrameCount`
  - `string w2d:component = "tracking.particle_filter"`
  - `string w2d:producedByRunId`
- Provenance:
  - `/World/W2D/Provenance/runs/<RUNID>` with required run metadata fields.

---

## 16. Quick Reference (Do / Don’t)

**Do**
- One output layer per component per run
- Enforce namespace ownership
- Use composition to “merge”
- Keep heavy data external or payloaded
- Put human decisions in a strong override layer

**Don’t**
- Flatten as your collaboration format
- Let two producers author the same property on the same prim
- Store absolute file paths in committed layers
- Put large binary data directly into the assembly stage
- Author dense point-cloud arrays inside protocol layers

---

## Appendix A — RunID & File Naming

Recommended:
- `RUNID = ULID` or `YYYYMMDD_HHMMSS_<short_hash>`
- Files:
  - `30_tracks_run_<RUNID>.usdc`
  - `40_events_run_<RUNID>.usda`

This enables reproducibility and side‑by‑side comparisons.
