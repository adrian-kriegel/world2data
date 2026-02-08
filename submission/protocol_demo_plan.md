# Protocol-to-Demo Implementation Plan (No Code Claims)

## Goal
Convert the OpenUSD Layering Protocol into a demo story with concrete milestones and acceptance checks.

## Milestone 1: Layered packaging skeleton
Deliverables:
- `scene/scene.usda` assembly entrypoint.
- `scene/layers/00_base.usda`, `90_overrides.usda`, optional `99_session.usda`.
- `scene/external/` structure for input video and recon assets.

Acceptance:
- Opening `scene/scene.usda` composes without errors.
- Stage metadata (`upAxis`, `metersPerUnit`, `timeCodesPerSecond`, `defaultPrim`) is defined once.

## Milestone 2: Per-run producer layers
Deliverables:
- `10_inputs_run_<RUNID>.usda`
- `20_recon_run_<RUNID>.usdc`
- `30_tracks_run_<RUNID>.usdc`
- `40_events_run_<RUNID>.usda`

Acceptance:
- Each layer writes only its owned namespace prefix.
- Each layer writes provenance record under `/World/W2D/Provenance/runs/<RUNID>`.

## Milestone 3: Referential integrity and replayability
Deliverables:
- Input video URI refs and point-cloud URI refs under `/World/W2D/Inputs/**`.
- Reconstruction/tracking/events linked by stable IDs (`w2d:uid`) and relationships.
- Intermediate checkpoint manifest per run.

Acceptance:
- Every entity has immutable `w2d:uid`.
- External asset paths are relative and resolve from layer location.

## Milestone 4: QA and conflict-safe override
Deliverables:
- Human correction layer (`90_overrides.usda`) with approval metadata.
- Explicit conflict policy docs in repo for "no shared property" rule.

Acceptance:
- Override layer can change upstream properties without editing upstream files.
- Override provenance (`overrideReason`, approver/timestamp where available) is present.

## Milestone 5: CI validation gate
Deliverables:
- Lightweight validator script/checklist for:
  - layer composition open
  - namespace ownership
  - required provenance
  - asset resolution
  - required IDs

Acceptance:
- CI fails on namespace/provenance violations.
- Demo branch can generate a deterministic composed stage from run layers.

## Demo Narrative Mapping
1. "No file merge conflicts": show immutable per-run layers.
2. "Deterministic composition": show `scene.usda` subLayer stack.
3. "Traceability": open provenance prim and run metadata.
4. "Human-in-the-loop": show override layer changing a property without touching source layers.
