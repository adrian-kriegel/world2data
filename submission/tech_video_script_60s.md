# Tech Video Script (60s max)

## Goal
Explain architecture and implementation depth.

## Timeline
0-10s: Stack overview:
- MASt3R (geometry)
- YOLOv8/SAM3 path (semantics)
- Gemini reasoning
- OpenUSD + Rerun outputs

10-24s: Explain pipeline flow:
video -> keyframes -> 3D reconstruction -> semantic/state fusion -> USD export

24-40s: Show concrete implementation highlights:
- deterministic output controls (seeded sampling)
- robust fallback logic for missing APIs
- particle filter integration for semantic temporal consistency
- explicit full-timeline check (start/end frame span shown in logs)

40-53s: Show test/validation:
- pytest pass snapshot
- generated scene graph metrics (frames, points, objects)
- call out current limits: reasoning path uses sampled keyframes, SAM3 prompt cap

53-60s: Protocol-aware next step:
- migrate from single output USD to layered composition (`scene.usda` + role layers)
- enforce namespace ownership + provenance per producer run

## Recording Checklist
- Keep one architecture diagram on-screen while speaking.
- Show code briefly with file names for credibility.
