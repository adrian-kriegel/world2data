# Tech Video Script (60s max)

## Goal
Explain architecture and implementation depth.

## Timeline
0-10s: Stack overview:
- MASt3R (geometry)
- YOLOv8/SAM3 path (semantics)
- Gemini reasoning
- OpenUSD + Rerun outputs

10-25s: Explain pipeline flow:
video -> keyframes -> 3D reconstruction -> semantic/state fusion -> USD export

25-40s: Show concrete implementation highlights:
- deterministic output controls (seeded sampling)
- robust fallback logic for missing APIs
- particle filter integration for semantic temporal consistency

40-52s: Show test/validation:
- pytest pass snapshot
- generated scene graph metrics

52-60s: Limitations and next step:
- SAM3 access is gated and must be approved
- next: stronger full-frame tracking in fusion path

## Recording Checklist
- Keep one architecture diagram on-screen while speaking.
- Show code briefly with file names for credibility.
