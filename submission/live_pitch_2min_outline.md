# Live Pitch Outline (2 minutes, 1-3 slides)

## Slide 1: Problem + Why Now (35s)
- Manual 3D scene setup is slow and expensive.
- Real-world AI/robotics teams need faster world grounding.
- World2Data: 2D video -> 3D dynamic scene graph -> OpenUSD.

## Slide 2: What We Built (55s)
- Pipeline steps:
  1. Keyframe extraction
  2. MASt3R 3D reconstruction
  3. Semantic/state enrichment (YOLOv8/SAM3/Gemini path with fallback)
  4. OpenUSD + Rerun outputs
- Show one architecture diagram + one screenshot from Rerun.

## Slide 3: Results + Impact (30s)
- Example run: ~4.0M points, 120 temporal frames, USD + scene graph + Rerun artifacts.
- Immediate value: faster simulation setup and repeatable scene understanding from simple video.
- Ask: "With more time, we move to strict layered OpenUSD composition + CI namespace/provenance validation."

## Demo Insert (within 2 min)
- 20-30s clip from your 60s demo video showing input -> 3D output.
