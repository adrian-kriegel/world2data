# Demo Pitfalls and Limits (Judge-Facing)

## Why this file exists
This is the quick, honest risk brief we can speak to during demo Q&A.

## Top common pitfalls (current prototype)
1. Stale artifact playback
- Symptom: Rerun shows fewer/older frames than latest run.
- Cause: opening an old `.rrd` from prior run.
- Mitigation: clear or overwrite `overnight_output/overnight.rrd` before demo run; verify timestamp.

2. Monolithic USD collaboration friction
- Symptom: multiple model outputs collide in one `.usda`.
- Cause: no strict per-component layer ownership in prototype output path.
- Mitigation: move to `scene.usda` assembly + role-based sublayers from `W2D_OpenUSD_Layering_Protocol.md`.

3. Silent context truncation in reasoning paths
- Symptom: later-scene events underrepresented.
- Cause: some reasoning prompts sample keyframes instead of using all.
- Mitigation: explicitly report frame-span coverage and where sampling is used.

4. Heavy model path runtime variance
- Symptom: inconsistent runtime on SAM3-heavy runs.
- Cause: segmentation/tracking cost scales with frame count and prompt count.
- Mitigation: cap prompt classes for demo, keep full-span keyframes, and disclose tradeoff.

## Model context/throughput budget (practical)
1. MASt3R reconstruction
- Budget driver: number of keyframes and pair graph strategy.
- Current demo-safe setup: full-span keyframes with explicit cap for runtime control.

2. YOLOv8 detection
- Budget driver: per-frame inference across extracted keyframes.
- Current behavior: runs across extracted keyframes; good for coverage, linear cost.

3. SAM3 segmentation
- Budget driver: `frames x prompts`.
- Current behavior: frame count can run full extracted sequence; prompt list is constrained (top classes) for practicality.

4. Gemini video analysis
- Budget driver: video size and upload mode.
- Current clip (~12.7MB) is below inline threshold and comfortably processed.

5. Gemini reasoning over keyframes
- Budget driver: multimodal prompt size.
- Current behavior: sampled keyframes are used in reasoning step; this is a known precision tradeoff.

## Are we near limits?
- For the current ~33s demo clip: no hard provider limit is hit.
- The highest practical risk is runtime/latency variance, not token hard-fail.

## What allows more detail safely
1. Increase keyframe cap gradually while monitoring overnight runtime.
2. Keep full-span timeline checks in logs (`start/end seconds`).
3. Use layered OpenUSD outputs to preserve intermediate artifacts and provenance.
4. Maintain per-stage checkpoint saves to resume after interruption.

## Demo speaking line
"Coverage is full-span, artifacts are reproducible, and our known limits are throughput tradeoffs rather than hidden failure modes."
