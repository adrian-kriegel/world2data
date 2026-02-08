# Global AI Hackathon - Concrete Execution Plan

## Current status (as of Feb 8, 2026)
- Core pipeline is operational and tested.
- Submission scaffolding is prepared under `submission/` and `scripts/`.
- Main remaining risk is external service access/quotas (SAM3 gate + Gemini limits).

## Plan with concrete deliverables

## Phase 1 (0-30 min): unblock gated dependencies
1. Run `python scripts/check_sam3_hf_access.py`.
2. If access is denied:
   - Open https://huggingface.co/facebook/sam3
   - Click **Request access** / accept terms.
   - Create token with **Read** scope at https://huggingface.co/settings/tokens.
   - Set env var `HF_TOKEN`.
3. Re-run the check script until it passes.

## Phase 2 (30-75 min): lock final artifacts
1. Generate one-page PDF:
   - `python submission/make_onepager_pdf.py`
2. Refresh demo outputs (legacy mode for reliability under quota limits):
   - `python pipeline_controller.py test_video.mp4 --output final_demo --legacy --no-rerun --max-keyframes 12`
3. Choose best artifact set (`real_video_output_*` or `final_demo_*`) for videos/screenshots.

## Phase 3 (75-130 min): record required videos
1. Demo video (<=60s):
   - Use `submission/demo_video_script_60s.md`
   - Record with OBS/Loom.
2. Tech video (<=60s):
   - Use `submission/tech_video_script_60s.md`
   - Include architecture + code + tests.
3. Export both to MP4 and ensure public-share links.

## Phase 4 (130-170 min): finalize submission package
1. Generate zip and manifest:
   - `python scripts/prepare_hackathon_submission.py`
2. Confirm checklist:
   - `submission/submission_checklist.md`
3. Prepare 1-3 live pitch slides:
   - follow `submission/live_pitch_2min_outline.md`.

## Phase 5 (170-210 min): submit to both portals
1. Submit form: https://tinyurl.com/HN-4-submit
2. Submit project: https://projects.hack-nation.ai/
3. Verify every link is publicly accessible.

## Quality gates before clicking submit
1. Summary is 150-300 words and clearly non-technical.
2. One-page PDF exists and is exactly one page.
3. Demo + tech video each <= 60 seconds.
4. Public GitHub repo is accessible without login.
5. Zip code backup exists in `dist/`.
