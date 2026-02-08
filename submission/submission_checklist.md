# Hackathon Submission Checklist

Deadline reference: **February 8, 2026, 9:00 AM ET**.

## Required artifacts
- [x] 150-300 word summary: `submission/project_summary_150_300_words.txt`
- [x] Demo video script (60s): `submission/demo_video_script_60s.md`
- [x] Tech video script (60s): `submission/tech_video_script_60s.md`
- [x] 1-page report markdown: `submission/one_pager.md`
- [x] 1-page PDF generated: `submission/World2Data_OnePager.pdf`
- [x] Pitfalls + limits memo: `submission/demo_pitfalls_and_limits.md`
- [x] Protocol execution plan: `submission/protocol_demo_plan.md`
- [ ] Public demo video link
- [ ] Public tech video link
- [ ] Public GitHub repository link
- [ ] Dataset link or `N/A`
- [x] Zip code backup in `dist/`

## Fast execution commands
```bash
# 1) Check SAM3 Hugging Face access
python scripts/check_sam3_hf_access.py

# 2) Build the one-page PDF
python submission/make_onepager_pdf.py

# 3) Create code zip + manifest
python scripts/prepare_hackathon_submission.py

# 4) Quick overnight artifact sanity
python - <<'PY'
import json
p='overnight_output/overnight_scene_graph.json'
d=json.load(open(p,'r',encoding='utf-8'))
print('frames=',d.get('num_frames'),'points=',d.get('num_points'))
PY
```

## Final portal submission targets
- Form: https://tinyurl.com/HN-4-submit
- Platform: https://projects.hack-nation.ai/
