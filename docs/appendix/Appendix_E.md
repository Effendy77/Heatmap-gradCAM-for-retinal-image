# Appendix E. HYBRID Grad-CAM Experiment — Final Reproducible Workflow

**Continuity with Supplementary Doc S7.** This appendix operationalizes a scripted, cohort-stratified pipeline (primary & secondary) with threshold-aware selection and provenance for exact reproducibility.

## Workflow
erify_inputs.py → select_cases.py → generate_gradcam.py → compose_panels.py → provenance_log.py

- **verify_inputs.py** — checks files/columns and left-eye pattern.
- **select_cases.py** — creates strata; labels TP/FP/FN/TN at chosen threshold; picks top-k closest to threshold.
- **generate_gradcam.py** — loads checkpoint; applies Grad-CAM at target layer; saves overlays per cohort/stratum/outcome.
- **compose_panels.py** — tiles images into labeled panels.
- **provenance_log.py** — records config + SHA-256 hashes of checkpoints, predictions, and panels.

## Inputs & Columns
Predictions CSV must contain: eid, a filename-like column, ground truth (mace or label), and the probability column (default DL_raw_prob). Left-eye filenames expected (e.g., 21015).
Strata: **Primary** = sex × age_bin; **Secondary** = sex × age_bin × diabetes_prevalent × hypertension_prevalent.
Age bins auto-computed if missing.

## Exact Commands
`ash
python verify_inputs.py --config configs/hybrid.yaml
python select_cases.py --config configs/hybrid.yaml --out selections/
python generate_gradcam.py --config configs/hybrid.yaml --selections selections/ --out cams/
python compose_panels.py --root cams/ --out panels/ --cols 6
python provenance_log.py --config configs/hybrid.yaml --panels_dir panels/ --out provenance.json
Environment

torch≥2.0.0, torchvision≥0.15.1, timm≥0.9.16, pytorch-grad-cam≥1.4.8, opencv-python-headless≥4.10, pandas≥2.2, numpy≥1.26.

Colour Map & Artefacts

Red = high attention; Blue = low. A vertical colour bar (0–1) is rendered alongside each Grad-CAM (visuals.add_colorbar: true). Borders/background outside the retinal area may appear saturated due to overlay blending and are non-informative; enabling preprocess.enable: true can reduce this.

Selection & Reporting

Default topk_per_outcome = 12 per stratum. State and justify the decision threshold (e.g., Youden 0.345 vs 0.5). Cite the code artifact (e.g., branch hybrid-final, tag v1.0-hybrid).
