# Appendix E. HYBRID Grad-CAM Experiment — Final Reproducible Workflow

**Continuity with Supplementary Doc S7.** This appendix operationalizes a scripted, cohort-stratified pipeline (primary & secondary) with threshold-aware selection and provenance for exact reproducibility.

## Workflow
`verify_inputs.py -> select_cases.py -> generate_gradcam.py -> compose_panels.py -> provenance_log.py`

- **verify_inputs.py** — checks files/columns and left-eye pattern.
- **select_cases.py** — creates strata; labels TP/FP/FN/TN at chosen threshold; picks top-k closest to threshold.
- **generate_gradcam.py** — loads checkpoint; applies Grad-CAM at target layer; saves overlays per cohort/stratum/outcome.
- **compose_panels.py** — tiles images into labeled panels.
- **provenance_log.py** — records config + SHA-256 hashes of checkpoints, predictions, and panels.

## Inputs & Columns
Predictions CSV must contain: `eid`, a filename-like column, ground truth (`mace` or `label`), and the probability column (default `DL_raw_prob`). Left-eye filenames expected (e.g., `21015`).
Strata: **Primary** = `sex × age_bin`; **Secondary** = `sex × age_bin × diabetes_prevalent × hypertension_prevalent`.
Age bins auto-computed if missing.

## Exact Commands
```bash
python verify_inputs.py --config configs/hybrid.yaml
python select_cases.py --config configs/hybrid.yaml --out selections/
python generate_gradcam.py --config configs/hybrid.yaml --selections selections/ --out cams/
python compose_panels.py --root cams/ --out panels/ --cols 6
python provenance_log.py --config configs/hybrid.yaml --panels_dir panels/ --out provenance.json

Environment

torch>=2.0.0, torchvision>=0.15.1, timm>=0.9.16, pytorch-grad-cam>=1.4.8, opencv-python-headless>=4.10, pandas>=2.2, numpy>=1.26.

Colour Map & Artefacts

Red = high attention; Blue = low. A vertical colour bar (0–1) is rendered alongside each Grad-CAM (visuals.add_colorbar: true). Borders/background outside the retinal area may appear saturated due to overlay blending and are non-informative; enabling preprocess.enable: true can reduce this.

Selection & Reporting

Default topk_per_outcome = 12 per stratum. State and justify the decision threshold (e.g., Youden 0.345 vs 0.5). Cite the code artifact (e.g., branch hybrid-final, tag v1.0-hybrid).
```

(That fixes the code-fence so “Environment / Colour Map / Selection” render correctly.)

# 2) Add the config file in the repo UI
1. In the **`hybrid-final`** branch, click **Add file → Create new file**.  
2. Name it: `configs/hybrid.yaml`.  
3. Paste this starter config (fill the `/path/to/...` later if you want; you can commit now):


```yaml
experiment_name: HYBRID-GradCAM-AppendixE
device: cuda

backbone: vit
target_layer: blocks.-1.norm1
image_size: 224

normalize_mean: [0.485, 0.456, 0.406]
normalize_std: [0.229, 0.224, 0.225]

probability_column: DL_raw_prob
left_eye_pattern: "21015"

threshold: 0.345
topk_per_outcome: 12

preprocess:
  enable: false

visuals:
  add_colorbar: true

images_dir:
  primary: /path/to/left_images_primary
  secondary: /path/to/left_images_secondary

checkpoints:
  primary: /path/to/fold0_primary_checkpoint.pth
  secondary: /path/to/fold0_secondary_checkpoint.pth

predictions_csv:
  primary: /path/to/primary_predictions.csv
  secondary: /path/to/secondary_predictions.csv

stratifications:
  primary:
    - by: [sex, age_bin]
      bins:
        - {label: lt50, min: 0, max: 50}
        - {label: 50to60, min: 50, max: 60}
        - {label: ge60, min: 60, max: 120}
  secondary:
    - by: [sex, age_bin, diabetes_prevalent, hypertension_prevalent]
      bins:
        - {label: lt50, min: 0, max: 50}
        - {label: 50to60, min: 50, max: 60}
        - {label: ge60, min: 60, max: 120}
```

# 3) (Optional) Add the flowchart image
- In `docs/appendix/`, click **Add file → Upload files**, and upload `AppendixE_Flowchart.png`.  
- Then add this line somewhere in `Appendix_E.md`:
  ```
  ![Grad-CAM pipeline](AppendixE_Flowchart.png)
  ```

# 4) Tidy the old scripts (optional but recommended)
To avoid confusing reviewers, move the older 3-step scripts into a folder named `legacy/` right in the GitHub UI:
- Open each file (`step_1_preprocess_input.py`, `step_2_run_gradcam.py`, `step_3_analyse_output.py`) → ✏️ edit → above the filename, change the path to `legacy/step_1_preprocess_input.py` (etc.) → **Commit changes**.  
This keeps history, but clearly separates the old approach.

# 5) Cut a citable release (UI only)
- Go to **Releases** → **Draft a new release**.  
- **Choose a tag:** `v1.0-hybrid` (create new).  
- **Target:** `hybrid-final`.  
- Title: “HYBRID Grad-CAM v1.0 (Appendix E)”.  
- Notes: “Final reproducible Grad-CAM workflow and docs.”  
- (Optional) Attach `docs/appendix/Appendix_E.md` and `AppendixE_Flowchart.png`.  
- **Publish release.**

---

If you do just #1 and #2, your Appendix will render perfectly and the repo will be reproducible from the UI alone. Want me to review the `hybrid_gradcam/gradcam_utils.py` on the branch and confirm the **colour bar code** is present? I can check and point to the exact lines next.
