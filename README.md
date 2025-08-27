# HYBRID Grad-CAM (Primary & Secondary)

A complete Grad-CAM pipeline for your HYBRID experiment with two cohorts:
- Primary prevention: stratify by sex + age bins
- Secondary prevention: stratify by sex + age bins + diabetes_prevalent + hypertension_prevalent

Supports per-cohort image roots, uses Fold 0 checkpoints (single model per cohort), and produces TP/FP/FN/TN panels per stratum.

## Quickstart (Windows, GPU)
```powershell
cd <unzipped>\gradcam-hybrid
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# If you need a CUDA-specific PyTorch wheel, follow the official install instructions for your setup.
