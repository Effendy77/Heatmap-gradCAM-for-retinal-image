# Retinal Grad-CAM Visualisation for MACE Prediction

This repository implements a visual explanation pipeline using **Gradient-weighted Class Activation Mapping (Grad-CAM)** on retinal fundus images for the prediction of Major Adverse Cardiovascular Events (MACE) using deep learning (DL) models.

It provides a modular and reproducible framework compatible with **RETFound-based ViT architectures** and uses the open-source [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) library.

---

## 📁 Repository Structure

```
.
├── step_1_preprocess_input.py      # Preprocess retinal images: crop, mask, resize
├── step_2_run_gradcam.py           # Run Grad-CAM with user-specified model + method
├── step_3_analyse_output.py        # Save and visualize Grad-CAM overlays
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Requirements

- Python 3.7+
- OpenCV
- NumPy
- Pillow
- Matplotlib
- pandas
- timm
- pytorch-grad-cam

```bash
pip install -r requirements.txt
```

---

## 🧪 Grad-CAM Highlights

- Compatible with **Vision Transformers (ViT)**, including RETFound
- Supports multiple CAM variants: `gradcam`, `gradcam++`, `layercam`, `eigencam`
- Custom ViT reshaping logic for attention layers
- Optionally enable smoothing:
  ```bash
  --aug_smooth --eigen_smooth
  ```

---

## 🔍 Output

Each sample generates:
- Original image (224x224)
- Grad-CAM heatmap
- Overlay image
- All saved under `/analysis_output/`

---

## 📖 Related Publication

> Effendy Bin Hashim et al., *“Retinal Biomarker-Based Deep Learning for Predicting 10-Year Cardiovascular Risk in Diabetes: A Comparative Evaluation with QRISK3”*, The Lancet Digital Health, 2025.

---

## License

MIT License
