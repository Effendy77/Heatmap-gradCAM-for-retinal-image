import os
import pandas as pd

# -------------------- CONFIG --------------------
VAL_DIR = r"D:/PREVIOUS_TRAINING/RETFound_MAE41ver4-CLAHE_T1diabetes/k-fold-splits/fold_5/val"
PREDICTION_CSV = r"D:/EXPERIMENT/gradCAM/T1D/predictions_fold5.csv"
OUTPUT_CSV = r"D:/EXPERIMENT/gradCAM/T1D/matched_predictions_fold5.csv"

# -------------------- SCRIPT --------------------
# Step 1: Load all image filenames from validation set (cv_event and non_cv_event)
all_images = []
for root, _, files in os.walk(VAL_DIR):
    for file in files:
        if file.lower().endswith(".png"):
            all_images.append(file)

# Sort filenames to match prediction order (ensure consistent order used during inference)
all_images = sorted(all_images)

# Step 2: Load predictions CSV
pred_df = pd.read_csv(PREDICTION_CSV)

# Step 3: Trim to match if needed
min_len = min(len(all_images), len(pred_df))
pred_df = pred_df.iloc[:min_len].copy()
pred_df['filename'] = all_images[:min_len]

# Step 4: Save merged output
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
pred_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved matched predictions: {OUTPUT_CSV}")
