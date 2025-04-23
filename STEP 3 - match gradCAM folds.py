import os
import pandas as pd

# CONFIG
TARGET_CSV = r"D:/EXPERIMENT/gradCAM/T1D/gradcam_targets.csv"
FOLD_ROOT = r"D:/PREVIOUS_TRAINING/RETFound_MAE41ver4-CLAHE_T1diabetes/k-fold-splits"
OUTPUT_CSV = r"D:/EXPERIMENT/gradCAM/T1D/gradcam_targets_with_folds.csv"

# Load the Grad-CAM targets
df = pd.read_csv(TARGET_CSV)
df['fold_number'] = None  # to store detected fold

# Get list of folds
fold_dirs = [f for f in os.listdir(FOLD_ROOT) if f.startswith("fold_")]

# Match image to fold
for fold in fold_dirs:
    fold_val_path = os.path.join(FOLD_ROOT, fold, "val")
    for root, _, files in os.walk(fold_val_path):
        file_set = set(files)
        for idx, row in df.iterrows():
            if row['filename'] in file_set:
                df.at[idx, 'fold_number'] = fold
                print(f"{row['filename']} → {fold}")

# Save output
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Updated Grad-CAM targets with fold info saved to:\n{OUTPUT_CSV}")
