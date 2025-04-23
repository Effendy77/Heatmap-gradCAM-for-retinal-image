import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------- CONFIGURE BELOW --------
CSV_PATH = r"D:/EXPERIMENT/gradCAM/T1D/matched_predictions_fold3.csv"  # Update for each subgroup
OUT_PATH = r"D:/EXPERIMENT/gradCAM/T1D/gradcam_targets.csv"            # Output file
PROB_THRESHOLD = 0.5                                                   # Classification threshold
# ----------------------------------

# Check if input file exists
if not os.path.exists(CSV_PATH):
    logging.error(f"Input file not found: {CSV_PATH}")
    raise FileNotFoundError(f"Input file not found: {CSV_PATH}")

# Load matched prediction file
try:
    df = pd.read_csv(CSV_PATH)
    logging.info(f"Loaded CSV file: {CSV_PATH}")
except Exception as e:
    logging.error(f"Error loading CSV file: {e}")
    raise

# Validate required columns
required_columns = ['filename', 'Predicted_Probability', 'True_Label']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    logging.error(f"Missing required columns in the input file: {missing_columns}")
    raise ValueError(f"Missing required columns: {missing_columns}")

# Filter for original left-eye images only (exclude flipped)
df = df[df['filename'].str.contains("21015") & ~df['filename'].str.contains("flipped")].copy()

# Apply prediction threshold
df['Predicted_Label'] = (df['Predicted_Probability'] >= PROB_THRESHOLD).astype(int)

# Classify into TP, TN, FP, FN
def classify(row):
    if row['True_Label'] == 1 and row['Predicted_Label'] == 1:
        return "TP"
    elif row['True_Label'] == 0 and row['Predicted_Label'] == 0:
        return "TN"
    elif row['True_Label'] == 1 and row['Predicted_Label'] == 0:
        return "FN"
    elif row['True_Label'] == 0 and row['Predicted_Label'] == 1:
        return "FP"
    return "Unknown"

df['Category'] = df.apply(classify, axis=1)

# Extract top 1 from each category
top_cases = df.sort_values(by='Predicted_Probability', ascending=False).groupby('Category').head(1)

# Save to CSV
try:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    top_cases.to_csv(OUT_PATH, index=False)
    logging.info(f"✅ Extracted Grad-CAM targets saved to: {OUT_PATH}")
except Exception as e:
    logging.error(f"Error saving output file: {e}")
    raise
