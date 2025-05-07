import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import glob
from google.colab import drive

# Check if drive is already mounted
if not os.path.exists('/content/drive'):
    # Mount Google Drive if not already mounted
    drive.mount('/content/drive')
else:
    print("Google Drive is already mounted.")

# Input and output folder paths
# Shihab's folder
input_folder = "/content/drive/MyDrive/AJP images/Random/generated/ATM23_CR9_FR15_PS2"
output_folder = "/content/drive/MyDrive/AJP images/Random/generated/ATM23_CR9_FR15_PS2"

# ERIk can put the folder name below making my folder name as comment with #
# input_folder = "/content/drive/MyDrive/PSI Project/LTRIMAGES2"
# output_folder = "/content/drive/MyDrive/PSI Project/LTRIMAGES2"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all image file paths from the input folder
image_paths = glob.glob(os.path.join(input_folder, "*.png"))  # Use glob.glob() correctly

# Print the number of images found
print(f"Found {len(image_paths)} images.")

# Shihab's output folder
output_csv = "/content/drive/MyDrive/AJP images/Random/generated/ATM23_CR9_FR15_PS2/A23_C9_F15_P2_metrics.csv"

# ERIk's output folder (commented)
# output_csv = "/content/drive/MyDrive/PSI Project/LTRIMAGES/testmetrics2.csv"

pixel_to_micron = 100 / 205  # Conversion factor

# Main processing loop
image_files = [f for f in os.listdir(input_folder)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

metrics_list = []
for filename in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_folder, filename)
    metrics = process_image(image_path)
    if "Error" in metrics:
        print(f"Error processing {filename}: {metrics['Error']}")
    else:
        metrics_list.append(metrics)

# Save results to CSV
df = pd.DataFrame(metrics_list)
df.to_csv(output_csv, index=False)
print(f"Processing complete. Metrics saved to {output_csv}")