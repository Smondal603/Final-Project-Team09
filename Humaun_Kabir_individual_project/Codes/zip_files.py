import os
import cv2
import numpy as np
import pandas as pd
import zipfile
from PIL import Image
from tqdm import tqdm
from google.colab import drive

# Mount Google Drive if needed
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')
else:
    print("Google Drive is already mounted.")

# Paths setup
zip_path = '/content/drive/MyDrive/AJP images/All images after augmentation_23k.zip'
output_csv = '/content/drive/MyDrive/AJP images/image_metrics.csv'
temp_extract_dir = '/content/temp_extract'

# Create temporary directory
os.makedirs(temp_extract_dir, exist_ok=True)

# Pixel to micron conversion
pixel_to_micron = 100 / 205  # Conversion factor

# Process images directly from zip while preserving original names
metrics_list = []
with zipfile.ZipFile(zip_path) as z:
    # Get all image files
    image_files = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for file in tqdm(image_files, desc="Processing images"):
        try:
            # Extract to temporary location
            temp_path = z.extract(file, temp_extract_dir)

            # Process with original name
            metrics = process_image(temp_path, os.path.basename(file))
            metrics_list.append(metrics)

            # Remove temporary file
            os.remove(temp_path)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

# Save results to CSV
df = pd.DataFrame(metrics_list)
df.to_csv(output_csv, index=False)
print(f"Processing complete. Metrics saved to {output_csv}")

# Clean up temp directory
if os.path.exists(temp_extract_dir):
    for root, dirs, files in os.walk(temp_extract_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(temp_extract_dir)