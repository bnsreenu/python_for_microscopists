# https://youtu.be/Y4-evQKobag
"""
Code generated using Microsoft Copilot on 28th July 2024

First prompt:

You are a biologist with microscopy and python coding skills. 
You are tasked with writing python code that loads a microscopy image with 
.czi extension (my_image.czi) containing three channels, AF568 (Red), AF488 (Green) 
and DAPI (Blue), respectively. 

Your goal is to segment individual nucleus using the third channel (DAPI) 
and measure the mean intensity of the first two channels, respectively, 
under each nucleus and calculate the ratio of these two intensities for each nucleus. 

Finally, plot the segmented nuclei image and save the mean intensities and ratios 
from each nuclei to a csv file. Note that the goal involves segmenting individual 
nucleus, not just the nuclei region via simple thresholding. 

Write the code for the above task and add appropriate comments for others to follow. 


Second prompt: Another prompt in the same chat after the initial code gave errors. 

Since the image is a .czi file, useczifile library to load the image. 
Note that the array dimensions for the loaded image would be (1, 2, C, h, w, 1).


"""


import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Load the microscopy image (my_image.czi)
image_path = "my_image.czi"
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Extract individual channels (AF568, AF488, DAPI)
af568_channel = image[:, :, 0]  # Red channel
af488_channel = image[:, :, 1]  # Green channel
dapi_channel = image[:, :, 2]   # Blue (DAPI) channel

# Threshold DAPI channel to segment nuclei
_, binary_nuclei = cv2.threshold(dapi_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours of segmented nuclei
contours, _ = cv2.findContours(binary_nuclei, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize lists to store mean intensities and ratios
mean_intensities_af568 = []
mean_intensities_af488 = []
ratios = []

for contour in contours:
    # Get mask for each nucleus
    mask = np.zeros_like(dapi_channel)
    cv2.drawContours(mask, [contour], 0, 255, -1)

    # Calculate mean intensity for AF568 and AF488 channels
    mean_af568 = np.mean(af568_channel[mask > 0])
    mean_af488 = np.mean(af488_channel[mask > 0])

    # Calculate ratio of AF568 intensity to AF488 intensity
    ratio = mean_af568 / mean_af488

    # Append to lists
    mean_intensities_af568.append(mean_af568)
    mean_intensities_af488.append(mean_af488)
    ratios.append(ratio)

# Create a DataFrame to store results
results_df = pd.DataFrame({
    "MeanIntensity_AF568": mean_intensities_af568,
    "MeanIntensity_AF488": mean_intensities_af488,
    "Ratio_AF568_AF488": ratios
})

# Save results to a CSV file
results_df.to_csv("nuclei_intensity_ratios.csv", index=False)

# Plot the segmented nuclei
plt.imshow(binary_nuclei, cmap="gray")
plt.title("Segmented Nuclei")
plt.axis("off")
plt.show()
