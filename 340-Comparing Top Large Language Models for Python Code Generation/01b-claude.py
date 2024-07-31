# https://youtu.be/Y4-evQKobag

"""
Code generated using Claude 3.5 Sonnet on 28th July 2024

New prompt: Not a continuation of another prompt. New prompt in new chat. 

You are a biologist with microscopy and python coding skills. 
You are tasked with writing python code that loads a microscopy image with 
.czi extension (my_image.czi) containing three channels, AF568 (Red), AF488 (Green) 
and DAPI (Blue), respectively. 

Your goal is to segment individual nucleus using the third channel (DAPI) 
and measure the mean intensity of the first two channels, respectively, 
under each nucleus and calculate the ratio of these two intensities for each nucleus. 

Finally, plot the segmented nuclei image and save the mean intensities and ratios 
from each nuclei to a csv file. Note that the goal involves segmenting individual 
nucleus, not just the nuclei region via simple thresholding. Therefore I recommend 
using stardist library and working with 2D_versatile_fluo pre-trained model. 

Write the code for the above task and add appropriate comments for others to follow. 

Since the image is a .czi file, useczifile library to load the image. 
Note that the array dimensions for the loaded image would be (1, 2, C, h, w, 1).

------

Code worked fine initially but gave wild results as the image wasn't normalized

So provided a second prompt:
    Predicting on non-float input... ( forgot to normalize? )'

"""


import numpy as np
import pandas as pd
from czifile import CziFile
from stardist.models import StarDist2D
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from skimage import exposure

# Load the .czi file
def load_czi(file_path):
    with CziFile(file_path) as czi:
        image_array = czi.asarray()
    # Reshape the array to (C, h, w)
    image_array = np.squeeze(image_array)
    return image_array

# Load the image
image = load_czi('my_image.czi')

# Extract individual channels
red_channel = image[0]
green_channel = image[1]
blue_channel = image[2]  # DAPI channel for nucleus segmentation

# Normalize the DAPI channel
blue_channel_normalized = exposure.rescale_intensity(blue_channel, out_range=(0, 1))

# Initialize StarDist model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# Predict nuclei using the normalized DAPI channel
labels, _ = model.predict_instances(blue_channel_normalized)

# Calculate properties for each nucleus
props = regionprops(labels, intensity_image=np.stack([red_channel, green_channel], axis=-1))

# Initialize lists to store results
nucleus_ids = []
red_intensities = []
green_intensities = []
intensity_ratios = []

# Calculate intensities and ratios for each nucleus
for prop in props:
    nucleus_ids.append(prop.label)
    red_intensity = prop.intensity_mean[0]
    green_intensity = prop.intensity_mean[1]
    red_intensities.append(red_intensity)
    green_intensities.append(green_intensity)
    intensity_ratios.append(red_intensity / green_intensity if green_intensity != 0 else np.nan)

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'Nucleus_ID': nucleus_ids,
    'Red_Intensity': red_intensities,
    'Green_Intensity': green_intensities,
    'Red_Green_Ratio': intensity_ratios
})

# Save results to CSV
results_df.to_csv('nucleus_intensity_results.csv', index=False)

# Plot segmented nuclei
plt.figure(figsize=(10, 10))
plt.imshow(labels, cmap='nipy_spectral')
plt.title('Segmented Nuclei')
plt.colorbar(label='Nucleus ID')
plt.axis('off')
plt.savefig('segmented_nuclei.png', dpi=300, bbox_inches='tight')
plt.close()

print("Analysis complete. Results saved to 'nucleus_intensity_results.csv' and 'segmented_nuclei.png'.")