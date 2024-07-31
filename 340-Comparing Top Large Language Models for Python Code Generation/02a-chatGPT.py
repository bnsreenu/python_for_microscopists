# https://youtu.be/Y4-evQKobag
"""
Code generated using chatGPT 4-o on 28th July 2024

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
import matplotlib.pyplot as plt
import pandas as pd
from skimage import filters, measure, morphology, segmentation, color
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import czifile

# Load the .czi file
czi_file = czifile.CziFile('my_image.czi')
image_data = czi_file.asarray()

# The image is expected to have the shape (1, 2, C, h, w, 1)
# Extract the respective channels assuming C=3 (DAPI, AF488, AF568)
dapi_channel = image_data[0, 0, 2, :, :, 0]  # Blue channel
af568_channel = image_data[0, 0, 0, :, :, 0] # Red channel
af488_channel = image_data[0, 0, 1, :, :, 0] # Green channel

# Segment individual nuclei using the DAPI channel
# First, apply a Gaussian filter to smooth the image
dapi_smoothed = filters.gaussian(dapi_channel, sigma=2)

# Apply a threshold to create a binary image
threshold = filters.threshold_otsu(dapi_smoothed)
binary_dapi = dapi_smoothed > threshold

# Remove small objects and fill holes to get more accurate nuclei
binary_dapi = morphology.remove_small_objects(binary_dapi, min_size=500)
binary_dapi = ndi.binary_fill_holes(binary_dapi)

# Compute the distance transform
distance = ndi.distance_transform_edt(binary_dapi)

# Find local maxima
local_maxi = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary_dapi)

# Perform watershed segmentation
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=binary_dapi)

# Measure properties of each labeled region
properties = measure.regionprops(labels, intensity_image=dapi_channel)

# Prepare lists to store the results
mean_intensity_af568 = []
mean_intensity_af488 = []
ratios = []

# Loop through each region and calculate the mean intensities and ratio
for prop in properties:
    coords = prop.coords
    mean_af568 = np.mean(af568_channel[coords[:, 0], coords[:, 1]])
    mean_af488 = np.mean(af488_channel[coords[:, 0], coords[:, 1]])
    ratio = mean_af568 / mean_af488 if mean_af488 != 0 else 0

    mean_intensity_af568.append(mean_af568)
    mean_intensity_af488.append(mean_af488)
    ratios.append(ratio)

# Save the results to a CSV file
results = pd.DataFrame({
    'Mean Intensity AF568': mean_intensity_af568,
    'Mean Intensity AF488': mean_intensity_af488,
    'Ratio AF568/AF488': ratios
})
results.to_csv('nuclei_intensity_ratios.csv', index=False)

# Plot the segmented nuclei image
plt.figure(figsize=(10, 10))
plt.imshow(color.label2rgb(labels, image=dapi_channel, bg_label=0))
plt.title('Segmented Nuclei')
plt.axis('off')
plt.show()
