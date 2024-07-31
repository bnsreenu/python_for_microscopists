# https://youtu.be/Y4-evQKobag

"""
Code generated using Meta AI, Llama 3.1 405-B on 28th July 2024

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

import czifile
import numpy as np
from skimage import filters, morphology, measure, segmentation
import matplotlib.pyplot as plt
import pandas as pd

# Load the CZI image
img = czifile.imread('my_image.czi')

# Get the DAPI channel (third channel)
dapi_channel = img[0, 0, 2, :, :, 0]

# Get the AF568 and AF488 channels (first two channels)
af568_channel = img[0, 0, 0, :, :, 0]
af488_channel = img[0, 0, 1, :, :, 0]

# Apply Gaussian filter to smooth the DAPI channel
dapi_smooth = filters.gaussian(dapi_channel, sigma=1)

# Find the local maxima (nuclei centers)
maxima = morphology.local_maxima(dapi_smooth)

# Apply watershed transform to segment individual nuclei
markers = measure.label(maxima)
nuclei_mask = segmentation.watershed(dapi_smooth, markers)

# Remove small objects (noise)
nuclei_mask = morphology.remove_small_objects(nuclei_mask, min_size=100)

# Create a list to store the results
results = []

# Iterate over each nucleus
for nucleus in measure.regionprops(nuclei_mask):
    # Get the mean intensities of AF568 and AF488 channels
    af568_mean = np.mean(af568_channel[nucleus.coords[:, 0], nucleus.coords[:, 1]])
    af488_mean = np.mean(af488_channel[nucleus.coords[:, 0], nucleus.coords[:, 1]])

    # Calculate the ratio
    ratio = af568_mean / af488_mean

    # Append the results to the list
    results.append([nucleus.label, af568_mean, af488_mean, ratio])

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the DAPI channel
ax.imshow(dapi_channel, cmap='gray')

# Plot the segmented nuclei
ax.imshow(nuclei_mask, cmap='jet', alpha=0.5)

# Show the plot
plt.show()

# Create a Pandas DataFrame from the results
df = pd.DataFrame(results, columns=['Nucleus ID', 'AF568 Mean', 'AF488 Mean', 'Ratio'])

# Save the DataFrame to a CSV file
df.to_csv('nuclei_intensities.csv', index=False)