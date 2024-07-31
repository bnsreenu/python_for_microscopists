# https://youtu.be/Y4-evQKobag
"""
Code generated using chatGPT 4-o on 28th July 2024

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

--------

Code worked the first time, no need for the second prompt. 

"""


import czifile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import regionprops_table
from stardist.models import StarDist2D
from stardist import random_label_cmap
from csbdeep.utils import normalize

# Load the .czi file
filename = 'my_image.czi'
with czifile.CziFile(filename) as czi:
    image_array = czi.asarray()

# Extract the channels from the image
# The image array shape is (1, 2, C, h, w, 1)
dapi_channel = image_array[0, 0, 2, :, :, 0]    # DAPI (Blue)
af568_channel = image_array[0, 0, 0, :, :, 0]   # AF568 (Red)
af488_channel = image_array[0, 0, 1, :, :, 0]   # AF488 (Green)

# Normalize the DAPI channel for segmentation
dapi_norm = normalize(dapi_channel, 1, 99.8)

# Load the StarDist model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# Perform nucleus segmentation
labels, _ = model.predict_instances(dapi_norm)

# Measure mean intensities of AF568 and AF488 under each nucleus
props = regionprops_table(labels, intensity_image=af568_channel, properties=('label', 'mean_intensity'))
df_af568 = pd.DataFrame(props).rename(columns={'mean_intensity': 'mean_AF568'})

props = regionprops_table(labels, intensity_image=af488_channel, properties=('label', 'mean_intensity'))
df_af488 = pd.DataFrame(props).rename(columns={'mean_intensity': 'mean_AF488'})

# Merge the dataframes on 'label'
df = pd.merge(df_af568, df_af488, on='label')

# Calculate the ratio of AF568 to AF488 intensities
df['ratio_AF568_AF488'] = df['mean_AF568'] / df['mean_AF488']

# Save the results to a CSV file
df.to_csv('nucleus_intensities_and_ratios.csv', index=False)

# Plot the segmented nuclei
plt.figure(figsize=(10, 10))
plt.imshow(dapi_channel, cmap='gray')
plt.imshow(labels, cmap=random_label_cmap(), alpha=0.5)
plt.title('Segmented Nuclei')
plt.axis('off')
plt.show()
