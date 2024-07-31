# https://youtu.be/Y4-evQKobag

"""
Code generated using Claude 3.5 Sonnet on 28th July 2024

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
import czifile
from scipy import ndimage as ndi
from skimage import filters, segmentation, measure, feature
import matplotlib.pyplot as plt
import pandas as pd
from skimage import img_as_ubyte

# Load the .czi file
with czifile.CziFile('my_image.czi') as czi:
    image_array = czi.asarray()

# Extract channels (assuming dimensions are [1, 2, C, h, w, 1])
af568 = np.squeeze(image_array[0, 0, 0, :, :, 0])  # Red channel
af488 = np.squeeze(image_array[0, 0, 1, :, :, 0])  # Green channel
dapi = np.squeeze(image_array[0, 0, 2, :, :, 0])   # Blue channel (DAPI)

# Segment nuclei using DAPI channel
# Apply Gaussian filter to reduce noise
dapi_smooth = filters.gaussian(dapi, sigma=5)

# Use Otsu's method to find optimal threshold
thresh = filters.threshold_otsu(dapi_smooth)

# Create binary mask
binary = dapi_smooth > thresh

# Use watershed algorithm for segmentation
distance = ndi.distance_transform_edt(binary)
coords = feature.peak_local_max(distance, footprint=np.ones((9, 9)), labels=binary)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers = measure.label(mask)
segmented = segmentation.watershed(-distance, markers, mask=binary)


# Measure properties of segmented regions
props = measure.regionprops(segmented, intensity_image=dapi)

# Initialize lists to store results
nucleus_id = []
af568_mean = []
af488_mean = []
ratio = []

# Measure intensities and calculate ratios for each nucleus
for prop in props:
    nucleus_id.append(prop.label)
    mask = prop.image
    af568_intensity = np.mean(af568[prop.slice][mask])
    af488_intensity = np.mean(af488[prop.slice][mask])
    af568_mean.append(af568_intensity)
    af488_mean.append(af488_intensity)
    ratio.append(af568_intensity / af488_intensity if af488_intensity != 0 else np.nan)

# Create a DataFrame with the results
results = pd.DataFrame({
    'Nucleus_ID': nucleus_id,
    'AF568_Mean': af568_mean,
    'AF488_Mean': af488_mean,
    'Ratio_AF568_AF488': ratio
})

# Save results to CSV
results.to_csv('nucleus_intensities.csv', index=False)

# Convert segmented image to 8-bit grayscale
segmented_8bit = img_as_ubyte(segmented / np.max(segmented))

# Plot segmented nuclei
plt.figure(figsize=(10, 10))
plt.imshow(segmented_8bit, cmap='gray')
plt.title('Segmented Nuclei')
plt.colorbar(label='Nucleus ID')
plt.axis('off')
plt.tight_layout()

# Save the plot as before (this will be in color with a colorbar)
plt.savefig('segmented_nuclei_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Save the 8-bit grayscale image without any plot elements
plt.imsave('segmented_nuclei_8bit.png', segmented_8bit, cmap='gray')

print("Analysis complete. Results saved to 'nucleus_intensities.csv', 'segmented_nuclei_plot.png', and 'segmented_nuclei_8bit.tif'.")