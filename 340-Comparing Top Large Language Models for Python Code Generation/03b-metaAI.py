# https://youtu.be/Y4-evQKobag

"""
Code generated using Meta AI, Llama 3.1 405-B on 28th July 2024

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

---------------
Code worked fine initially but gave wild results as the image wasn't normalized

So provided a second prompt:
    Predicting on non-float input... ( forgot to normalize? )'

"""



# Nucleus Segmentation and Intensity Measurement

import czifile
import numpy as np
from stardist.models import StarDist2D
from skimage import measure
import pandas as pd
import matplotlib.pyplot as plt

# Load the microscopy image
img = czifile.imread('my_image.czi')

# Get the image dimensions
print(img.shape)  # Output: (1, 2, 3, h, w, 1)

# Extract the DAPI channel (Blue)
dapi_channel = img[0, 0, 2, :, :, 0]


# Normalize the DAPI channel
dapi_channel_normalized = dapi_channel / np.max(dapi_channel)



# Extract the AF568 (Red) and AF488 (Green) channels
af568_channel = img[0, 0, 0, :, :, 0]
af488_channel = img[0, 0, 1, :, :, 0]


# Initialize the StarDist model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# Segment the nuclei
nuclei_labels, _ = model.predict_instances(dapi_channel_normalized)


# Get the number of nuclei
num_nuclei = np.max(nuclei_labels)
print(f"Number of nuclei: {num_nuclei}")

# Initialize lists to store mean intensities and ratios
mean_intensities_af568 = []
mean_intensities_af488 = []
ratios = []

# Loop through each nucleus
for i in range(1, num_nuclei + 1):
    # Get the nucleus mask
    nucleus_mask = nuclei_labels == i
    
    # Calculate the mean intensity of AF568 (Red) and AF488 (Green) channels
    mean_af568 = np.mean(af568_channel[nucleus_mask])
    mean_af488 = np.mean(af488_channel[nucleus_mask])
    
    # Calculate the ratio of mean intensities
    ratio = mean_af568 / mean_af488
    
    # Append to the lists
    mean_intensities_af568.append(mean_af568)
    mean_intensities_af488.append(mean_af488)
    ratios.append(ratio)

# Create a Pandas DataFrame
df = pd.DataFrame({
    'Nucleus ID': range(1, num_nuclei + 1),
    'Mean Intensity AF568': mean_intensities_af568,
    'Mean Intensity AF488': mean_intensities_af488,
    'Ratio': ratios
})

# Save to CSV file
df.to_csv('nuclei_intensities.csv', index=False)

# Plot the segmented nuclei image
plt.imshow(nuclei_labels, cmap='jet')
plt.title('Segmented Nuclei')
plt.show()