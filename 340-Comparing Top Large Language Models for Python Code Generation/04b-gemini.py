# https://youtu.be/Y4-evQKobag
"""

Code generated using Google Gemini (1.5 Flash) on 28th July 2024

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

----

No point in even running the code. The language model doesn't seem to know how
stardist model gets imported and used to segment nuclei. 

"""



import czifile
import numpy as np
from skimage.io import imshow
from stardist import  models
from stardist.data import Patches
from stardist.utils import calculate_metric_map
from scipy.ndimage import label
import pandas as pd

# Load the CZI image
with czifile.CziFile('my_image.czi') as czi:
    img = czi.asarray()

# Extract the relevant channels: AF568 (Red), AF488 (Green), and DAPI (Blue)
red_channel = img[0, 0, 0, :, :]
green_channel = img[0, 0, 1, :, :]
dapi_channel = img[0, 0, 2, :, :]

# Load the pre-trained StarDist model
model = models.create_model('2d_versatile_fluo')
model.load_weights('path/to/your/pretrained/model.h5')

# Preprocess the DAPI channel for StarDist
X = np.expand_dims(dapi_channel, axis=0)
X = np.expand_dims(X, axis=0)  # Add channel dimension

# Predict object masks using StarDist
probs, dist_maps = model.predict(X)

# Threshold probability maps to obtain object masks
labels, _ = label(probs[0, 0] > 0.5)

# Measure mean intensities for each nucleus
mean_red_intensities = []
mean_green_intensities = []
ratios = []

for label_id in range(1, labels.max() + 1):
    mask = labels == label_id
    mean_red_intensity = np.mean(red_channel[mask])
    mean_green_intensity = np.mean(green_channel[mask])
    ratio = mean_red_intensity / mean_green_intensity
    mean_red_intensities.append(mean_red_intensity)
    mean_green_intensities.append(mean_green_intensity)
    ratios.append(ratio)

# Create a DataFrame to store results
results_df = pd.DataFrame({
    'Mean Red Intensity': mean_red_intensities,
    'Mean Green Intensity': mean_green_intensities,
    'Ratio': ratios
})

# Save results to CSV
results_df.to_csv('nucleus_measurements.csv', index=False)

# Visualize segmented nuclei
imshow(labels)
