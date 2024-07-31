# https://youtu.be/Y4-evQKobag

"""
Code generated using Google Gemini (1.5 Flash) on 28th July 2024

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
import pandas as pd
import czifile
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from scipy import ndimage
from skimage.feature import peak_local_max


# Function to load CZI image and extract channels
def load_czi_image(file_path):
    """Loads a CZI image and extracts the three specified channels.

    Args:
        file_path (str): Path to the CZI image file.

    Returns:
        tuple: A tuple containing three numpy arrays representing the Red, Green, and Blue channels.
    """
    with czifile.CziFile(file_path) as czi:
        array = czi.asarray()

    # Extract the desired scale, series, and channels
    red_channel = array[0, 0, 0, :, :]  # Assuming first scale, first series, first channel
    green_channel = array[0, 0, 1, :, :]
    blue_channel = array[0, 0, 2, :, :]

    return red_channel, green_channel, blue_channel


# Function to segment nuclei using watershed algorithm
def segment_nuclei(dapi_channel):
    """Segments individual nuclei in the DAPI channel using the watershed algorithm.

    Args:
        dapi_channel (numpy.ndarray): The DAPI channel image.

    Returns:
        numpy.ndarray: A labeled image where each nucleus is assigned a unique label.
    """
    # Thresholding to create a binary mask
    thresh = threshold_otsu(dapi_channel)
    binary_mask = dapi_channel > thresh

    # Distance transform for markers
    distance_transform = ndimage.distance_transform_edt(binary_mask)
    local_maxima = peak_local_max(distance_transform, min_distance=20, labels=binary_mask)

    markers = ndimage.label(local_maxima)[0]

    # Watershed segmentation
    labels = watershed(-distance_transform, markers, mask=binary_mask)
    return labels

# Function to calculate mean intensities and ratios
def calculate_intensities_and_ratios(red_channel, green_channel, labels):
    """Calculates mean intensities of the Red and Green channels for each segmented nucleus and computes the ratio.

    Args:
        red_channel (numpy.ndarray): The Red channel image.
        green_channel (numpy.ndarray): The Green channel image.
        labels (numpy.ndarray): The labeled image of segmented nuclei.

    Returns:
        pandas.DataFrame: A DataFrame containing the mean Red intensity, mean Green intensity, and ratio for each nucleus.
    """
    num_nuclei = labels.max()
    mean_red_intensities = []
    mean_green_intensities = []
    ratios = []

    for i in range(1, num_nuclei + 1):
        mask = labels == i
        red_mean = np.mean(red_channel[mask])
        green_mean = np.mean(green_channel[mask])
        ratio = red_mean / green_mean
        mean_red_intensities.append(red_mean)
        mean_green_intensities.append(green_mean)
        ratios.append(ratio)

    data = {'Mean Red Intensity': mean_red_intensities, 'Mean Green Intensity': mean_green_intensities, 'Ratio': ratios}
    df = pd.DataFrame(data)
    return df

# Main execution
if __name__ == "__main__":
    file_path = 'my_image.czi'
    red, green, blue = load_czi_image(file_path)
    labels = segment_nuclei(blue)
    results = calculate_intensities_and_ratios(red, green, labels)
    results.to_csv('nucleus_data.csv', index=False)

    # Visualization
    plt.imshow(labels, cmap='nipy_spectral')
    plt.title('Segmented Nuclei')
    plt.show()
