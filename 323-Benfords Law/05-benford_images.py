"""
Verification of Benford's law in image data.

This code analyzes the leading digit distribution of Discrete Cosine Transform (DCT) 
coefficients in grayscale images. It is about verifying the Benford's Law to highlight 
the frequencies of leading digits (1 through 9) in the pixel values derived 
from DCT. 

The code reads images in grayscale using opencv library, computes the DCT coefficients, 
and plots the observed Benford's Law distribution for each image. 
Then we plot the data to visualize and compare the occurrence of leading digits 
in different images. 

In case you wonder why go through the pain of converting pixel values to DCT...

In the context of Benford's Law, the distribution of leading digits is expected 
to follow a logarithmic pattern, where smaller digits (1, 2, 3) occur more 
frequently than larger digits (4, 5, 6, 7, 8, 9). 
When pixel values are confined to a small range, it can disrupt this natural 
logarithmic distribution. For example, in 8 bit images, our pixels have values between
0 to 255. So any bright pixel will always have a leading digit of 2 and never
have values 3 or greater. 

"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def leading_digit(pixel_value):
    """
    Helper function to extract the leading digit of a pixel value.
    """
    while pixel_value >= 10:
        pixel_value //= 10
    return pixel_value

def calculate_observed_counts(coefficients):
    """
    Calculate the observed leading digit frequencies from the given coefficients.
    """
    leading_digits = [leading_digit(abs(coeff)) for coeff in coefficients]
    observed_counts = [leading_digits.count(digit) for digit in range(1, 10)]
    return observed_counts

def plot_observed_benford_law(image_paths, labels):
    """
    Plot the observed Benford's Law distribution for a list of images.

    Parameters:
    - image_paths (list): List of paths to image files.
    - labels (list): List of labels for each image.

    """
    plt.figure(figsize=(10, 6))

    # Define custom colors for observed lines
    observed_colors = ['darkred', 'darkgreen', 'darkblue']

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        dct_coefficients = cv2.dct(np.float32(image))
        selected_coefficients = dct_coefficients.flatten()

        observed_counts = calculate_observed_counts(selected_coefficients)

        sns.lineplot(x=range(1, 10), y=observed_counts, label=f'{labels[i]} - Observed', color=observed_colors[i])

    plt.xlabel('Leading Digit')
    plt.ylabel('Frequency')
    plt.title('Leading Digit Distribution (Observed)')
    plt.legend()
    plt.show()

# Example usage:
image_paths = ['data/sat_img1.jpg', 'data/sat_img2.jpg', 'data/sat_img3.jpg']
labels = ['Image 1', 'Image 2', 'Image 3']
plot_observed_benford_law(image_paths, labels)
