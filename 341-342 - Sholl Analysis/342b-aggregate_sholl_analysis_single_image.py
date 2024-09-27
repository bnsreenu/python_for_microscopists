# https://youtu.be/SB9hVcYIoXk
"""
Author: Dr. Sreenivas Bhattiprolu
Date: September 2024
Version: 1.1

Performs Sholl analysis on individual soma in a single image showing multiple soma and combines them. 

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.draw import disk

def get_multiple_soma_centers(image):
    """
    Allow manual selection of multiple soma centers via mouse clicks with improved visibility.
    Soma centers are identified by clicking on the image, and they are marked with a visible circle and label.
    """
    soma_coords = []  # List to store coordinates of soma centers
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add the clicked coordinates to the list
            soma_coords.append((x, y))
            # Draw a larger, more visible circle on the image at the clicked position
            cv2.circle(image_copy, (x, y), 10, (0, 255, 0), 2)
            # Add a label next to the circle indicating the order of clicks
            label = str(len(soma_coords))
            cv2.putText(image_copy, label, (x+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow(window_name, image_copy)

    # Create a window for manual soma selection
    window_name = "Select Multiple Somas (Click on each soma, press 'q' when done)"
    image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert image to color for colored markers
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Keep showing the image until 'q' is pressed
    while True:
        cv2.imshow(window_name, image_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return soma_coords

def process_image_multi_soma(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply binary thresholding to create a binary image
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Skeletonize the binary image to obtain a thin version of the structures
    skeleton = skeletonize(binary_img // 255)
    
    # Get soma centers from the user
    soma_centers = get_multiple_soma_centers(img)
    if not soma_centers:
        print("No soma selected. Exiting analysis.")
        return None

    # Define the maximum radius for Sholl analysis (30% of the smallest dimension)
    max_radius = int(min(img.shape) * 0.3)
    # Create a range of radii to perform the Sholl analysis
    radii = np.arange(20, max_radius, 20)
    
    all_profiles = []  # List to store Sholl profiles for each neuron
    img_with_circles = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert image to color for displaying circles

    # Iterate over each soma center
    for i, soma_center in enumerate(soma_centers):
        intersection_counts = []  # List to store intersection counts for current neuron
        for radius in radii:
            # Create a circular mask for the current radius
            rr, cc = disk(soma_center, radius, shape=img.shape)
            circle_mask = np.zeros_like(skeleton, dtype=bool)
            circle_mask[rr, cc] = 1
            # Calculate the intersection between the skeleton and the circular mask
            intersection = np.logical_and(skeleton, circle_mask)
            count = np.count_nonzero(intersection)  # Count the number of intersections
            intersection_counts.append(count)
            # Draw the Sholl circle on the image
            cv2.circle(img_with_circles, soma_center, radius, (0, 255, 0), 1)
        
        # Append the radii and counts to the list of all profiles
        all_profiles.append((radii, intersection_counts))
        # Mark the soma center with a larger circle and label it
        cv2.circle(img_with_circles, soma_center, 10, (255, 0, 0), 2)
        cv2.putText(img_with_circles, f'{i+1}', (soma_center[0]+15, soma_center[1]+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return img, skeleton, img_with_circles, all_profiles

def plot_results(img, skeleton, img_with_circles, all_profiles):
    """
    Plot the original image, skeletonized image, and image with Sholl analysis circles.
    Additionally, plot the Sholl analysis results for all selected neurons.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot the original image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Plot the skeletonized image
    plt.subplot(1, 3, 2)
    plt.imshow(skeleton, cmap='gray')
    plt.title('Skeletonized Image')
    plt.axis('off')

    # Plot the image with Sholl analysis circles
    plt.subplot(1, 3, 3)
    plt.imshow(img_with_circles)
    plt.title('Image with Sholl Analysis Circles')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Plot Sholl analysis results for all neurons
    plt.figure(figsize=(10, 6))
    for i, (radii, counts) in enumerate(all_profiles):
        plt.plot(radii, counts, 'o-', label=f'Neuron {i+1}')
    plt.title('Sholl Analysis: All Neurons')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Number of Intersections')
    plt.legend()
    plt.grid(True)
    plt.show()

def aggregate_profiles(all_profiles):
    """
    Aggregate Sholl profiles across multiple neurons by interpolating them
    to a common set of radii, and then calculating the mean and standard deviation.
    """
    # Find the maximum radius across all profiles
    max_radius = max(max(profile[0]) for profile in all_profiles)
    # Define a common set of radii for interpolation
    common_radii = np.arange(20, max_radius, 20)
    
    interpolated_counts = []
    # Interpolate each profile to the common radii
    for radii, counts in all_profiles:
        interp_counts = np.interp(common_radii, radii, counts)
        interpolated_counts.append(interp_counts)
    
    # Calculate the average and standard deviation across all profiles
    average_profile = np.mean(interpolated_counts, axis=0)
    std_profile = np.std(interpolated_counts, axis=0)
    
    return common_radii, average_profile, std_profile

def plot_aggregated_results(all_profiles, common_radii, average_profile, std_profile):
    """
    Plot the aggregated Sholl profile, showing the mean and standard deviation,
    along with individual Sholl profiles for each neuron.
    """
    plt.figure(figsize=(10, 6))
    # Plot the aggregated Sholl profile with error bars
    plt.errorbar(common_radii, average_profile, yerr=std_profile, capsize=5, fmt='o-', label='Average Profile')
    plt.fill_between(common_radii, average_profile - std_profile, average_profile + std_profile, alpha=0.3)
    plt.title("Aggregated Sholl Profile")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Average Number of Intersections")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot individual Sholl profiles with the average profile overlaid
    plt.figure(figsize=(10, 6))
    for i, (radii, counts) in enumerate(all_profiles):
        plt.plot(radii, counts, 'o-', alpha=0.7, label=f'Neuron {i+1}')
    plt.plot(common_radii, average_profile, 'r-', linewidth=2, label='Average Profile')
    plt.title("Individual and Average Sholl Profiles")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Number of Intersections")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
image_path = "multiple-neurons.jpg"  # Replace with your image path
results = process_image_multi_soma(image_path)

if results:
    img, skeleton, img_with_circles, all_profiles = results
    plot_results(img, skeleton, img_with_circles, all_profiles)
    common_radii, average_profile, std_profile = aggregate_profiles(all_profiles)
    plot_aggregated_results(all_profiles, common_radii, average_profile, std_profile)
