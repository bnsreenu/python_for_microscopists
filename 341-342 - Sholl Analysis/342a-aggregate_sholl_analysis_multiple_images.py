# https://youtu.be/SB9hVcYIoXk
"""
Author: Dr. Sreenivas Bhattiprolu
Date: September 2024
Version: 1.1

Performs Sholl analysis on individual images in a directory and combines them. 

When combining Sholl profiles from multiple neurons, we often encounter a challenge: 
different neurons may have different sizes, leading to varying maximum radii in 
their Sholl analyses. Tdifferent neurons may have different sizes, leading to varying maximum radii in 
their Sholl analyses..

To address this:

We first determine the largest radius used across all neurons.
We then create a common set of radii points, typically ranging from the smallest 
to the largest radius used in any of the analyses.
For each neuron's profile, we use interpolation to estimate what the intersection 
counts would be at these common radii points. Interpolation essentially "fills in" 
values between the actual measured points, allowing us to estimate intersection 
counts at radii that weren't directly measured for that particular neuron.
This process results in each neuron having estimated intersection counts for the 
same set of radii, even if the original measurements didn't extend to all of these radii.
With this standardized data, we can now directly compare and average the profiles across 
all neurons, calculating mean intersection counts and standard deviations at each radius point.

This interpolation step is crucial because it allows us to meaningfully combine 
data from neurons of different sizes or from images with different scales. 
It ensures that when we calculate the average profile or compare neurons, 
we're comparing equivalent points along the radius, regardless of the absolute size of each neuron.

Aggregated Sholl Profile plot shows the average number of intersections at each 
radius across all analyzed neurons. It includes Error bars or a shaded area 
indicating the standard deviation or variability at each point.

Individual and Average Sholl Profiles plot overlays the Sholl profiles of all 
individual neurons along with the average profile.

    
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.draw import disk
import os

def get_soma_center(image):
    """
    Allow manual selection of the soma center via mouse click.
    """
    soma_coords = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            soma_coords.append((x, y))
            cv2.destroyAllWindows()

    window_name = "Select Soma (Click on the cell body)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Display the image until the user selects the soma center or presses 'q'
    while True:
        cv2.imshow(window_name, image)
        if cv2.waitKey(1) & 0xFF == ord('q') or soma_coords:
            break

    cv2.destroyAllWindows()
    return soma_coords[0] if soma_coords else None

def process_image(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Threshold the image to create a binary image
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Skeletonize the binary image
    skeleton = skeletonize(binary_img // 255)
    
    # Allow the user to manually select the soma center
    soma_center = get_soma_center(img)
    if soma_center is None:
        print(f"Soma selection cancelled for {image_path}. Skipping this image.")
        return None, None, None, None, None

    # Set up parameters for Sholl analysis
    max_radius = int(max(img.shape) * 0.6)  # Max radius as a percentage of image size
    radii = np.arange(20, max_radius, 20)  # Radii at which intersections are counted
    intersection_counts = []

    # Convert the grayscale image to BGR for displaying colored circles
    img_with_circles = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Perform Sholl analysis by counting intersections at each radius
    for radius in radii:
        rr, cc = disk(soma_center, radius, shape=img.shape)  # Generate a disk-shaped mask
        circle_mask = np.zeros_like(skeleton, dtype=bool)
        circle_mask[rr, cc] = 1  # Apply the mask to create a circle

        # Perform intersection by AND operation between the skeleton and the circle
        intersection = np.logical_and(skeleton, circle_mask)
        count = np.count_nonzero(intersection)
        intersection_counts.append(count)

        # Draw the circle on the image for visualization
        cv2.circle(img_with_circles, soma_center, radius, (0, 255, 0), 1)

    return img, skeleton, img_with_circles, radii, intersection_counts

def plot_image_results(img, skeleton, img_with_circles, radii, intersection_counts, filename):
    # Plot original, skeletonized, and circled images side by side
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Original Image: {filename}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(skeleton, cmap='gray')
    plt.title('Skeletonized Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_with_circles)
    plt.title('Image with Sholl Analysis Circles')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Plot Sholl analysis results (intersections vs radius)
    plt.figure(figsize=(8, 6))
    plt.plot(radii, intersection_counts, 'o-')
    plt.title(f'Sholl Analysis: {filename}')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Number of Intersections')
    plt.grid(True)
    plt.show()

def process_multiple_images(image_folder):
    # Process each image in the folder and store Sholl profiles
    all_profiles = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            print(f"Processing {filename}...")
            
            results = process_image(image_path)
            if results[0] is not None:
                img, skeleton, img_with_circles, radii, counts = results
                plot_image_results(img, skeleton, img_with_circles, radii, counts, filename)
                all_profiles.append((radii, counts))
    
    return all_profiles

def aggregate_profiles(all_profiles):
    # Determine the maximum radius used across all profiles
    max_radius = max(max(profile[0]) for profile in all_profiles)
    
    # Define a common set of radii for interpolation
    common_radii = np.arange(20, max_radius, 20)
    
    # Interpolate intersection counts to the common set of radii
    interpolated_counts = []
    for radii, counts in all_profiles:
        interp_counts = np.interp(common_radii, radii, counts)  # Interpolation happens here
        interpolated_counts.append(interp_counts)
    
    # Calculate the average profile and standard deviation across all neurons
    average_profile = np.mean(interpolated_counts, axis=0)
    std_profile = np.std(interpolated_counts, axis=0)
    
    return common_radii, average_profile, std_profile

def plot_aggregated_results(all_profiles, common_radii, average_profile, std_profile):
    # Plot the aggregated Sholl profile with error bars
    plt.figure(figsize=(10, 6))
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
    for radii, counts in all_profiles:
        plt.plot(radii, counts, 'o-', alpha=0.3)
    plt.plot(common_radii, average_profile, 'r-', linewidth=2, label='Average Profile')
    plt.title("Individual and Average Sholl Profiles")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Number of Intersections")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
image_folder = "images"  # Replace with your folder path
all_profiles = process_multiple_images(image_folder)
common_radii, average_profile, std_profile = aggregate_profiles(all_profiles)
plot_aggregated_results(all_profiles, common_radii, average_profile, std_profile)
