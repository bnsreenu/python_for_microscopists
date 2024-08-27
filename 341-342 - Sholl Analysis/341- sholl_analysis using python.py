# https://youtu.be/vK9Cf8hrY3Q
"""
Sholl analysis is a method used in neuroscience to quantify the complexity of 
neuronal branching patterns.

It is performed to measure how neuron dendrites or axons branch out from the cell body.

A series of concentric circles is overlaid on an image of a neuron.
The circles are centered on the cell body (soma).
The number of times neuronal processes intersect each circle is counted.
When performing automated analysis, the soma is assumed to be the largest, 
brightest object in the image.

Concentric circles are Placed around the soma because branching typically radiates outward.
Usually spaced at regular intervals (e.g., 10 μm apart).
Extend far enough to cover the entire dendritic field.

Analysis:
The intersection counts are plotted against circle radius.
This plot shows how branching complexity changes with distance from the soma.

This type of anlysis is usually performed when ...
Comparing neuron morphology between different cell types or conditions.
Assessing changes in neuronal structure during development or disease.

Image credit:
    neuron2.png: https://www.mbfbioscience.com/wp-content/uploads/2022/06/neurontracingconfocalmicroscopy-661x1024.png
    
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.draw import disk
from scipy import ndimage

# User configuration
SOMA_DETECTION_METHOD = "manual"  # Options: "auto" or "manual"

def identify_soma_auto(image):
    """
    Automatically identify the soma (cell body) in the neuron image.
    Basically, the brightest spot, which is usually the soma but can be wrong. 
    Use manual soma selection if auto isn;t working on your image. 
    
    Args:
        image (numpy.ndarray): Grayscale image of the neuron.
    
    Returns:
        tuple: (x, y) coordinates of the identified soma center.
    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.circle(mask, max_loc, 20, 255, -1)
    moments = cv2.moments(mask)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return (cx, cy)

def manual_soma_selection(image):
    """
    Allow manual selection of the soma center via mouse click.
    
    Args:
        image (numpy.ndarray): Grayscale image of the neuron.
    
    Returns:
        tuple: (x, y) coordinates of the manually selected soma center.
    """
    soma_coords = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            soma_coords.append((x, y))
            cv2.destroyAllWindows()

    window_name = "Select Soma (Click on the cell body)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, image)
        if cv2.waitKey(1) & 0xFF == ord('q') or soma_coords:
            break

    cv2.destroyAllWindows()
    return soma_coords[0] if soma_coords else None

def get_soma_center(image, method):
    """
    Get the soma center using the specified method.
    
    Args:
        image (numpy.ndarray): Grayscale image of the neuron.
        method (str): "auto" for automatic detection, "manual" for user selection.
    
    Returns:
        tuple: (x, y) coordinates of the soma center.
    """
    if method == "auto":
        return identify_soma_auto(image)
    elif method == "manual":
        soma_center = manual_soma_selection(image)
        if soma_center is None:
            print("Soma selection cancelled. Using default center.")
            soma_center = (image.shape[1] // 2, image.shape[0] // 2)
        return soma_center
    else:
        raise ValueError("Invalid soma detection method. Choose 'auto' or 'manual'.")

# Main analysis code. 

# Load image and convert to grey. 
image_path = "images/neurons3.png"  # neurons2.png neurons.jpg
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create a binary image using Otsu's thresholding
_, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Create a skeletonized version of the binary image
skeleton = skeletonize(binary_img // 255)

# Detect the soma center using autom or manual
soma_center = get_soma_center(img, SOMA_DETECTION_METHOD)
print(f"Selected soma center: {soma_center}")

# Sholl Analysis Parameters
radius_step = 20  # Distance between concentric circles
#max_radius = min(img.shape) // 2  # Maximum radius for analysis
max_radius = int(max(img.shape) * 0.6)  # Use 75% of the larger dimension

# Prepare for Sholl analysis
radii = np.arange(radius_step, max_radius, radius_step)
intersection_counts = []

# Create a color image to overlay Sholl analysis circles
img_with_circles = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Perform Sholl analysis.
#For each radius in the predefined range, we create a circular mask centered on the soma.

for radius in radii:
    # Create a circular mask
    rr, cc = disk(soma_center, radius, shape=img.shape) #Generate coordinates for a circle of given radius
    circle_mask = np.zeros_like(skeleton, dtype=bool)  #Create a boolean mask where the circle is located
    circle_mask[rr, cc] = 1
    
    # Count intersections between the circle and the skeleton
    #identify where the circle intersects with the neuron's branches.
    intersection = np.logical_and(skeleton, circle_mask) #se a logical AND operation between the skeleton and the circle mask.
    count = np.count_nonzero(intersection) #count the number of non-zero pixels in this intersection, which represents the number of branch intersections at this radius.
    intersection_counts.append(count)  #Add intersection counts to the list. 
    
    # Draw the circle on the image for visualization
    cv2.circle(img_with_circles, soma_center, radius, (0, 255, 0), 1)

# Create an overlay of the skeleton on the original image
img_with_skeleton_overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_with_skeleton_overlay[skeleton == 1] = [255, 0, 0]  # Red color for the skeleton

# Visualization of results
plt.figure(figsize=(12, 12))
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

plt.figure(figsize=(12, 12))
plt.imshow(binary_img, cmap='gray')
plt.title('Binary Image')
plt.axis('off')
plt.show()

plt.figure(figsize=(12, 12))
plt.imshow(skeleton, cmap='gray')
plt.title('Skeletonized Image')
plt.axis('off')
plt.show()

plt.figure(figsize=(12, 12))
plt.imshow(img_with_circles)
plt.title('Image with Sholl Analysis Circles')
plt.axis('off')
plt.show()

plt.figure(figsize=(12, 12))
plt.imshow(img_with_skeleton_overlay)
plt.title('Original Image with Skeleton Overlay')
plt.axis('off')
plt.show()


######################

#Linear fit method (N as a function of Radius). 
#Other type of fits are also often performed, like semi-log and log-log
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Assuming radii and intersection_counts are already defined


def linear_fit(r, *params):
    return sum(param * r**i for i, param in enumerate(params))

degree = 3  # Adjust as needed
popt, pcov = optimize.curve_fit(linear_fit, radii, intersection_counts, p0=[1]*(degree+1))

# Calculate R-squared
residuals = intersection_counts - linear_fit(radii, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((intersection_counts - np.mean(intersection_counts))**2)
r_squared = 1 - (ss_res / ss_tot)

# Calculate standard errors of the parameters
perr = np.sqrt(np.diag(pcov))

# Plot
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(radii, intersection_counts, 'o', label='Data')
plt.plot(radii, linear_fit(radii, *popt), '-', label='Polynomial fit')
plt.title("Linear Sholl Profile")
plt.xlabel("Radius (pixels)")
plt.ylabel("Number of Intersections (N)")
plt.legend()

# Print fit details
print("Fit Details for Linear Sholl Profile:")
print(f"Degree of polynomial: {degree}")
print("Coefficients:")
for i, (param, err) in enumerate(zip(popt, perr)):
    print(f"  a{i} = {param:.4e} ± {err:.4e}")
print(f"R-squared: {r_squared:.4f}")

# Calculate AIC and BIC
n = len(radii)
k = len(popt)
aic = 2*k + n*np.log(ss_res/n)
bic = np.log(n)*k + n*np.log(ss_res/n)
print(f"AIC: {aic:.4f}")
print(f"BIC: {bic:.4f}")

