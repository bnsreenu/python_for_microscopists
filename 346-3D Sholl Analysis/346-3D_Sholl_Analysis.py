# https://youtu.be/wm2e0FpvnIA
"""
3D Sholl Analysis Approach Summary:

This code performs 3D Sholl analysis on neuronal image stacks by defining concentric spheres around a selected soma. The process involves:

1. Soma Selection (Manual or Automatic):
   - Manual Method:
     * The user manually selects the soma by drawing rectangles on two orthogonal planes (XY and YZ views) of the 3D image stack.
     * The X and Z coordinates are taken directly from the slice positions selected using the sliders
     * The system calculates the soma center y coordinate by averaging the coordinates of the rectangle centers in both views.
   - Automatic Method:
     * The code identifies the soma by finding the region of highest intensity within a cubic window (default 9x9x9 pixels) that slides through the 3D image stack.
     * The center of the highest intensity cube is selected as the soma center.

2. Image Processing:
   - The 3D image stack is binarized using Otsu's thresholding method.
   - The binary image is then skeletonized to represent the neuron's structure as a thin line in 3D space.

3. Sholl Analysis:
   - Concentric spheres are defined around the soma center, with radii increasing at a specified step size.
   - For each sphere, the code counts the number of intersections between the sphere's surface and the skeletonized neuron in 3D space.
   - This is done by creating a distance transform of the skeleton and identifying voxels that fall within a thin shell at each radius.

The code creates a series of concentric spherical shells around the soma center 
by calculating the distance of each point in 3D space from the soma center. 
For each spherical shell, it then counts how many times the skeletonized neuron 
intersects with that shell by performing a logical AND operation between the neuron 
skeleton and the shell, effectively counting the number of neuron branches at 
each distance from the soma.
In simpler terms: Imagine creating a series of increasingly larger hollow spheres 
centered on the soma, and for each sphere, counting how many times the neuron's 
branches pass through that sphere's surface.



4. Visualization:
   - The results are visualized through multiple plots:
     a) A 3D plot of the skeletonized neuron
     b) A 2D Sholl profile showing intersection counts vs. radius
     c) 2D projections (XY, XZ, YZ) with Sholl circles overlaid
     d) A 3D rendering of the skeletonized neuron with transparent Sholl spheres
   - An interactive 3D visualization is generated using Plotly and saved as an HTML file, allowing for:
     * Dynamic rotation, zooming, and panning of the 3D model
     * Toggling visibility of different components (neuron, spheres, soma center)
     * Color-coded spheres to represent different radii
     * Preset viewing angles (top view, side view, reset view)

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d
from scipy.ndimage import distance_transform_edt, uniform_filter
import cv2
import tifffile as tiff
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go
import os


def enhanced_soma_selection(image_stack):
    """
    Allow manual selection of the soma center by showing XY and YZ views.
    User can scroll through slices and draw rectangles in each view.
    
    Args:
        image_stack (numpy.ndarray): 3D grayscale image stack of the neuron.
    
    Returns:
        tuple: (x, y, z) coordinates of the soma center calculated from rectangles.
    """
    # Initialize variables
    current_slice_xy = image_stack.shape[0] // 2
    current_slice_yz = image_stack.shape[2] // 2
    points_xy = {"start": None, "end": None}
    points_yz = {"start": None, "end": None}
    display_xy = None
    display_yz = None
    drawing_xy = False
    drawing_yz = False

    def update_views():
        """Update both view windows with current slices and rectangles."""
        nonlocal display_xy, display_yz
        
        # Normalize and prepare XY view
        slice_xy = cv2.normalize(image_stack[current_slice_xy], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        display_xy = cv2.cvtColor(slice_xy, cv2.COLOR_GRAY2BGR)
        
        # Normalize and prepare YZ view
        slice_yz = cv2.normalize(np.rot90(image_stack[:, :, current_slice_yz]), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        display_yz = cv2.cvtColor(slice_yz, cv2.COLOR_GRAY2BGR)

        # Draw rectangles if they exist
        if points_xy["start"] and drawing_xy:
            cv2.rectangle(display_xy, points_xy["start"], points_xy["end"], (0, 255, 0), 2)
        elif points_xy["start"] and points_xy["end"]:
            cv2.rectangle(display_xy, points_xy["start"], points_xy["end"], (0, 255, 0), 2)
            center = ((points_xy["start"][0] + points_xy["end"][0]) // 2,
                     (points_xy["start"][1] + points_xy["end"][1]) // 2)
            cv2.circle(display_xy, center, 3, (0, 0, 255), -1)

        if points_yz["start"] and drawing_yz:
            cv2.rectangle(display_yz, points_yz["start"], points_yz["end"], (0, 255, 0), 2)
        elif points_yz["start"] and points_yz["end"]:
            cv2.rectangle(display_yz, points_yz["start"], points_yz["end"], (0, 255, 0), 2)
            center = ((points_yz["start"][0] + points_yz["end"][0]) // 2,
                     (points_yz["start"][1] + points_yz["end"][1]) // 2)
            cv2.circle(display_yz, center, 3, (0, 0, 255), -1)

        # Display slice information
        cv2.putText(display_xy, f'Z slice: {current_slice_xy}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_yz, f'X slice: {current_slice_yz}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(window_name_xy, display_xy)
        cv2.imshow(window_name_yz, display_yz)

    def on_trackbar(val):
        """Handle trackbar movements."""
        nonlocal current_slice_xy, current_slice_yz
        try:
            current_slice_xy = cv2.getTrackbarPos('Z Slice', window_name_xy)
            current_slice_yz = cv2.getTrackbarPos('X Slice', window_name_yz)
            update_views()
        except cv2.error:
            pass

    def draw_rectangle_xy(event, x, y, flags, param):
        """Handle mouse events for XY view."""
        nonlocal drawing_xy, points_xy
        if event == cv2.EVENT_LBUTTONDOWN:
            points_xy["start"] = (x, y)
            drawing_xy = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing_xy:
            points_xy["end"] = (x, y)
            update_views()
        elif event == cv2.EVENT_LBUTTONUP:
            points_xy["end"] = (x, y)
            drawing_xy = False
            update_views()

    def draw_rectangle_yz(event, x, y, flags, param):
        """Handle mouse events for YZ view."""
        nonlocal drawing_yz, points_yz
        if event == cv2.EVENT_LBUTTONDOWN:
            points_yz["start"] = (x, y)
            drawing_yz = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing_yz:
            points_yz["end"] = (x, y)
            update_views()
        elif event == cv2.EVENT_LBUTTONUP:
            points_yz["end"] = (x, y)
            drawing_yz = False
            update_views()

    # Create and setup windows
    window_name_xy = "XY View - Draw rectangle (click & drag)"
    window_name_yz = "YZ View - Draw rectangle (click & drag)"
    cv2.namedWindow(window_name_xy, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_yz, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_xy, 800, 600)
    cv2.resizeWindow(window_name_yz, 800, 600)

    # Create trackbars
    cv2.createTrackbar('Z Slice', window_name_xy, current_slice_xy, image_stack.shape[0] - 1, on_trackbar)
    cv2.createTrackbar('X Slice', window_name_yz, current_slice_yz, image_stack.shape[2] - 1, on_trackbar)

    # Set mouse callbacks
    cv2.setMouseCallback(window_name_xy, draw_rectangle_xy)
    cv2.setMouseCallback(window_name_yz, draw_rectangle_yz)

    # Initial view update
    update_views()

    print("\nInstructions:")
    print("1. Use sliders to navigate through the stack")
    print("2. Draw rectangles in both views around the soma")
    print("3. Press 'r' to reset selection")
    print("4. Press 'q' to finish selection\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points_xy = {"start": None, "end": None}
            points_yz = {"start": None, "end": None}
            update_views()

    cv2.destroyAllWindows()

    if points_xy["start"] and points_xy["end"] and points_yz["start"] and points_yz["end"]:
        # Calculate centers from rectangles
        center_xy = ((points_xy["start"][0] + points_xy["end"][0]) // 2,
                    (points_xy["start"][1] + points_xy["end"][1]) // 2)
        center_yz = ((points_yz["start"][0] + points_yz["end"][0]) // 2,
                    (points_yz["start"][1] + points_yz["end"][1]) // 2)

        # Print debug information
        print("\nDebug Information:")
        print(f"XY rectangle center: {center_xy}")
        print(f"YZ rectangle center: {center_yz}")
        print(f"Selected slices - X: {current_slice_yz}, Z: {current_slice_xy}")

        # Calculate final soma center
        soma_center = (
            current_slice_yz,    # X coordinate (from X slice slider)
            (center_xy[1] + center_yz[1]) // 2,  # Y coordinate (average from both rectangles)
            current_slice_xy     # Z coordinate (from Z slice slider)
        )

        print(f"Final soma center (x, y, z): {soma_center}\n")
        return soma_center
    else:
        print("No selection made. Using default center.")
        return None


def automated_soma_detection(image_stack, cube_size=9):   
    """
    Automatically detect the soma center based on maximum intensity in a sliding cube.
    
    Args:
        image_stack (numpy.ndarray): 3D grayscale image stack of the neuron.
        cube_size (int): Size of the cubic window to use for intensity calculation.
    
    Returns:
        tuple: (x, y, z) coordinates of the detected soma center.
    """
    # Calculate the sum of intensities in a sliding cubic window
    intensity_sum = uniform_filter(image_stack, size=cube_size, mode='constant') #uniform_filter from scipy.ndimage is used to calculate the average intensity in a sliding cubic window
    
    # Find the coordinates of the maximum intensity sum
    z, y, x = np.unravel_index(np.argmax(intensity_sum), intensity_sum.shape)
    
    return (x, y, z)



def get_soma_center(image_stack, method):
    """
    Get the soma center using the specified method.
    
    Args:
        image_stack (numpy.ndarray): 3D grayscale image stack of the neuron.
        method (str): "manual" for user selection, "auto" for automated detection.
    
    Returns:
        tuple: (x, y, z) coordinates of the soma center.
    """
    if method == "manual":
        soma_center = enhanced_soma_selection(image_stack)
        if soma_center is None:
            print("Soma selection cancelled. Using default center.")
            soma_center = (image_stack.shape[2] // 2, image_stack.shape[1] // 2, image_stack.shape[0] // 2)
    elif method == "auto":
        soma_center = automated_soma_detection(image_stack)
        print(f"Automatically detected soma center: {soma_center}")
    else:
        raise ValueError("Invalid soma detection method. Choose 'manual' or 'auto'.")
    
    return soma_center

def visualize_soma_center(image_stack, soma_center):
    """
    Visualize the soma center on XY, XZ, and YZ planes.
    
    Args:
        image_stack (numpy.ndarray): 3D grayscale image stack of the neuron.
        soma_center (tuple): (x, y, z) coordinates of the soma center.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    x, y, z = soma_center

    # XY plane
    ax1.imshow(image_stack[z], cmap='gray')
    ax1.plot(x, y, 'r.', markersize=10)
    ax1.set_title('XY Plane')
    ax1.axis('off')

    # XZ plane
    ax2.imshow(image_stack[:, y, :].T, cmap='gray')
    ax2.plot(x, z, 'r.', markersize=10)
    ax2.set_title('XZ Plane')
    ax2.axis('off')

    # YZ plane
    ax3.imshow(image_stack[:, :, x].T, cmap='gray')
    ax3.plot(y, z, 'r.', markersize=10)
    ax3.set_title('YZ Plane')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

def perform_3d_sholl_analysis(skeleton_3d, soma_center, radii):
    """
    Perform 3D Sholl analysis using spheres.
    
    Args:
        skeleton_3d (numpy.ndarray): 3D skeletonized image of the neuron.
        soma_center (tuple): (x, y, z) coordinates of the soma center.
        radii (numpy.ndarray): Array of radii for Sholl analysis.
    
    Returns:
        list: Intersection counts for each radius.
    """
    intersection_counts = []
    y, x, z = np.ogrid[-soma_center[0]:skeleton_3d.shape[0]-soma_center[0],
                       -soma_center[1]:skeleton_3d.shape[1]-soma_center[1],
                       -soma_center[2]:skeleton_3d.shape[2]-soma_center[2]]
    distances = np.sqrt(x*x + y*y + z*z)
    
    for radius in radii:
        shell = (distances <= radius) & (distances > radius - 1)
        intersections = np.sum(skeleton_3d & shell)
        intersection_counts.append(intersections)
    
    return intersection_counts




def plot_3d_skeleton(skeleton_3d):
    """
    Plot the 3D skeletonized image.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.where(skeleton_3d)
    ax.scatter(x, y, z, c='b', alpha=0.1, s=1)
    ax.set_title('3D Skeletonized Image')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_3d_sholl_profile(radii, intersection_counts):
    """
    Plot the 3D Sholl analysis profile.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(radii, intersection_counts, '-o')
    ax.set_title('3D Sholl Profile')
    ax.set_xlabel('Radius (pixels)')
    ax.set_ylabel('Number of Intersections')
    plt.show()

def plot_2d_projections(image_stack, soma_center, radii):
    """
    Plot XY, XZ, and YZ projections with Sholl circles.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # XY plane
    mid_z = image_stack.shape[0] // 2
    ax1.imshow(image_stack[mid_z], cmap='gray')
    for radius in radii[::3]:
        circle = plt.Circle((soma_center[1], soma_center[2]), radius, fill=False, color='r', alpha=0.5)
        ax1.add_artist(circle)
    ax1.set_title('XY Projection')
    ax1.axis('off')

    # XZ plane
    mid_y = image_stack.shape[1] // 2
    ax2.imshow(image_stack[:, mid_y, :].T, cmap='gray')
    for radius in radii[::3]:
        circle = plt.Circle((soma_center[2], soma_center[0]), radius, fill=False, color='r', alpha=0.5)
        ax2.add_artist(circle)
    ax2.set_title('XZ Projection')
    ax2.axis('off')

    # YZ plane
    mid_x = image_stack.shape[2] // 2
    ax3.imshow(image_stack[:, :, mid_x].T, cmap='gray')
    for radius in radii[::3]:
        circle = plt.Circle((soma_center[1], soma_center[0]), radius, fill=False, color='r', alpha=0.5)
        ax3.add_artist(circle)
    ax3.set_title('YZ Projection')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()





def plot_3d_neuron_with_spheres(skeleton_3d, soma_center, radii):
    """
    Plot 3D visualization of the skeletonized neuron with cross-sectional Sholl analysis spheres.
    
    Parameters:
    skeleton_3d (numpy.ndarray): 3D array of the skeletonized neuron
    soma_center (tuple): (x, y, z) coordinates of the soma center
    radii (numpy.ndarray): Array of radii for Sholl analysis spheres
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the skeletonized neuron
    z, x, y = np.where(skeleton_3d > 0)
    ax.scatter(x, y, z, c='blue', alpha=0.1, s=1)

    # Plot cross-sectional spheres
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)

    for radius in radii[::3]:  # Plot every third sphere for clarity
        x = radius * np.outer(np.cos(u), np.sin(v)) + soma_center[1]
        y = radius * np.outer(np.sin(u), np.sin(v)) + soma_center[2]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + soma_center[0]
        
        # Plot only a quarter of each sphere
        ax.plot_surface(x[:50,:25], y[:50,:25], z[:50,:25], color='r', alpha=0.1)
        
        # Plot the edges of the quarter sphere
        ax.plot(x[0,:25], y[0,:25], z[0,:25], color='r', alpha=0.5)
        ax.plot(x[-1,:25], y[-1,:25], z[-1,:25], color='r', alpha=0.5)
        ax.plot(x[:50,0], y[:50,0], z[:50,0], color='r', alpha=0.5)

    ax.set_title('3D Skeletonized Neuron with Sholl Analysis Spheres')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set aspect ratio to be equal
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

    # Adjust the view angle for better visibility
    ax.view_init(elev=20, azim=45)

    plt.show()




def plot_3d_neuron_with_spheres_plotly(skeleton_3d, soma_center, radii, output_path='3d_sholl_analysis.html'):
    """
    Create a revised interactive 3D plot of the skeletonized neuron with visible Sholl analysis spheres using Plotly
    and save it as an HTML file.
    """
    fig = go.Figure()

    # Plot the skeletonized neuron
    z, x, y = np.where(skeleton_3d > 0)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.5),
        name='Neuron'
    ))

    # Plot Sholl analysis spheres
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)

    for i, radius in enumerate(radii[::2]):  # Plot every second sphere for balance
        x = radius * np.outer(np.cos(u), np.sin(v)) + soma_center[1]
        y = radius * np.outer(np.sin(u), np.sin(v)) + soma_center[2]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + soma_center[0]

        color = plt.cm.viridis(i / len(radii[::2]))  # Use viridis colormap
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            surfacecolor=np.full_like(x, i),
            opacity=0.2,
            showscale=False,
            colorscale=[[0, f'rgba({color[0]*255},{color[1]*255},{color[2]*255},0.2)'], 
                        [1, f'rgba({color[0]*255},{color[1]*255},{color[2]*255},0.2)']],
            name=f'Sphere r={radius}',
        ))

    # Add soma center
    fig.add_trace(go.Scatter3d(
        x=[soma_center[1]], y=[soma_center[2]], z=[soma_center[0]],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Soma Center'
    ))

    # Update layout
    fig.update_layout(
        title='Interactive 3D Skeletonized Neuron with Sholl Analysis Spheres',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=900,
        height=900,
        autosize=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(label="Reset View",
                         method="relayout",
                         args=[{"scene.camera": dict(eye=dict(x=1.25, y=1.25, z=1.25))}]),
                    dict(label="Top View",
                         method="relayout",
                         args=[{"scene.camera": dict(eye=dict(x=0, y=0, z=2.5))}]),
                    dict(label="Side View",
                         method="relayout",
                         args=[{"scene.camera": dict(eye=dict(x=2.5, y=0, z=0))}])
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    # Save the plot as an HTML file
    fig.write_html(output_path)
    print(f"Interactive 3D plot saved to {os.path.abspath(output_path)}")

def visualize_results(image_stack, skeleton_3d, soma_center, radii, intersection_counts):
    plot_3d_skeleton(skeleton_3d)
    plot_3d_sholl_profile(radii, intersection_counts)
    plot_2d_projections(image_stack, soma_center, radii)
    plot_3d_neuron_with_spheres(skeleton_3d, soma_center, radii)  # Updated call


def main():
    # Configuration parameters
    SOMA_DETECTION_METHOD = 'auto'  # 'auto' or 'manual'
    CUBE_SIZE = 9  # Size of the cube for automated soma detection (only used if SOMA_DETECTION_METHOD is 'auto')
    image_path = "3D_neuron_stack_512_512_512.tiff"
    
    # Load image stack
    img_stack = tiff.imread(image_path)

    # Create a binary image using Otsu's thresholding
    binary_img_stack = np.array([(cv2.threshold(slice_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] // 255) 
                                 for slice_img in img_stack])

    # Create a skeletonized version of the binary image
    skeleton_3d = skeletonize_3d(binary_img_stack)

    # Detect the soma center using the selected method
    if SOMA_DETECTION_METHOD == 'auto':
        soma_center = automated_soma_detection(img_stack, cube_size=CUBE_SIZE)
    else:
        soma_center = get_soma_center(img_stack, SOMA_DETECTION_METHOD)
    
    print(f"Selected soma center: {soma_center}")

    # Visualize the soma center
    visualize_soma_center(img_stack, soma_center)

    # Sholl Analysis Parameters
    radius_step = 7  # Distance between concentric spheres
    max_radius = int(max(img_stack.shape) * 0.6)  # Use 60% of the larger dimension

    # Prepare for Sholl analysis
    radii = np.arange(radius_step, max_radius, radius_step)

    # Perform 3D Sholl analysis
    intersection_counts = perform_3d_sholl_analysis(skeleton_3d, soma_center, radii)

    # Visualize the results
    visualize_results(img_stack, skeleton_3d, soma_center, radii, intersection_counts)
    
    # interactive Plotly visualization
    plot_3d_neuron_with_spheres_plotly(skeleton_3d, soma_center, radii, 'sholl_analysis_3d.html')

if __name__ == "__main__":
    main()