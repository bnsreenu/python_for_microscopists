# https://youtu.be/pQfUMp51sI4
"""
Grain segmentation using < 10 lines of code in python
uses voronoi_labeling from the pyclesperanto library in python

# For installation instructions of the pyclesperanto package, 
please refer to the following link
# https://github.com/clEsperanto/pyclesperanto_prototype

"""

###### GRAIN SEGMENTATION IN LESS THAN 10 LINES OF CODE
from skimage import img_as_ubyte, io
import matplotlib.pyplot as plt
import numpy as np
import pyclesperanto_prototype as cle

input_image_original = img_as_ubyte(io.imread("grains.jpg", as_gray=True))
input_image = np.invert(input_image_original)
binary = cle.binary_not(cle.threshold_otsu(input_image))
labels = cle.voronoi_labeling(binary)
cle.imshow(labels, labels=True)



########LET US HAVE A CLOSER LOOK############

from skimage import img_as_ubyte, io
import matplotlib.pyplot as plt
import numpy as np

input_image_original = img_as_ubyte(io.imread("grains.jpg", as_gray=True))
plt.imshow(input_image_original, cmap='gray')

input_image = np.invert(input_image_original)
plt.imshow(input_image, cmap='gray')

import pyclesperanto_prototype as cle

#Leverage GPU, if you have one
# select a specific OpenCL / GPU device and see which one was chosen
cle.select_device('RTX')
input_gpu = cle.push(input_image)


##########################################
#Straightforward segmentation without any processing

#Binarize (Thresholding using otsu)
binary = cle.binary_not(cle.threshold_otsu(input_gpu))
cle.imshow(binary)
           
#Use voronoi labeling method to generate labels
labels = cle.voronoi_labeling(binary)

#Visualize
fig, axs = plt.subplots(1, 3, figsize=(15, 15))
cle.imshow(input_gpu, plot=axs[0])
cle.imshow(binary, plot=axs[1])
cle.imshow(labels, plot=axs[2], labels=True)

#If we do not want to include the edge touching grains...
labeled_excl_edges = cle.exclude_labels_on_edges(labels)
cle.imshow(labeled_excl_edges, labels=True)

####################
#Segmentation with some image processing e.g., erosion, dilation

binary = cle.binary_not(cle.threshold_otsu(input_gpu))
cle.imshow(binary)

# binary closing: dilation (minimum) followed by erosion (maximum)
binary_dilated = cle.minimum_box(binary, radius_x=2, radius_y=2)
cle.imshow(binary_dilated)

binary_eroded = cle.maximum_box(binary_dilated, radius_x=2, radius_y=2)
cle.imshow(binary_eroded)

#Generate labels
labels = cle.voronoi_labeling(binary_eroded)

fig, axs = plt.subplots(1, 3, figsize=(15, 15))
cle.imshow(input_gpu, plot=axs[0])
cle.imshow(binary, plot=axs[1])
cle.imshow(labels, plot=axs[2], labels=True)

#If we do not want to include the edge touching grains...
labeled_excl_edges = cle.exclude_labels_on_edges(labels)
cle.imshow(labeled_excl_edges, labels=True)


#######################################################
# Number of objects segmented?
#This will be the maximum label assigned to an object 9as each object is assigned unique label value)
num_objects = cle.maximum_of_all_pixels(labeled_excl_edges)
print("Total objects detected are: ", num_objects)

#Save segmented/labeled image to disk
# save image to disc
from skimage.io import imsave
labeled_array = cle.pull(labeled_excl_edges)
#This is a uint32 labeled image with each object given an integer value.
plt.imshow(labeled_array)  

imsave("labeled_image.tif", labeled_array)  #Open in imageJ for better visualization

#################################

#Extract statistics and plotting using seaborn / matplotlib
statistics = cle.statistics_of_labelled_pixels(input_gpu, labeled_excl_edges) 

import pandas as pd
stats_table = pd.DataFrame(statistics)    
print(stats_table.info())

import seaborn as sns
sns.kdeplot(stats_table['area'], shade=True)

sns.kdeplot(stats_table['mean_intensity'], shade=True)

#Map by grain size (pixel count)
pixel_count_map = cle.label_pixel_count_map(labeled_excl_edges)
cle.imshow(pixel_count_map, color_map='jet')

#Extension ratio map
#The extension ratio is a shape descriptor derived from the maximum distance 
# of pixels to their object's centroid divided by the average distance of 
#pixels to the centroid.
extension_ratio_map = cle.extension_ratio_map(labeled_excl_edges)
cle.imshow(extension_ratio_map, color_map='jet')
