# https://youtu.be/evgRgDfVuEc
"""
Object segmentation and analysis using voronoi otsu labeling 
in the pyclesperanto library in python

We will be using a multichannel CZI (Zeiss) input image for this exercise.
This requires czi file library. 

pip install czifile

For standard images (e.g., jpg, tif, etc.) use skimage, cv2, or tifffile to read
input images. 

# For installation instructions of the pyclesperanto package, 
please refer to the following link
# https://github.com/clEsperanto/pyclesperanto_prototype

"""

import sys
print("python version is: ", sys.version)


import czifile #Only required to read czi files
print("czifile version is: ", czifile.__version__)

from matplotlib import pyplot as plt
import numpy as np
import pyclesperanto_prototype as cle
from skimage import exposure, img_as_ubyte

#Read the input image
img = czifile.imread('Osteosarcoma_01.czi')

#Look at the dimensions.... multichannel and multidimensional
print(img.shape)  #6 dimensions
#Time series, scenes, channels, x, y, z, RGB
#IN this example (Osteosarcoma) we have 1 time series, 1 scene, 3 channels and each channel grey image
#size 1104 x 1376

#Let us extract only relevant pixels, all channels in x and y
img_3channel=img[0, 0, :, :, :, 0]
print(img_3channel.shape)

plt.imshow(img_3channel[0,:,:]) #Channel 1
plt.imshow(img_3channel[1,:,:]) #Channel 2
plt.imshow(img_3channel[2,:,:]) #Channel 3

#Our image of interest containing the objects (nuclei) is in channel 3
#So, let us extract that specific channel for analysis. 
DAPI = (img_3channel[2,:,:]) 
print(DAPI.dtype) #uint16. Need to convert to 8 bit

#Normalize then scale to 255 and convert to uint8 - using skimage
DAPI_8bit = img_as_ubyte(exposure.rescale_intensity(DAPI))
plt.imshow(DAPI_8bit, cmap='gray')

#Crop a small region for testing purposes
#DAPI_8bit = cle.crop(DAPI_8bit, start_x=0, start_y=0, width=256, height=256)
#plt.imshow(DAPI_8bit, cmap='gray')

# list names of all available OpenCL-devices
print("Available OpenCL devices:" + str(cle.available_device_names()))

# select a specific OpenCL / GPU device and see which one was chosen
device = cle.select_device('RTX')
print("Used GPU: ", device)

#Push the image to gpu memory
DAPI_gpu = cle.push(DAPI_8bit)
print("Image size in GPU: " + str(DAPI_gpu.shape))


cle.imshow(DAPI_gpu, color_map='gray')

############ voronoi_otsu_labeling library ##################
# voronoi_otsu_labeling(image, spot_sigma=some_number, outline_sigma=another_number)
#spot_sigma= depends on how close the detected objects can be. 
#Low number may divide large objects into multiple objects.
#outline_sigma = how precise the outline needs to be for the segmented objects (use a low number)
segmented = cle.voronoi_otsu_labeling(DAPI_gpu, spot_sigma=5, 
                                      outline_sigma=1)
cle.imshow(segmented, labels=True)

#Remove edge touching objects
segmented_excl_edges = cle.exclude_labels_on_edges(segmented)
cle.imshow(segmented_excl_edges, labels=True)

# Number of objects segmented?
#This will be the maximum label assigned to an object 9as each object is assigned unique label value)
num_objects = cle.maximum_of_all_pixels(segmented_excl_edges)
print("Total objects detected are: ", num_objects)

#Save segmented image to disk
# save image to disk
from skimage.io import imsave
segmented_array = cle.pull(segmented_excl_edges)
#This is a uint32 labeled image with each object given an integer value.
plt.imshow(segmented_array)  

imsave("result.tif", segmented_array)  #Open in imageJ for better visualization

###############################################################

#Useful plotting within cle

#Pixel count map - map by object size
pixel_count_map = cle.label_pixel_count_map(segmented_excl_edges)
cle.imshow(pixel_count_map, color_map='jet')

#Extension ratio map
#The extension ratio is a shape descriptor derived from the maximum distance 
# of pixels to their object's centroid divided by the average distance of 
#pixels to the centroid.
extension_ratio_map = cle.extension_ratio_map(segmented_excl_edges)
cle.imshow(extension_ratio_map, color_map='jet')

#Mean / minimum / maximum / standard-deviation intensity map
mean_intensity_map = cle.label_mean_intensity_map(DAPI_8bit, segmented_excl_edges)
cle.imshow(mean_intensity_map, color_map='jet')

minimum_intensity_map = cle.minimum_intensity_map(DAPI_8bit, segmented_excl_edges)
cle.imshow(minimum_intensity_map, color_map='jet')

maximum_intensity_map = cle.maximum_intensity_map(DAPI_8bit, segmented_excl_edges)
cle.imshow(maximum_intensity_map, color_map='jet')

stddev_intensity_map = cle.standard_deviation_intensity_map(DAPI_8bit, segmented_excl_edges)
cle.imshow(stddev_intensity_map, color_map='jet')

#Neighbor count maps
enlarged_labels = cle.extend_labeling_via_voronoi(segmented_excl_edges)
cle.imshow(enlarged_labels, labels=True)

touching_neighbor_count_map = cle.touching_neighbor_count_map(enlarged_labels)
cle.imshow(touching_neighbor_count_map, color_map='jet')

proximal_neighbor_count_map = cle.proximal_neighbor_count_map(segmented_excl_edges, max_distance=70)
cle.imshow(proximal_neighbor_count_map, color_map='jet')

#Distance to neighbor maps

n_nearest_neighbor_distance_map = cle.average_distance_of_n_closest_neighbors_map(segmented_excl_edges, n=3)
cle.imshow(n_nearest_neighbor_distance_map, color_map='jet')

################################################################

#Extract statistics and plot using seaborn / matplotlib
statistics = cle.statistics_of_labelled_pixels(DAPI_gpu, segmented_excl_edges) 

import pandas as pd
stats_table = pd.DataFrame(statistics)    

print(stats_table.describe())
print(stats_table.columns)

import seaborn as sns
sns.kdeplot(stats_table['area'], shade=True)

sns.kdeplot(stats_table['mean_intensity'], shade=True)

#Correlation plot
sns.pairplot(stats_table, x_vars=["mean_distance_to_centroid", "mean_intensity", "mean_distance_to_mass_center"], y_vars="area", size=6, aspect=0.75)

###############################################################################
#Draw bounding boxes around detected objects

#Convert cle object to numpy array
my_image=np.uint8(cle.pull(DAPI_8bit))
minx=stats_table['bbox_min_x'].values
miny=stats_table['bbox_min_y'].values
maxx=stats_table['bbox_max_x'].values
maxy=stats_table['bbox_max_y'].values

import cv2

for i in range (len(minx)):
    print(i)
    (x_min, y_min),(x_max,y_max) = (int(minx[i]),int(miny[i])), (int(maxx[i]), int(maxy[i]))
    cv2.rectangle(my_image,(x_min,y_min),(x_max,y_max),(255,255,255), 2) # add rectangle to image

cv2.imshow("Image", my_image)
cv2.waitKey(0) 
cv2.destroyAllWindows() 
#########################################################################

