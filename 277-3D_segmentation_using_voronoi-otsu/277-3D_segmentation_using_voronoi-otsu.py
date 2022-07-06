# https://youtu.be/-9w0oIpMgVw
"""
3D object segmentation
uses voronoi_labeling from the pyclesperanto library in python

# For installation instructions of the pyclesperanto package, 
please refer to the following link
# https://github.com/clEsperanto/pyclesperanto_prototype

"""

#from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import numpy as np
import czifile 

input_image = czifile.imread('3d_nuclei_image.czi')
input_image= input_image[0,0,:,:,:,0]


import pyclesperanto_prototype as cle

# select a specific OpenCL / GPU device and see which one was chosen
cle.select_device('RTX')

input_gpu = cle.push(input_image)

def show(image_to_show, labels=False):
    """
    This function generates three projections: in X-, Y- and Z-direction and shows them.
    """
    projection_x = cle.maximum_x_projection(image_to_show)
    projection_y = cle.maximum_y_projection(image_to_show)
    projection_z = cle.maximum_z_projection(image_to_show)

    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    cle.imshow(projection_x, plot=axs[0], labels=labels)
    cle.imshow(projection_y, plot=axs[1], labels=labels)
    cle.imshow(projection_z, plot=axs[2], labels=labels)

show(input_gpu)
print(input_gpu.shape)

###########################
#If you do not have isotropic pixels or need to perform background corrections
#follow the tutorials from here...
# https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/Segmentation_3D.ipynb
###########################

#Segmentation
segmented = cle.voronoi_otsu_labeling(input_gpu, spot_sigma=13, 
                                      outline_sigma=1)
show(segmented, labels=True)

segmented_array = cle.pull(segmented)

#Wrtie image as tif. Ue imageJ for visualization
from skimage.io import imsave
imsave("segmented_image.tif", segmented_array) 


# Write dataset as multi-dimensional OMETIFF *image*
#Use ZEN or any other scientific image visualization s/w
from apeer_ometiff_library import io

# Expand image array to 5D of order (T, Z, C, X, Y)
# This is the convention for OMETIFF format as written by APEER library
final = np.expand_dims(segmented_array, axis=0)
final = np.expand_dims(final, axis=0)

final=np.swapaxes(final, 2, 1)

final = final.astype(np.int8)

print("Shape of the segmented volume is: T, Z, C, X, Y ", final.shape)
print(final.dtype)

# Write dataset as multi-dimensional OMETIFF *image*
io.write_ometiff("segmented_multi_channel.ome.tiff", final)


