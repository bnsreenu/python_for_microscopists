# https://youtu.be/kt8SZrSkGGw
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, io, util, measure
from skimage import img_as_ubyte


############ Using SCIKIT-IMAGE

sk_3dimg = io.imread('images/3d_image.tif')

sk_3dimg=img_as_ubyte(sk_3dimg)


def show_plane(ax, plane, cmap="gray", title=None):
    ax.imshow(plane, cmap=cmap)
    ax.axis("off")

    if title:
        ax.set_title(title)


(n_plane, n_row, n_col) = sk_3dimg.shape
_, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))

show_plane(a, sk_3dimg[n_plane // 2], title=f'Plane = {n_plane // 2}')
show_plane(b, sk_3dimg[:, n_row // 2, :], title=f'Row = {n_row // 2}')
show_plane(c, sk_3dimg[:, :, n_col // 2], title=f'Column = {n_col // 2}')


def display(im3d, cmap="gray", step=2):
    _, axes = plt.subplots(nrows=5, ncols=6, figsize=(16, 14))

    vmin = im3d.min()
    vmax = im3d.max()

    for ax, image in zip(axes.flatten(), im3d[::step]):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])


display(sk_3dimg)

#Some modules in skimage and opencv can work with 3D images. 
#But not necessarily peforming 3D operations. 

from skimage.filters import gaussian
gaussian_smoothed = gaussian(sk_3dimg)

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(sk_3dimg[50,:,:])
ax1.title.set_text('Original')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(gaussian_smoothed[50,:,:])
ax2.title.set_text('Smoothed')

#But most modules and complex operations may have to be applied slice by slice
#Process each image
#For complex operations, I recommend creating a function and calling it in the loop.
binary_img = []
for image in range(sk_3dimg.shape[0]):
    input_img = sk_3dimg[image,:,:]  #Grey images. For color add another dim.
    #thresh = threshold_otsu(input_img)
    thresh = 50
    binary = input_img > thresh
    binary_img.append(binary)
    binary_img_8bit = img_as_ubyte(binary_img)

processed_img = np.array(binary_img)

display(processed_img)

plt.imshow(processed_img[100,:,:], cmap='gray')

io.imsave('images/processed.tif', processed_img)


#Similar process can be applied to time series images
sk_time_series = io.imread('images/time_series.tif')
#TIF format is limited when it comes to handling multidimensional images

############## Using TIFFFILE #####################

"""
skimage uses tifffile in the backend so data will be identical
either way.

Use tifffile
pip install tifffile
"""

import tifffile

#
tifff_3dimg = tifffile.imread("images/3d_image.tif")
tifff_time_series = tifffile.imread("images/time_series.tif")


####################################################################################
#reading czi files
# pip install czifile 
# to import the package you need to use import czifile
# https://pypi.org/project/czifile/


################################


import czifile
from matplotlib import pyplot as plt
img = czifile.imread('images/multi_channel_z_stack_time_series.czi')


print(img.shape)  #7 dimensions
#Scenes, Time series, channels, z, y, x, RGB
#IN this example we have 1 scene, 11 time series, 2 channels, 23 z slices, 587x587 pixels of gray

#Let us extract only relevant pixels, all channels in x and y for
#scene=0, time=5, channel=1, z=11, y, x, RGB=0
img1_ch0=img[0, 5, 0, 11, :, :, 0]
img1_ch1=img[0, 5, 1, 11, :, :, 0]


fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img1_ch0, cmap='cubehelix')
ax1.title.set_text('1st channel')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(img1_ch1, cmap='hot')
ax2.title.set_text('2nd channel')
