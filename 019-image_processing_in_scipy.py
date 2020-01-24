#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=s_hDL2fGvow&t=

"""
#Image processing using Scipy
Scipy is a python library that is part of numpy stack. 
It contains modules for linear algebra, FFT, signal processing and
image processing. Not designed for image processing but has a few tools

"""

#You can use imread from scipy to read images

from scipy import misc
img = misc.imread("images/monkey.jpg")
print(type(img))   #numpy array

#since it gives a message about imread being depreciated I will use
#skimage which also gives a numpy array. 

from skimage import io
img = io.imread("images/monkey.jpg")
print(type(img))  #numpy array

from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

img = img_as_ubyte(io.imread("images/monkey.jpg", as_gray=True))
#img_as_ubyte converts image to 8 bit unsigned int.


print(type(img))
print(img.shape, img.dtype)
#plt.imshow(img) 

#individual pixel values
print(img[0,0])   #reports pixel value at 0,0. Remove img_as_ubte and see the value.
#also make as_grey=True and see the above values

#pixel values from a slice
print(img[10:15, 20:25])  #Values from a slice

mean_grey = img.mean()
max_value = img.max()
min_value = img.min()
print(mean_grey, min_value, max_value)
plt.imshow(img)


#geometric transformation
#flipped

flipped_img_LR = np.fliplr(img)
flipped_img_UD = np.flipud(img)

plt.subplot(2,1,1)
plt.imshow(img, cmap="Greys")
plt.subplot(2,2,3)
plt.imshow(flipped_img_LR, cmap="Blues")
plt.subplot(2,2,4)
plt.imshow(flipped_img_UD, cmap="hsv")

#For all other options: https://matplotlib.org/tutorials/colors/colormaps.html

#Rotation
rotated_img = ndimage.rotate(img, 45)
plt.imshow(rotated_img)
rotated_img_noreshape = ndimage.rotate(img, 45, reshape=False)
plt.imshow(rotated_img_noreshape)


###################3
#Filtering
#Local filters: replace the value of pixels by a function of the values of neighboring pixels.

from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

img = img_as_ubyte(io.imread("images/nucleiTubolin_small_noisy.jpg", as_gray=True))
img1 = img_as_ubyte(io.imread("images/test_image.jpg", as_gray=True))
img2 = img_as_ubyte(io.imread("images/test_images/aeroplane/1.jpg", as_gray=False))

uniform_filtered_img = ndimage.uniform_filter(img, size=9)
#plt.imshow(uniform_filtered_img)

#Gaussian filter: from scipy.ndimage
# Gaussian filter smooths noise but also edges

blurred_img = ndimage.gaussian_filter(img, sigma=3)  #also try 5, 7
#plt.imshow(blurred_img)

#Median filter is better than gaussian. A non-local means is even better
median_img = ndimage.median_filter(img, 3)
#plt.imshow(median_img)

#Edge detection
sobel_img = ndimage.sobel(img2, axis=0)  #Axis along which to calculate sobel
#plt.imshow(sobel_img)


#for a list of all filters
#https://docs.scipy.org/doc/scipy/reference/ndimage.html









