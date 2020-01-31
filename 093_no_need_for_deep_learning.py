#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=5V6Dg1-PuqA



"""
Colorizing images using traditional means.
While deep learning help swith natural images, for microscopy images we
don't need to get skin tones and sky color correct. So follow easier methods.
"""

#Pillow colorize module to define black and white points. Simplest way.
# importing image object from PIL 
from PIL import Image, ImageOps 
  
# creating an image object 
img = Image.open(r"BSE.jpg").convert("L") 
  
# image colorize function 
img1 = ImageOps.colorize(img, black ="blue", white ="red") 
img1.show() 

######################################
#Opencv color maps

import cv2

"""
COLORMAP_JET
COLORMAP_RAINBOW
COLORMAP_SPRING
COLORMAP_SUMMER
COLORMAP_HSV
"""

grey_img = cv2.imread("BSE.jpg", 0)
color_img = cv2.applyColorMap(grey_img, cv2.COLORMAP_JET)

cv2.imshow("Color image", color_img)
cv2.waitKey()
cv2.destroyAllWindows()
########################################

"""
DENOISING
Non-local means


"""

#Gaussian
from skimage import img_as_float
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd

img = img_as_float(io.imread("images/BSE_noisy.jpg"))
#Need to convert to float as we will be doing math on the array

gaussian_img = nd.gaussian_filter(img, sigma=3)
plt.imshow(gaussian_img, cmap='gray')


#Anisotropic
import matplotlib.pyplot as plt
import cv2
from skimage import io
from medpy.filter.smoothing import anisotropic_diffusion

img = io.imread("BSE_noisy.jpg", as_gray=True)

img_filtered = anisotropic_diffusion(img, niter=5, kappa=50, gamma=0.1, option=2) #For more input parameters check the above link
plt.imshow(img_filtered, cmap='gray')
plt.imsave("anisotropic_result.jpg", img_filtered, cmap='gray')



###############################################################

#SEGMENTATION
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from skimage.filters import threshold_otsu

img=io.imread("images/scratch.jpg")
entropy_img = entropy(img, disk(3))
thresh = threshold_otsu(entropy_img)
binary = entropy_img <= thresh
plt.imshow(binary)





###

import matplotlib.pyplot as plt
import numpy as np

from skimage import io, img_as_ubyte
from skimage.filters import threshold_multiotsu


# The input image.
image = io.imread("images/BSE.jpg")

# Applying multi-Otsu threshold for the default value, generating
# three classes.
thresholds = threshold_multiotsu(image, classes=4)

# Using the threshold values, we generate the three regions.
regions = np.digitize(image, bins=thresholds)
output = img_as_ubyte(regions)

plt.imshow(output)
############################################################









