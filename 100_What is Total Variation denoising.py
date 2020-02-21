#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/G39dVoiivZk

"""

Works well for random gaussian noise but not as good for salt and pepper

https://hal.archives-ouvertes.fr/hal-00437581/document
"""
import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_tv_chambolle
from matplotlib import pyplot as plt

img = img_as_float(io.imread('images/BSE_25sigma_noisy.jpg', as_gray=True))


plt.hist(img.flat, bins=100, range=(0,1))  #.flat returns the flattened numpy array (1D)


denoise_img = denoise_tv_chambolle(img, weight=0.1, eps=0.0002, n_iter_max=200, multichannel=False)

"""
denoise_tv_chambolle(image, weight=0.1, eps=0.0002, n_iter_max=200, multichannel=False)
weight: The greater weight, the more denoising (at the expense of fidelity to input).
eps: Relative difference of the value of the cost function that determines the stop criterion. 
n_iter_max: Max number of iterations used for optimization

"""


plt.hist(denoise_img.flat, bins=100, range=(0,1))  #.flat returns the flattened numpy array (1D)


cv2.imshow("Original", img)
cv2.imshow("TV Filtered", denoise_img)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

