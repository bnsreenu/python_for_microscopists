#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/Va4Rwoy1v88

"""

https://scikit-image.org/docs/dev/auto_examples/filters/plot_nonlocal_means.html

Works well for random gaussian noise but not as good for salt and pepper
https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf

The non-local means algorithm replaces the value of a pixel by an average 
of a selection of other pixels values: small patches centered on the other 
pixels are compared to the patch centered on the pixel of interest, and the 
average is performed only for pixels that have patches close to the current patch. 
"""
import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma

img_gaussian_noise = img_as_float(io.imread('images/BSE_25sigma_noisy.jpg', as_gray=True))
img_salt_pepper_noise = img_as_float(io.imread('images/BSE_salt_pepper.jpg', as_gray=True))

img = img_gaussian_noise

sigma_est = np.mean(estimate_sigma(img, multichannel=True))
#sigma_est = 0.1

denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
                               patch_size=5, patch_distance=3, multichannel=False)

"""
When the fast_mode argument is False, a spatial Gaussian weighting is applied 
to the patches when computing patch distances. When fast_mode is True a 
faster algorithm employing uniform spatial weighting on the patches is applied.

Larger h allows more smoothing between disimilar patches.

"""

#denoise_img_as_8byte = img_as_ubyte(denoise_img)



cv2.imshow("Original", img)
cv2.imshow("NLM Filtered", denoise_img)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

