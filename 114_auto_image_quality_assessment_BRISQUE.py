#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/Vj-YcXswTek

"""
BRISQUE calculates the no-reference image quality score for an image using the 
Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE). 

BRISQUE score is computed using a support vector regression (SVR) model trained on an 
image database with corresponding differential mean opinion score (DMOS) values. 
The database contains images with known distortion such as compression artifacts, 
blurring, and noise, and it contains pristine versions of the distorted images. 
The image to be scored must have at least one of the distortions for which the model was trained.

Mittal, A., A. K. Moorthy, and A. C. Bovik. "No-Reference Image Quality Assessment in the Spatial Domain.
" IEEE Transactions on Image Processing. Vol. 21, Number 12, December 2012, pp. 4695–4708.
https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf

To install imquality
https://pypi.org/project/image-quality/
"""
import numpy as np
from skimage import io, img_as_float
import imquality.brisque as brisque

#img = img_as_float(io.imread('noisy_images/BSE.jpg', as_gray=True))
img = img_as_float(io.imread('noisy_images/BSE_50sigma_noisy.jpg', as_gray=True))

score = brisque.score(img)
print("Brisque score = ", score)


#Now let us check BRISQUE scores for a bunch of blurred images.

img0 = img_as_float(io.imread('noisy_images/BSE.jpg', as_gray=True))
img25 = img_as_float(io.imread('noisy_images/BSE_1sigma_blur.jpg', as_gray=True))
img50 = img_as_float(io.imread('noisy_images/BSE_2sigma_blur.jpg', as_gray=True))
img75 = img_as_float(io.imread('noisy_images/BSE_3sigma_blur.jpg', as_gray=True))
img100 = img_as_float(io.imread('noisy_images/BSE_5sigma_blur.jpg', as_gray=True))
img200 = img_as_float(io.imread('noisy_images/BSE_10sigma_blur.jpg', as_gray=True))


score0 = brisque.score(img0)
score25 = brisque.score(img25)
score50 = brisque.score(img50)
score75 = brisque.score(img75)
score100 = brisque.score(img100)
score200 = brisque.score(img200)

print("BRISQUE Score for 0 blur = ", score0)
print("BRISQUE Score for 1 sigma blur = ", score25)
print("BRISQUE Score for 2 sigma blur = ", score50)
print("BRISQUE Score for 3 sigma blur = ", score75)
print("BRISQUE Score for 5 sigma blur = ", score100)
print("BRISQUE Score for 10 sigma blur = ", score200)


# Peak signal to noise ratio (PSNR) is Not a good metric.

from skimage.metrics import peak_signal_noise_ratio

psnr_25 = peak_signal_noise_ratio(img0, img25)
psnr_50 = peak_signal_noise_ratio(img0, img50)
psnr_75 = peak_signal_noise_ratio(img0, img75)
psnr_100 = peak_signal_noise_ratio(img0, img100)
psnr_200 = peak_signal_noise_ratio(img0, img200)

print("PSNR for 1 sigma blur = ", psnr_25)
print("PSNR for 2 sigma blur = ", psnr_50)
print("PSNR for 3 sigma blur = ", psnr_75)
print("PSNR for 5 sigma blur = ", psnr_100)
print("PSNR for 10 sigma blur = ", psnr_200)

