#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/XbCK46n7U80

"""
This code demonstrates potential applications in image processing by manipulating DCT.
In this example, we perform averaging of images in real and DCT space. 
Since DCT is linear, an average of images in spatial domain would be identical to performing
DCT on each image and averaging the DCT and then inverse DCT to bring back the original image.

DCT is similar to DFT. DFT uses a set of complex exponential functions, 
while the DCT uses only (real-valued) cosine functions.

DCT is used in data compression applications such as JPEG.This is becasue DCT representation
has more of the information concentrated in small number of coefficients. This is useful for
compression algorithms because you can approximate original information in a relatively small set
of DCT coefficients. 
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.fftpack import dct, idct

from os import listdir

image_dir = "C:/PfM/Python_files/New_stuff/noisy_images/25_sigma/"  # Path to image directory

# Load images
filenames = listdir(image_dir)
filenames.sort()

imgs = []
for f in filenames:
	imgs.append((cv2.imread(image_dir + f, 0)).astype(np.float32))


height, width = imgs[0].shape

# Apply the weighted average to images and corresponding DCT images, respectively. 
avg_img = np.zeros([height, width], np.float32)
dct_avg_img = np.zeros([height, width], np.float32)

for i in range(len(imgs)):
    avg_img = cv2.addWeighted(avg_img, i/(i+1.0), imgs[i], 1/(i+1.0), 0) #Original image average
    dct_avg_img = cv2.addWeighted(dct_avg_img, i/(i+1.0), dct(imgs[i]), 1/(i+1.0), 0) #DCT average


reverse_img = idct(dct_avg_img) #Convert averaged DCT back to real space. 


plt.imsave("noisy_images/00-dct_averaged_img.jpg", reverse_img, cmap="gray")
plt.imsave("noisy_images/00-averaged_img.jpg", avg_img, cmap="gray")


fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(imgs[0], cmap='gray')
ax1.title.set_text('Input Image 1')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(imgs[1], cmap='gray')
ax2.title.set_text('Input Image 2')

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(avg_img, cmap='gray')
ax3.title.set_text('Average of Images')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(reverse_img, cmap='gray')
ax4.title.set_text('Image from DCT average')
plt.show()