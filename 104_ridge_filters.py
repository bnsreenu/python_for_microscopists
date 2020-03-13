#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/Fdhn5_gT1wY

"""
Ridge filters can be used to detect ridge-like structures, such as neurites, tubes, vessels, wrinkles.
The present class of ridge filters relies on the eigenvalues of the Hessian matrix of image intensities to detect ridge
structures where the intensity changes perpendicular but not along the structure.

"""



from skimage import io, filters, feature
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2

#Ridge operators 
#https://scikit-image.org/docs/dev/auto_examples/edges/plot_ridge_filter.html#sphx-glr-auto-examples-edges-plot-ridge-filter-py
from skimage.filters import meijering, sato, frangi, hessian


img = io.imread("images/leaf.jpg")
img = rgb2gray(img)

#sharpened = unsharp_mask(image0, radius=1.0, amount=1.0)
meijering_img = meijering(img)
sato_img = sato(img)
frangi_img = frangi(img)
hessian_img = hessian(img)

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(meijering_img, cmap='gray')
ax2.title.set_text('Meijering')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(sato_img, cmap='gray')
ax3.title.set_text('Sato')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(frangi_img, cmap='gray')
ax4.title.set_text('Frangi')
plt.show()