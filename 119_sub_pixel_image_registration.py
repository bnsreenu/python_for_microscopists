#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/Ij6nsrs8NAo


"""
https://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html

The register_translation function uses cross-correlation in Fourier space, 
and also by employing an upsampled matrix-multiplication DFT to achieve subpixel precision
Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, 
“Efficient subpixel image registration algorithms,” Optics Letters 33, 156-158 (2008). 
https://pdfs.semanticscholar.org/b597/8b756bdcad061e3269eafaa69452a0c43e1b.pdf

"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, io
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

image = io.imread("images/BSE.jpg")
offset_image = io.imread("images/BSE_transl.jpg")
# offset image translated by (-17.45, 18.75) in y and x 

# subpixel precision
#Upsample factor 100 = images will be registered to within 1/100th of a pixel.
#Default is 1 which means no upsampling.  
shifted, error, diffphase = register_translation(image, offset_image, 100)
print(f"Detected subpixel offset (y, x): {shifted}")

from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(shifted[0], shifted[1]), mode='constant')
#plt.imshow(corrected_image)

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()