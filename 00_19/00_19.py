#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=s_hDL2fGvow&t=

#for a list of all filters
#https://docs.scipy.org/doc/scipy/reference/ndimage.html

"""
#Image processing using Scipy
Scipy is a python library that is part of numpy stack. 
It contains modules for linear algebra, FFT, signal processing and
image processing. Not designed for image processing but has a few tools

"""

#You can use imread from scipy to read images

  #numpy array

#since it gives a message about imread being depreciated I will use
#skimage which also gives a numpy array. 

from skimage import io

img = io.imread(r"python_for_microscopists\images\monkey.jpg")
print(type(img))  # should print: <class 'numpy.ndarray'>


from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

