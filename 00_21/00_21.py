#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=jcUx-TQpcM8

#Scratch Assay on time series images
#https://pmc.ncbi.nlm.nih.gov/articles/PMC5154238/pdf/kcam-08-05-969641.pdf
#In paper it talks about artificially creating a wound by scratching
#a surface of monolayer of the cell and eventually it heals and the scratch
#gets filled (quantifying the wound area as how fast the wound is healing)
# We segment the wound area against healed area after repeating it for multiple
#imgs, we will get a plot. 
#

import matplotlib.pyplot as plt #to plot
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from skimage.filters import threshold_otsu

img = io.imread(r"..\images\scratch.jpg") #importing img
plt.imshow(img)  
entropy_img = entropy(img, disk(3))
plt.imshow(entropy_img)
thresh = threshold_otsu(entropy_img)
print(thresh)
binary = entropy_img <= thresh
plt.imshow(binary)
print(np.sum(binary == True))

#yellow=1 (clean), purple=0

#img has scratch and healed region: can't use threshold filter as pixel value same throughout 
#use entropy filter (clean area will have low entropy)
#####################################################################################################################

import matplotlib.pyplot as plt #to plot
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from skimage.filters import threshold_otsu
import glob #helps in walk through the folder
time = 0
time_list=[]
area_list=[]
#path = r"..\images\scratch_assay*.*"
path = r"..\images\scratch_assay*.jpg"
for file in glob.glob(path): #assign file name
    dict={}
    img=io.imread(file) #read the file and then calculate entropy
#use entropy filter (clean area will have low entropy)
    entropy_img = entropy(img, disk(6))
#otsu filter from skimage, calculate entropy of img
    thresh = threshold_otsu(entropy_img)
    binary = entropy_img <= thresh
    print(np.sum(binary == True))
    time += 1
    print("time=", time, "hr  ", "Scratch area=", scratch_area, "pix\N{SUPERSCRIPT TWO}")
    time_list.append(time)
    area_list.append(scratch_area)
    time += 1


import glob
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
import os

path = r"..\images\scratch_assay\*.jpg"
files = glob.glob(path)
print("Found files:", files)

time = 0

for file in files:
    print(f"\nProcessing: {file}")
    img = io.imread(file)

    # Convert to grayscale if image is RGB
    if img.ndim == 3:
        img = rgb2gray(img)

    entropy_img = entropy((img * 255).astype(np.uint8), disk(6))
    thresh = threshold_otsu(entropy_img)
    binary = entropy_img <= thresh
    print(f"Time {time} - Binary sum:", np.sum(binary))

    plt.imshow(binary, cmap='gray')
    plt.title(f'Time {time}')
    plt.axis('off')
    plt.show()

    time += 1

#print(time_list, area_list)
plt.plot(time_list, area_list, 'bo')  #Print blue dots scatter plot

#Print slope, intercept
from scipy.stats import linregress
#print(linregress(time_list, area_list))


slope, intercept, r_value, p_value, std_err = linregress(time_list, area_list)
print("y = ",slope, "x", " + ", intercept  )
print("R\N{SUPERSCRIPT TWO} = ", r_value**2)
#print("r-squared: %f" % r_value**2)

