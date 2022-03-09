# https://youtu.be/UUP_omOSKuc
"""
For labeling your images using Label Studio:
https://labelstud.io/

Let us work in Anaconda command prompt. (You can use other command prompts)
Check environments: conda env list

Create a new environment to install Label Studio
conda create --name give_some_name pip
(Need to specify pip as a dependency, otherwise it will not be available)

(To specify python version for your env..)
conda create -n give_some_name python=3.7

Now activate the env.
conda activate give_some_name

# Install the Label Studio package
pip install -U label-studio

# Launch it!
label-studio

Open your browser and go to the URL displayed on your screen, typically
http://0.0.0.0:8080/

###################################

Mask/annotation handling after saving them from Label Studio
(Semantic segmentation)

If you have multiple masks in a folder for a single image where each mask 
represents a specific class. It makes sense to load all masks for a given
class into a single numpy array. You can save these combined images for further
use in your machine learning exercises. 

Here, we read files containing a specific string, load them into python and
perform some preprocessing before combining all masks from a specific class 
into a single numpy array. 

Our Labels = Houses, Roads, Water
"""
import os
from skimage import io
import numpy as np
from matplotlib import pyplot as plt


#Loading a png mask image for inspection
test_mask_png = io.imread("labels_as_png/task-9-annotation-6-by-1-tag-Roads-0.png")
plt.imshow(test_mask_png, cmap='gray')
print(np.unique(test_mask_png))  #This is not a true binary image.

#Let us load a numpy array saved from Label Studio
test_mask_np = np.load("labels_as_numpy/task-9-annotation-6-by-1-tag-Roads-0.npy")
plt.imshow(test_mask_np, cmap='gray')
print(np.unique(test_mask_np)) #This is not a true binary image.

#Need to binarize the image. Simple thresholding for values above 0. 
#Convert all values above 0 to 1 to assign a pixel value of 1 for the Houses class.
#Similarly convert other values for other classes to 2, 3, etc. 
my_mask = np.where(test_mask_png>0, 1, test_mask_png)
print(np.unique(my_mask))
plt.imshow(my_mask, cmap='gray')

#Now, let us read images from all classes and change pixel values to 1, 2, 3, ...
#You can also combine them into a single image (numpy array) for simple handling in future
#(Changing pixel values is optional if you do not intend to combine them into a single array)
#It is better to keep them separate, especially for multilabel segmentation
#where classes can overlap. 

label_folder = "labels_as_png/"
houses_masks = []
roads_masks = []
water_masks = []

all_masks=[]

for filename in os.listdir(label_folder):
    #print(filename)
    if "Houses" in filename:
        print(filename)
        houses_mask = io.imread(label_folder + filename)
        houses_mask = np.where(houses_mask>0, 1, houses_mask)
        houses_masks.append(houses_mask)
    elif "Roads" in filename:
        print(filename)
        roads_mask = io.imread(label_folder + filename)
        roads_mask = np.where(roads_mask>0, 2, roads_mask)
        roads_masks.append(roads_mask)
    elif "Water" in filename:
        print(filename)
        water_mask = io.imread(label_folder + filename)
        water_mask = np.where(water_mask>0, 3, water_mask)
        water_masks.append(water_mask)
    


#Now, convert the list to array and proceed with your work.
#NOTE that you need to resize masks (or crop) to same size to combine them
#into numpy arrays. You need to resize both input images and masks exactly the 
#same way.