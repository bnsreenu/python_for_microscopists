# https://youtu.be/IGnIRp5dW_c
"""
Prediction followed by watershed.
"""

import segmentation_models as sm

#from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model


# ########################################################################
# ###Model 1
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

#model = load_model('saved_models/mito_res34_backbone_100epochs_with_border.hdf5', compile=False)
model = load_model('saved_models/from_colab/mito_res34_backbone_100epochs_with_border_1200images.hdf5', compile=False)

##############################################

#Test some random single small images

test_img = cv2.imread('data/test_images/img242.tif', 1)

test_img_input=np.expand_dims(test_img, 0)

test_img_input1 = preprocess_input(test_img_input)

test_pred1 = model.predict(test_img_input1)
test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.title('Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(222)
plt.title('TPrediction')
plt.imshow(test_prediction1[:,:], cmap='gray')
plt.show()
######################################################

#Watershed
from skimage import measure, color

sure_bg = (test_prediction1 == 0).astype(np.uint8)
plt.imshow(sure_bg, cmap='gray')
print(np.unique(sure_bg))  #Background will be pixel value 1

sure_fg = (test_prediction1 == 2).astype(np.uint8)
plt.imshow(sure_fg, cmap='gray')
print(np.unique(sure_fg))  #Foreground will be pixel value 1. Need to change so we can separate from background
sure_fg[sure_fg==1]=2  #Replace values 1 with 2.


unknown = (test_prediction1 == 1).astype(np.uint8)
plt.imshow(unknown, cmap='gray')
print(np.unique(unknown))

#Now we create a marker and label the regions inside. 
# For sure regions, both foreground and background will be labeled with positive numbers.
# Unknown regions will be labeled 0. 

#For markers let us use ConnectedComponents. 
ret3, markers = cv2.connectedComponents(sure_fg)
plt.imshow(markers)
print(np.unique(markers))


#One problem rightnow is that the entire background pixels is given value 0.
#This means watershed considers this region as unknown.
#So let us add 10 to all labels so that sure background is not 0, but 10
markers = markers+10

# Now, mark the region of unknown with zero
markers[unknown==1] = 0
plt.imshow(markers, cmap='gray')   #Look at the 3 distinct regions.


#Now we are ready for watershed filling. 
markers = cv2.watershed(test_img, markers)
plt.imshow(markers, cmap='gray')
#The boundary region will be marked -1
#https://docs.opencv.org/3.3.1/d7/d1b/group__imgproc__misc.html#ga3267243e4d3f95165d55a618c65ac6e1

#Let us color boundaries in yellow. 
#test_prediction1[markers == -1] = [0,255,255]  

img2 = color.label2rgb(markers, bg_label=0)

cv2.imshow('Overlay on original image', test_img)
cv2.imshow('Colored Grains', img2)
cv2.waitKey(0)


#Now, time to extract properties of detected cells
# regionprops function in skimage measure module calculates useful parameters for each object.

props = measure.regionprops_table(markers, intensity_image=test_img, 
                              properties=['label',
                                          'area', 'equivalent_diameter',
                                          'mean_intensity', 'solidity'])
    
import pandas as pd
df = pd.DataFrame(props)
df = df[df.equivalent_diameter > 20]  #Remove background or other regions that may be counted as objects
df = df[df.equivalent_diameter < 100]

print(df.head())



##################################
