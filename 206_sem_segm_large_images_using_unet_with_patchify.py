# https://youtu.be/LM9yisNYfyw

"""
Applying trained model to large images by dividing them into smaller patches
using patchify.

By loading pre-augmented or patched images and masks.
Use save_patches.py to save patches generated into local drive

This code uses 256x256 images/masks.

pip install patchify
"""
from simple_unet_model import simple_unet_model 
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify


def get_model():
    return simple_unet_model(256, 256, 1)

model = get_model()

#model.load_weights('mitochondria_gpu_tf1.4.hdf5')
model.load_weights('mitochondria_50_plus_100_epochs.hdf5')
#Apply a trained model on large image
large_image = cv2.imread('data/01-1.tif', 0)
#This will split the image into small images of shape [3,3]
patches = patchify(large_image, (256, 256), step=64)  #Step=256 for 256 patches means no overlap

predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        print(i,j)
        
        single_patch = patches[i,j,:,:]
        single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
        single_patch_input=np.expand_dims(single_patch_norm, 0)

#Predict and threshold for values above 0.5 probability
        single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
        predicted_patches.append(single_patch_prediction)

predicted_patches = np.array(predicted_patches)

predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256,256) )
reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
plt.imshow(reconstructed_image, cmap='gray')
plt.imsave('data/results/segm.jpg', reconstructed_image, cmap='gray')

plt.hist(reconstructed_image.flatten())  #Threshold everything above 0

# final_prediction = (reconstructed_image > 0.01).astype(np.uint8)
# plt.imshow(final_prediction)

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(large_image, cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(reconstructed_image, cmap='gray')
plt.show()

#plt.imsave('input.jpg', test_img[:,:,0], cmap='gray')
#plt.imsave('data/results/output2.jpg', reconstructed_image, cmap='gray')

##################################
#Watershed to convert semantic to instance
#########################
from skimage import measure, color, io

#Watershed
img = cv2.imread('data/results/segm.jpg')  #Read as color (3 channels)
img_grey = img[:,:,0]

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(img_grey,cv2.MORPH_OPEN,kernel, iterations = 2)

sure_bg = cv2.dilate(opening,kernel,iterations=10)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

ret2, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

ret3, markers = cv2.connectedComponents(sure_fg)
markers = markers+10

markers[unknown==255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [0,255,255]  

img2 = color.label2rgb(markers, bg_label=0)

cv2.imshow('Overlay on original image', large_image)
cv2.imshow('Colored Grains', img2)
cv2.waitKey(0)


props = measure.regionprops_table(markers, intensity_image=img_grey, 
                              properties=['label',
                                          'area', 'equivalent_diameter',
                                          'mean_intensity', 'solidity'])
    
import pandas as pd
df = pd.DataFrame(props)
df = df[df.mean_intensity > 100]  #Remove background or other regions that may be counted as objects
   
print(df.head())











