# https://youtu.be/LM9yisNYfyw
"""
Applying trained model to large images by dividing them into smaller patches.


This code uses 256x256 images/masks.

"""

from simple_unet_model import simple_unet_model 
import cv2
import numpy as np
from keras.utils import normalize
from matplotlib import pyplot as plt

patch_size=256


def get_model():
    return simple_unet_model(256, 256, 1)



def prediction(model, image, patch_size):
    segm_img = np.zeros(image.shape[:2])  #Array with zeros to be filled with segmented values
    patch_num=1
    for i in range(0, image.shape[0], 256):   #Steps of 256
        for j in range(0, image.shape[1], 256):  #Steps of 256
            #print(i, j)
            single_patch = image[i:i+patch_size, j:j+patch_size]
            single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            single_patch_shape = single_patch_norm.shape[:2]
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
            segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
          
            print("Finished processing patch number ", patch_num, " at position ", i,j)
            patch_num+=1
    return segm_img

##########
#Load model and predict
model = get_model()
#model.load_weights('mitochondria_gpu_tf1.4.hdf5')
model.load_weights('mitochondria_50_plus_100_epochs.hdf5')

#Large image
large_image = cv2.imread('data/01-1.tif', 0)
segmented_image = prediction(model, large_image, patch_size)
plt.hist(segmented_image.flatten())  #Threshold everything above 0

plt.imsave('data/results/segm.jpg', segmented_image, cmap='gray')

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(large_image, cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(segmented_image, cmap='gray')
plt.show()

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


