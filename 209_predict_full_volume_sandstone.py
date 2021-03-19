# https://youtu.be/q-p8v1Bxvac

"""
Author: Dr. Sreenivas Bhattiprolu

Multiclass semantic segmentation using U-Net - prediction on large images
and 3D volumes (slice by slice)

To annotate images and generate labels, you can use APEER (for free):
www.apeer.com 
"""

from simple_multi_unet_model import multi_unet_model #Uses softmax (From video 208)

from keras.utils import normalize
import os
import glob
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify


n_classes=4 #Number of classes for segmentation
IMG_HEIGHT = 128
IMG_WIDTH  = 128
IMG_CHANNELS = 1

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.load_weights('sandstone_50_epochs_catXentropy_acc.hdf5')  
#model.load_weights('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')  


segm_images = []
path = "all_images/*.tif"
from pathlib import Path
for file in glob.glob(path):
    #print(file)     #just stop here to see all file names printed
    name = Path(file).stem #Get the original file name
    #print(name)
  
    large_image = cv2.imread(file, 0)
    
    patches = patchify(large_image, (128, 128), step=128)  #Step=256 for 256 patches means no overlap
    
    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            #print(i,j)
            
            single_patch = patches[i,j,:,:]
            
            single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            single_patch_input=np.expand_dims(single_patch_norm, 0)
    
            single_patch_prediction = (model.predict(single_patch_input))
            single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]
    
            predicted_patches.append(single_patch_predicted_img)
    
    predicted_patches = np.array(predicted_patches)
    
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 128,128) )
    
    reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
    #Here you can save individual images, or save on milti_dim tiff file
    #cv2.imwrite('segmented_images/' + name + '_segmented.tif', reconstructed_image)
    segm_images.append(reconstructed_image)
    print("Finished segmenting image: ", name)
    
    
final_segm_image = np.array(segm_images).astype(np.uint8)   

from tifffile import imsave
imsave('segmented_images/sandstone_segmented_test.tif', final_segm_image)
    

    
    
    
    
    
    
    
    
    
    
    