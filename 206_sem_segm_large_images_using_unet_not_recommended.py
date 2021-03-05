# https://youtu.be/LM9yisNYfyw

"""
Applying trained model to large images. Not recommended, this code is intended for
educational purposes only so we can see how we can get crappy results if we are not careful.

"""

from simple_unet_model import simple_unet_model   #Use normal unet model
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


#You can import model without defining input dimensions but not recommended. 
#Models work best when it segments objects of similar size as they got trained.
# def get_model():
#     return simple_unet_model(None, None, 1)

#The right way to import model. In this case it got trained on 256x256 images
def get_model():
    return simple_unet_model(256, 256, 1)

model = get_model()
#######################################################################
#Predict on a few images

## mitochondria_50_plus_100_epochs.hdf5 Model got trained on 256x256 images for 50 epochs
#But weights can be applied to larger images if we re-import model without 
# defining input dimensions for the input layer.
#Beware that this is not recommended for semantic segmentation as we will soon find out...
model.load_weights('mitochondria_50_plus_100_epochs.hdf5')  


#Apply the trained model on large image
large_image = cv2.imread('data/01-1.tif', 0)

#Resize - DO NOT do this mistake
# large_image = Image.fromarray(large_image)
# large_image = large_image.resize((256, 256))
# large_image = np.array(large_image)

large_image_norm = np.expand_dims(normalize(np.array(large_image), axis=1),2)
large_image_input=np.expand_dims(large_image_norm, 0)

#Predict and threshold for values above 0.5 probability
large_image_prediction = (model.predict(large_image_input)[0,:,:,0] > 0.5).astype(np.uint8)


plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('External Image')
plt.imshow(large_image, cmap='gray')
plt.subplot(222)
plt.title('Prediction of external Image')
plt.imshow(large_image_prediction, cmap='gray')
plt.show()

#plt.imsave('data/results/output2.jpg', reconstructed_image, cmap='gray')











