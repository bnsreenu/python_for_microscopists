# https://youtu.be/0W6MKZqSke8
"""
Author: Dr. Sreenivas Bhattiprolu 

Prediction using smooth tiling as descibed here...

https://github.com/Vooban/Smoothly-Blend-Image-Patches


"""

import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
import segmentation_models as sm

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from smooth_tiled_predictions import predict_img_with_smooth_windowing

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

img = cv2.imread("data/images/N-34-66-C-c-4-3.tif")  #N-34-66-C-c-4-3.tif, N-34-97-D-c-2-4.tif
input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
input_img = preprocess_input(input_img)

original_mask = cv2.imread("data/masks/N-34-66-C-c-4-3.tif")
original_mask = original_mask[:,:,0]  #Use only single channel...
#original_mask = to_categorical(original_mask, num_classes=n_classes)

from keras.models import load_model
model = load_model("landcover_25_epochs_RESNET_backbone_batch16.hdf5", compile=False)
                  
# size of patches
patch_size = 256

# Number of classes 
n_classes = 4

         
###################################################################################
#Predict using smooth blending

# Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
# Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
predictions_smooth = predict_img_with_smooth_windowing(
    input_img,
    window_size=patch_size,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=n_classes,
    pred_func=(
        lambda img_batch_subdiv: model.predict((img_batch_subdiv))
    )
)


final_prediction = np.argmax(predictions_smooth, axis=2)

#Save prediction and original mask for comparison
plt.imsave('data/test_images/N-34-66-C-c-4-3.tif_segmented.jpg', final_prediction)
plt.imsave('data/test_images/N-34-66-C-c-4-3.tif_mask.jpg', original_mask)
###################


plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Testing Image')
plt.imshow(img)
plt.subplot(222)
plt.title('Testing Label')
plt.imshow(original_mask)
plt.subplot(223)
plt.title('Prediction with smooth blending')
plt.imshow(final_prediction)
plt.show()

#############################
