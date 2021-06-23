# https://youtu.be/xlC385lEzIY
"""

@author: Sreenivas

# TTA - Should be called prediction time augmentation
#We can augment each input image, predict augmented images and average all predictions


"""


import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import random

model = tf.keras.models.load_model("mitochondria_load_from_disk_focal_dice_50epochs.hdf5", compile=False)


image_directory = 'data2/test_images/test/'
mask_directory = 'data2/test_masks/test/'


SIZE = 256
image_dataset = []  
mask_dataset = []  

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'tif'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'tif'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))


#
image_dataset = np.array(image_dataset) / 255.

#D not normalize masks, just rescale to 0 to 1.
mask_dataset = (np.array(mask_dataset)) /255.


#Demonstrate TTP on single image 
n = random.randint(0, mask_dataset.shape[0])
temp_test_img = image_dataset[n,:,:,:]
temp_test_img = image_dataset[n,:,:,:]
temp_mask = mask_dataset[n,:,:]

p0 = model.predict(np.expand_dims(temp_test_img, axis=0))[0][:, :, 0]

p1 = model.predict(np.expand_dims(np.fliplr(temp_test_img), axis=0))[0][:, :, 0]
p1 = np.fliplr(p1)

p2 = model.predict(np.expand_dims(np.flipud(temp_test_img), axis=0))[0][:, :, 0]
p2 = np.flipud(p2)

p3 = model.predict(np.expand_dims(np.fliplr(np.flipud(temp_test_img)), axis=0))[0][:, :, 0]
p3 = np.fliplr(np.flipud(p3))

thresh = 0.3
p = (((p0 + p1 + p2 + p3) / 4) > thresh).astype(np.uint8)
 
plt.figure(figsize=(12, 12))
plt.subplot(231)
plt.title('Original mask')
plt.imshow(temp_mask, cmap='gray')
plt.subplot(232)
plt.title('Prediction No Aug')
plt.imshow(p0>thresh, cmap='gray')
plt.subplot(233)
plt.title('Prediction LR')
plt.imshow(p1>thresh, cmap='gray')
plt.subplot(234)
plt.title('Prediction UD')
plt.imshow(p2>thresh, cmap='gray')
plt.subplot(235)
plt.title('Prediction LR and UD')
plt.imshow(p3>thresh, cmap='gray')
plt.subplot(236)
plt.title('Average Prediction')
plt.imshow(p>thresh, cmap='gray')
plt.show()



#Now that we know the transformations are working, let us extend to all predictions
predictions = []
for image in image_dataset:
    
    pred_original = model.predict(np.expand_dims(image, axis=0))[0][:, :, 0]
    
    pred_lr = model.predict(np.expand_dims(np.fliplr(image), axis=0))[0][:, :, 0]
    pred_lr = np.fliplr(pred_lr)
    
    pred_ud = model.predict(np.expand_dims(np.flipud(image), axis=0))[0][:, :, 0]
    pred_ud = np.flipud(pred_ud)
    
    pred_lr_ud = model.predict(np.expand_dims(np.fliplr(np.flipud(image)), axis=0))[0][:, :, 0]
    pred_lr_ud = np.fliplr(np.flipud(pred_lr_ud))
    
    preds = (pred_original + pred_lr + pred_ud + pred_lr_ud) / 4
    
    predictions.append(preds)


predictions = np.array(predictions)

threshold = 0.5
predictions_th = predictions > threshold

import random
test_img_number = random.randint(0, mask_dataset.shape[0]-1)
test_img = image_dataset[test_img_number]
ground_truth=mask_dataset[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = predictions_th[test_img_number]

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth, cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()