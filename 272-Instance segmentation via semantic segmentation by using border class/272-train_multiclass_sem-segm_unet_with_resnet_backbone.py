# https://youtu.be/IGnIRp5dW_c
"""
Author: Sreenivas Bhattiprolu

Multiclass semantic segmentation using U-Net

Followed by watershed segmentation.

For the segmentation models library:
https://github.com/qubvel/segmentation_models
"""

import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.utils import normalize
from tensorflow.keras.metrics import MeanIoU

import tensorflow.keras as keras
print(tf.__version__)
print(keras.__version__)


#Resizing images, if needed
SIZE_X = 256
SIZE_Y = 256
n_classes=3 #Number of classes for segmentation

total_images_to_load = 500

#Capture training image info as a list
train_images = []

for directory_path in glob.glob("data/generated_patches/images/"):
    count = 0
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        if count < total_images_to_load:
          my_img_path=directory_path+"img"+str(count)+".tif"
          
          #print("Reading image :", my_img_path)
          img = cv2.imread(my_img_path, 1)       
          #img = cv2.resize(img, (SIZE_Y, SIZE_X))
          train_images.append(img)
        else:
          break
        count+=1
#Convert list to array for machine learning processing        
train_images = np.array(train_images)
print(train_images.shape)


#Capture training image info as a list
train_masks = [] 

for directory_path in glob.glob("data/generated_patches/masks_with_border/"):
    count = 0
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        if count < total_images_to_load:
          my_mask_path=directory_path+"mask"+str(count)+".tif"
          #print("Reading mask :", my_mask_path)
          mask = cv2.imread(my_mask_path, 0)       
          #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
          train_masks.append(mask)
        else:
          break
        count+=1
#Convert list to array for machine learning processing        
train_masks = np.array(train_masks)
print(train_masks.shape)

###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

#################################################
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#Create a subset of data for quick testing
#Picking 20% for testing and remaining for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.2, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

from tensorflow.keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

print(y_train_cat.shape)
print(y_test_cat.shape)

######################################################
#Reused parameters in all models

n_classes=3
activation='softmax'

LR = 0.0001
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


########################################################################
###Model 1
BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train1 = preprocess_input1(X_train)
X_test1 = preprocess_input1(X_test)

# define model
model1 = sm.Unet(BACKBONE1, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model1.compile(optim, total_loss, metrics=metrics)

#model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

print(model1.summary())


history1=model1.fit(X_train1, 
          y_train_cat,
          batch_size=8, 
          epochs=5,
          verbose=1,
          validation_data=(X_test1, y_test_cat))


model1.save('saved_models/mito_res34_backbone_100epochs_with_border.hdf5')
############################################################


##########################################################

###
#plot the training and validation accuracy and loss at each epoch
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history1.history['iou_score']
val_acc = history1.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()

#####################################################

from tensorflow.keras.models import load_model

### FOR NOW LET US FOCUS ON A SINGLE MODEL

#Set compile=False as we are not loading it for training, only for prediction.
model1 = load_model('saved_models/mito_res34_backbone_100epochs_with_border.hdf5', compile=False)


#Test some random images
import random
test_img_number = random.randint(0, len(X_test1))
test_img = X_test1[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)

test_img_input1 = preprocess_input1(test_img_input)

test_pred1 = model1.predict(test_img_input1)
test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction1, cmap='gray')
plt.show()
