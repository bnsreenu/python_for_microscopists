# https://youtu.be/yqkNslkzLk4

"""
@author: Sreenivas Bhattiprolu

LGBM vs XGBOOST for SEMANTIC SEGMENTATION
by extracting features using VGG16 imagenet pretrained weights.

Note: Annotate images at www.apeer.com to create labels. 

Code last tested on: 
    Tensorflow: 2.2.0
    Keras: 2.3.1
    Python: 3.7

pip install xgboost  

https://lightgbm.readthedocs.io/en/latest/
pip install lightgbm
    
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import pickle
from datetime import datetime 
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from keras.models import Model
#from keras.layers import Conv2D
import os
from keras.applications.vgg16 import VGG16


print(os.listdir("images/"))

#Resizing images is optional, CNNs are ok with large images
SIZE_X = 1024 #Resize images (height  = X, width = Y)
SIZE_Y = 996

#Capture training image info as a list
train_images = []

for directory_path in glob.glob("images/train_images"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        #train_labels.append(label)
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob("images/train_masks"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        train_masks.append(mask)
        #train_labels.append(label)
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)

#Use customary x_train and y_train variables
X_train = train_images
y_train = train_masks
y_train = np.expand_dims(y_train, axis=3) #May not be necessary.. leftover from previous code 


#Load VGG16 model wothout classifier/fully connected layers
#Load imagenet weights that we are going to use as feature generators
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE_X, SIZE_Y, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0

#After the first 2 convolutional layers the image dimension changes. 
#So for easy comparison to Y (labels) let us only take first 2 conv layers
#and create a new model to extract features
#New model with only first 2 conv layers
new_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
new_model.summary()

#Now, let us apply feature extractor to our training data
features=new_model.predict(X_train)

#Plot features to view them
square = 8
ix=1
for _ in range(square):
    for _ in range(square):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features[0,:,:,ix-1], cmap='gray')
        ix +=1
plt.show()

#Reassign 'features' as X to make it easy to follow
X=features
X = X.reshape(-1, X.shape[3])  #Make it compatible for Random Forest and match Y labels

#Reshape Y to match X
Y = y_train.reshape(-1)

#Combine X and Y into a dataframe to make it easy to drop all rows with Y values 0
#In our labels Y values 0 = unlabeled pixels. 
dataset = pd.DataFrame(X)
dataset['Label'] = Y
print(dataset['Label'].unique())
print(dataset['Label'].value_counts())

##If we do not want to include pixels with value 0 
##e.g. Sometimes unlabeled pixels may be given a value 0.
dataset = dataset[dataset['Label'] != 0]

#Redefine X and Y 
X_for_training = dataset.drop(labels = ['Label'], axis=1)
X_for_training = X_for_training.values  #Convert to array
Y_for_training = dataset['Label']
Y_for_training = Y_for_training.values  #Convert to array

########################
#Load validation image and prepare it for prediction
validation_img = cv2.imread('images/test_images/test1_img.tif', cv2.IMREAD_COLOR)       
validation_img = cv2.resize(validation_img, (SIZE_Y, SIZE_X))
validation_img = cv2.cvtColor(validation_img, cv2.COLOR_RGB2BGR)
validation_img = np.expand_dims(validation_img, axis=0)

X_validation_feature = new_model.predict(validation_img)
X_validation_feature = X_validation_feature.reshape(-1, X_validation_feature.shape[3])

#Load corresponding ground truth image (Mask) and reshape it for comparison with prediction
truth = cv2.imread('images/test_images/test1_mask.tif', 0).reshape(-1)
####################################################################################
#Note: You can work with pandas dataframes instead of arrays by please beware that
#xboost drops columns with zero values which creates a mismatch between column names
#for training and future testing datasets. So it is safe to work with arrays. 


#XGBOOST
import xgboost as xgb
xgb_model = xgb.XGBClassifier()

start = datetime.now() 
# Train the model on training data
xgb_model.fit(X_for_training, Y_for_training) 
stop = datetime.now()

#Execution time of the model 
execution_time_xgb = stop-start 
print("XGBoost execution time is: ", execution_time_xgb)


#Save model for future use
# filename = 'model_XG.sav'
# pickle.dump(model, open(filename, 'wb'))

################################################################
#########################################################
#Check accuracy and IoU on validation image

#Load model.... 
#loaded_model = pickle.load(open(filename, 'rb'))


prediction_xgb = xgb_model.predict(X_validation_feature)

#Pixel accuracy - not a good metric for semantic segmentation
#Print overall accuracy

print ("Accuracy = ", metrics.accuracy_score(truth, prediction_xgb))

##############################################

#IOU for each class is..
# IOU = true_positive / (true_positive + false_positive + false_negative).

#Using built in keras function
from keras.metrics import MeanIoU
num_classes = 5
IOU_XGB = MeanIoU(num_classes=num_classes)  
IOU_XGB.update_state(truth, prediction_xgb)
print("Mean IoU for XGBoost = ", IOU_XGB.result().numpy())

#############################################################################

#Light GBM

import lightgbm as lgb
 #Class names for LGBM start at 0 so reassigning labels from 1,2,3,4 to 0,1,2,3
Y_for_training = Y_for_training-1 
d_train = lgb.Dataset(X_for_training, label=Y_for_training)

# https://lightgbm.readthedocs.io/en/latest/Parameters.html
lgbm_params = {'learning_rate':0.05, 'boosting_type':'dart',    #Try dart for better accuracy
              'objective':'multiclass',
              'metric': 'multi_logloss',
              'num_leaves':100,
              'max_depth':10,
              'num_class':4}  #no.of unique values in the target class not inclusive of the end value

start=datetime.now()
lgb_model = lgb.train(lgbm_params, d_train, 50) #50 iterations. Increase iterations for small learning rates
stop=datetime.now()

execution_time_lgbm = stop-start
print("LGBM execution time is: ", execution_time_lgbm)

#Prediction on test data
prediction_lgb=lgb_model.predict(X_validation_feature)

#Model predicts probabilities. Need to convert these to classes.
prediction_lgbm = np.array([np.argmax(i) for i in prediction_lgb])
prediction_lgbm = prediction_lgbm+1  #Change labels back to original to compare against ground truth
prediction_image_lgbm = prediction_lgbm.reshape(mask.shape)
plt.imshow(prediction_image_lgbm, cmap='gray')
print ("Accuracy with LGBM = ", metrics.accuracy_score(truth, prediction_lgbm))

IOU_LGBM = MeanIoU(num_classes=num_classes)  
IOU_LGBM.update_state(truth, prediction_lgbm)
print("Mean IoU for LGBM = ", IOU_LGBM.result().numpy())

##################################################
#SUMMARY
print("################################################")
print("LGBM execution time is: ", execution_time_lgbm)
print("XGBoost execution time is: ", execution_time_xgb)
print("################################################")
print ("Accuracy with LGBM = ", metrics.accuracy_score(prediction_lgbm, truth))
print ("Accuracy with XGBoost= ", metrics.accuracy_score(prediction_xgb, truth))
print("################################################")
print("Mean IoU for LGBM = ", IOU_LGBM.result().numpy())
print("Mean IoU for XGBoost = ", IOU_XGB.result().numpy())
