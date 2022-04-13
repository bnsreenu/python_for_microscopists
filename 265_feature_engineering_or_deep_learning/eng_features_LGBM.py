# https://youtu.be/JNjhKrwe-4k
"""
Semantic segmentation using feature extraction and traditional classifier (LGBM)
Also, try XGBoost, Random Forest, SVM, etc. 

For Light GBM
pip install lightgbm

For XGBoost
pip install xgboost  

#Last tested on python 3.7.9
"""

import numpy as np
import cv2
import pandas as pd
import pickle
import tifffile as tiff

from matplotlib import pyplot as plt

############################################################################
def feature_extraction(img):
    df = pd.DataFrame()


#All features generated must match the way features are generated for TRAINING.
#Feature1 is our original image pixels
    img2 = img.reshape(-1)
    df['Original Image'] = img2

#Geerate FEATURES and add them to the data frame
#Feature 3 is canny edge
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    from skimage.filters import roberts, sobel, scharr, prewitt

#Feature 4 is Roberts edge
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

#Feature 5 is Sobel
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

#Feature 6 is Scharr
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    #Feature 7 is Prewitt
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    #Feature 8 is Gaussian with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    #Feature 9 is Gaussian with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    #Feature 10 is Median with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1


    return df
################################################################


img = tiff.imread('data/train_images/Sandstone_Versa0000.tif')
mask = tiff.imread('data/train_images/Sandstone_Versa0000_mask.tif')

plt.imshow(img, cmap='gray')

plt.imshow(mask, cmap='gray')

df = feature_extraction(img)

mask = mask.reshape(-1)
df['Labels'] = mask

print(df.head())

#Define the dependent variable that needs to be predicted (labels)
Y = df["Labels"].values

#Define the independent variables
X = df.drop(labels = ["Labels"], axis=1) 

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)
##################################################################

import lightgbm as lgb

# https://lightgbm.readthedocs.io/en/latest/Parameters.html
#For boosting type use 'gbdt' or 'dart'
d_train = lgb.Dataset(X_train, label=y_train)

lgbm_params = {'learning_rate':0.05, 'boosting_type':'dart',    #Try dart for better accuracy
              'objective':'multiclass',
              'metric': 'multi_logloss',
              'num_leaves':90,
              'max_depth':9,
              'num_class':4}  #no.of unique values in the target class not inclusive of the end value

model = lgb.train(lgbm_params, d_train, 100) #50 iterations. Increase iterations for small learning rates
#Save model for future use
filename = 'saved_models/model_LGBM.sav'
pickle.dump(model, open(filename, 'wb'))
model = pickle.load(open(filename, 'rb'))

#Model predicts probabilities. Need to convert these to classes.
prediction_lgb=model.predict(X_test)
prediction_lgbm = np.array([np.argmax(i) for i in prediction_lgb])

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_lgbm))

from sklearn.metrics import confusion_matrix
#Get the confusion matrix
cm = confusion_matrix(y_test, prediction_lgbm)
class2_acc = cm[1,1]/(cm[1,1] + cm[1,0] + cm[1,2] + cm[1,3] + cm[0,1]+ cm[2,1]+ cm[3,1])
print("Class 2 accuracy is: ", class2_acc)

from lightgbm import plot_importance
plot_importance(model, ax=None, height=0.2, xlim=None, ylim=None, 
                title='Feature importance', xlabel='Feature importance', 
                ylabel='Features', importance_type='split', 
                max_num_features=None, ignore_zero=True, 
                grid=True, precision=3)


####################################################
#Applying trained model to segment multiple files. 
test_image_stack = tiff.imread('data/test_images/test_sandstone_images.tif')
test_mask_stack = tiff.imread('data/test_images/test_sandstone_masks.tif')

from matplotlib import pyplot as plt

filename = "saved_models/model_LGBM.sav"
loaded_model = pickle.load(open(filename, 'rb'))

segmented_stack=[]
for i in range(test_image_stack.shape[0]):
    print("Now segmenting image number: ", i)     #just stop here to see all file names printed
    test_img = test_image_stack[i,:,:]

#Call the feature extraction function.
    X = feature_extraction(test_img)
    prediction = loaded_model.predict(X)
    result = np.array([np.argmax(i) for i in prediction])
    segmented = result.reshape((test_img.shape))
    segmented_stack.append(segmented)

segmented_stack = np.array(segmented_stack)
output_stack = segmented_stack.astype(np.uint8)


tiff.imwrite('data/lgbm_segmented.tif', output_stack)
####################################################
#Predict on a few images
#Load images and masks if not already done... 
test_image_stack = tiff.imread('data/test_images/test_sandstone_images.tif')
test_mask_stack = tiff.imread('data/test_images/test_sandstone_masks.tif')
segmented_stack = tiff.imread('data/lgbm_segmented.tif')

#Using built in keras function
from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_mask_stack, segmented_stack)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

plt.imshow(test_image_stack[0, :,:], cmap='gray')
plt.imshow(segmented_stack[0], cmap='gray')
#######################################################################


import random
test_img_number = random.randint(0, test_mask_stack.shape[0]-1)
#test_img_number=9
test_img = test_image_stack[test_img_number]
ground_truth=test_mask_stack[test_img_number]
segmented_img = segmented_stack[test_img_number]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(segmented_img, cmap='jet')
plt.show()