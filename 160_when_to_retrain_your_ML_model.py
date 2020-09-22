#https://youtu.be/OINr5AfxycQ

#DOwnload images from: https://www.epfl.ch/labs/cvlab/data/data-em/
#Modify them to simulate 'new' set of images, for example, change contrast or add noise

import numpy as np
import cv2
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt
import tifffile as tiff
from matplotlib import pyplot as plt
from skimage.exposure import equalize_adapthist


def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x


img1 = tiff.imread('images/mito_train_imgs.tif')
set1 = normalize(img1)


img2 = tiff.imread('images/mito_train_imgs_low_contrast.tif')
set2 = normalize(img2)


#Get a subset for faster comparison
#Let us pick first 10 and last 10 images from each set. 
#Including raw images
img1_first10 = img1[:10, :, :]
img1_middle10 = img1[170:180, :, :]

#img1_last10 = img1[-10:, :, :]
img2_first10 = img2[:10, :, :]
img2_middle10 = img2[170:180, :, :]

#img2_last10 = img2[-10:, :, :]
set1_first10 = set1[:10, :, :]
set1_middle10 = set1[170:180, :, :]

#set1_last10 = set1[-10:, :, :]
set2_first10 = set2[:10, :, :]
set2_middle10 = set2[170:180, :, :]

#set2_last10 = set2[-10:, :, :]

#Understand the differences
plt.figure(figsize=(16, 8))
plt.subplot(221)
plt.hist(img1_first10.reshape(-1), bins=40)
plt.subplot(222)
plt.hist(img2_first10.reshape(-1), bins=40)

plt.subplot(223)
plt.hist(set1_first10.reshape(-1), bins=40)
plt.subplot(224)
plt.hist(set2_middle10.reshape(-1), bins=40)
plt.show()


##########################################################
#KS statistic is not a good metric. It even fails for the first 10 and last 10 images from same dataset.
#Same for normalized or unnormalized values.

from scipy.stats import ks_2samp  # Calculate the T-test for the means of two independent samples of scores.
print('ks_2samp for img1 top 10 vs itself =  = ', ks_2samp(img1_first10.reshape(-1), img1_first10.reshape(-1))) #<< 0.05 Two distribuiutions are different

print('ks_2samp for img1 top 10 vs img1 middle 10 =  = ', ks_2samp(img1_first10.reshape(-1), img1_middle10.reshape(-1))) #<< 0.05 Two distribuiutions are different
print('ks_2samp for img1 top 10 vs img2 middle 10 = ', ks_2samp(img1_first10.reshape(-1), img2_first10.reshape(-1))) #<< 0.05 Two distribuiutions are different

#Compare 1st 10 images from 1st set and last 10 from the 2nd set
print('ks_2samp for norm. img1 top 10 vs img2 middle 10 =  = ', ks_2samp(set1_first10.reshape(-1), set2_middle10.reshape(-1))) #<< 0.05 Two distribuiutions are different


############################################################

#Applying trained model to segment multiple files. 
#Get a smaller subset for a fun exercise.... 
set1_df = pd.DataFrame(set1_first10.reshape(-1))
set2_df = pd.DataFrame(set2_middle10.reshape(-1)) #Also check with last10

set1_subset = set1_df.sample(5000, random_state=42)
set2_subset = set2_df.sample(5000, random_state=42)


#label data so we know which dataset they belong to after combining them
set1_subset['data_ID'] = 1
set2_subset['data_ID'] = 2 

set1_set2 = set1_subset.append(set2_subset)
set1_set2.reset_index(inplace=True, drop=True)

from sklearn.utils import shuffle
shuffled = shuffle(set1_set2, random_state=42)
shuffled.reset_index(inplace=True, drop=True)


y = shuffled['data_ID'].values
x = shuffled.drop(labels = ['data_ID'], axis=1)
x = x.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)


#from sklearn.svm import LinearSVC, SVC
#model_SVM = SVC(kernel="linear", C=0.025)
#model_SVM.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier(max_depth=5, n_estimators=10)
model_RF.fit(X_train, y_train)

prediction = model_RF.predict(X_test)


#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction))

from sklearn.metrics import confusion_matrix
import seaborn as sns
#Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(y_test, prediction)
#print(cm)
sns.heatmap(cm, annot=True)

#Print individual accuracy values for each class, based on the confusion matrix
print("Dataset 1 accuracy = ", cm[0,0] / (cm[0,0]+cm[1,0]))
print("Dataset 2 accuracy = ",   cm[1,1] / (cm[0,1]+cm[1,1]))
#High Accuracy means the model is able to tell difference between two datasets. 
#So we need to retrain the model. 

