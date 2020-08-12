
# https://youtu.be/vOBbKNwi6Go

"""
@author: Sreenivas Bhattiprolu

Random Forest performs well on imbalanced datasets 
because of its hierarchical structure allows it to learn signals from all classes.
"""

import numpy as np
import cv2
import pandas as pd

 
img = cv2.imread('images/Sandstone_Versa0180_image.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

#Save original image pixels into a data frame. This is our Feature #1.
img2 = img.reshape(-1)
df = pd.DataFrame()
df['Original Image'] = img2

#Generate Gabor features
num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
kernels = []
for theta in range(2):   #Define number of thetas
    theta = theta / 4. * np.pi
    for sigma in (1, 3):  #Sigma with 1 and 3
        for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
            for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
            
                
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
#                print(gabor_label)
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
                          
#Now, add a column in the data frame for the Labels
#For this, we need to import the labeled image
labeled_img = cv2.imread('images/Sandstone_Versa0180_mask.png')
#Remember that you can load an image with partial labels 
#But, drop the rows with unlabeled data

labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)
df['Labels'] = labeled_img1

print(df.head())


print(df.Labels.unique())  #Look at the labels in our dataframe
print(df['Labels'].value_counts())

#Let us drop background pixels. In this example they have a value 33
# Normally you may want to also drop unlabeled pixels with value 0
#df = df[df.Labels != 33]
#print(df['Labels'].value_counts()) 

#Still label 231 is over-represented and label 65 is underrepresented



#########################################################

#Define the dependent variable that needs to be predicted (labels)
Y = df["Labels"].values

#Define the independent variables
X = df.drop(labels = ["Labels"], axis=1) 


X = X.iloc[:, :].as_matrix() #Convert X from pandas dataframe to Numpy array



#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

#################################################################
############################################################################
#For deep learning....

	# create model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(8, input_dim=33, activation='relu'))
#model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))
	# Compile model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

from keras.utils import normalize, to_categorical
from sklearn.preprocessing import LabelEncoder

X_train_deep = normalize(X_train, axis=1)
X_test_deep = normalize(X_test, axis=1)

#Encode Y labels to 0, 1, 2, 3, then one hot encode. 
le = LabelEncoder()
le.fit(y_train)
y_train_deep = le.transform(y_train)
y_train_deep_cat = to_categorical(y_train_deep)

y_test_deep = le.transform(y_test)
y_test_deep_cat = to_categorical(y_test_deep)


#Automatically define class weights
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',    #Can also give a dictionary
                                                 np.unique(y_train_deep),   #Classes
                                                 y_train_deep)   #Y 

#Manually define class weights
#Try all equal weights first then try to bring all to about 30k using appropriate weights
#NOTE: AFter encoding: 33=0, 65=1 (under represented), 201=2, 231=3 (overrep.)

#print(df['Labels'].value_counts())
values, counts = np.unique(y_train_deep, return_counts=True)
print(values, counts)

class_weights_manual = {0: 0.1,
                1: 1.6,
                2: 0.108,
                3: 0.015}


#class_weights_manual = {0: 0.1,
#                1: 0.1,
#                2: 0.1,
#                3: 0.1}



history = model.fit(X_train_deep, y_train_deep_cat, epochs=10, 
                    class_weight=class_weights_manual, validation_data = (X_test_deep, y_test_deep_cat))



from matplotlib import pyplot as plt
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




prediction_deep = model.predict(X_test_deep)
# evaluate the keras model
_, accuracy = model.evaluate(X_test_deep, y_test_deep_cat, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

mythreshold=0.5
y_pred_thresh = (prediction_deep >= mythreshold).astype(int)

y_pred_thresholded=np.argmax(y_pred_thresh, axis=-1)

#y_pred_thresholded = y_pred_thresholded.argmax(1)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test_deep, y_pred_thresholded)  
print(cm)


print("Pixel 33 accuracy = ", cm[0,0] / (cm[0,0]+cm[1,0]+cm[2,0]+cm[3,0]))
print("Pixel 65 accuracy = ",   cm[1,1] / (cm[0,1]+cm[1,1]+cm[2,1]+cm[3,1]))
print("Pixel 201 accuracy = ",   cm[2,2] / (cm[0,2]+cm[1,2]+cm[2,2]+cm[3,2]))
print("Pixel 231 accuracy = ",   cm[3,3] / (cm[0,3]+cm[1,3]+cm[2,3]+cm[3,3]))