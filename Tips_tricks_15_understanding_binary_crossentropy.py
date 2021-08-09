#https://youtu.be/Md4b67HvmRo

"""
An understanding of binary cross entropy using simple math

Our goal is to assign a class label to an image either as 1 (dog) or 0 (cat).
For that we normally extract features and fit a classifier. 

Let us say our features are float values that go between -1 and 2. 
[-1.8, -0.8, -0.5, -0.1, 0.5, 0.6, 0.9, 1.1, 1.5, 1.9]
Also, let us say y_true to be the true labels for the images. 
[0, 0, 1, 0, 1, 0, 1, 1, 1, 1]

Note that 3rd and 6th images are misclassified. 
3rd image is probably a dog that looks like a cat so the features fall in between cat features. 
6th image is probably a dog that looks like a cat so features are close to dog. 

Let us use this premise to calculate BCE using both logloss from scikit learn
and binary cross entropy from keras. 


"""


# Calculation of binary cross entropy using the library from keras

import numpy as np

features = np.array([-1.8, -0.8, -0.5, -0.1, 0.5, 0.6, 0.9, 1.1, 1.5, 1.9])
y_true = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 1])

#Let us fit our data to a model to get the probabilities for each data point. 
#Logistic regression (uses sigmoid function)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(features.reshape(-1, 1), y_true)
y_pred = model.predict_proba(features.reshape(-1, 1))[:, 1].ravel()

from math import log

avg_log_prob_negative = (-1)*(1/10)*(log(1-y_pred[0]) + log(1-y_pred[1]) + log(y_pred[2]) 
            + log(1-y_pred[3]) + log(y_pred[4]) + log(1-y_pred[5]) 
            + log(y_pred[6]) + log(y_pred[7]) + log(y_pred[8]) + 
            + log(y_pred[9])
            )

print("Calculated negative average log of probabilities = %.4f" %avg_log_prob_negative)
              
from sklearn.metrics import log_loss
#Loss using the log loss from scikit learn package
loss = log_loss(y_true, y_pred)
print('Log Loss using sklearn log_loss = %.4f' %loss)

import tensorflow as tf
#Loss using the binary cross entropy from keras
bce = tf.keras.losses.BinaryCrossentropy()  
bce = bce(y_true, y_pred).numpy()

print('Log Loss using keras BCE = %.4f' %bce)


#####################
