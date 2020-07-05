#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/i2JSH5tn2qc

"""
What in the world is one hot encoding?

"""

# San Francisco = 1
# Chicago = 2
# New York = 3

import numpy as np
# define example
age = np.array([55, 45, 64, 23, 34, 22, 87, 43])
city = np.array(['1', '1', '3', '2','3', '1', '2', '2'])
x = age
from keras.utils import to_categorical
y = to_categorical(city)


###################################################################

import numpy as np
# define example
age = np.array([55, 45, 64, 23, 34, 22, 87, 43])
city = np.array(['San Francisco', 'San Francisco', 
                 'New York', 'Chicago','New York', 
                 'San Francisco', 'Chicago', 'Chicago'])
x = age

from keras.utils import to_categorical

#try one hot encoding... it fails as the values are not integers.
y = to_categorical(city)

#First, Encode target labels with value between 0 and n_classes-1.
from sklearn.preprocessing import LabelEncoder  #
label_encoder = LabelEncoder() #Create an instance 

#Fit label encoder and return encoded labels as a vector
vector = label_encoder.fit_transform(city) #create a vector

#Convert to categorical
y = to_categorical(vector)



#############################################
#Another example by generating blobs 
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from numpy import where

# generate classification dataset with 3 centers (labels/classes)
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
#First plot the data without categorical encoding to see the 3 clusters
y = to_categorical(y)
#Look at y in the variable explorer. 

# scatter plot for each class value
for class_value in range(3):
	# select indices of points with the class label
	row_ix = where(y == class_value)
	# scatter plot for points with a different color
	plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show plot
plt.show()