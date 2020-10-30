
# https://youtu.be/MKr2RQ_DqJE

"""
@author: Sreenivas Bhattiprolu

Autokeras:
    
Automatic model search for :
    
Image classification: autokeras.ImageClassifier
Image regression: autokeras.ImageRegressor (e.g. mnist)
Text classification: autokeras.TextClassifier
Text Regression: autokeras.TextRegressor
Structured data classification: autokeras.StructuredDataClassifier  (e.g Breast cancer dataset)
Structured data regression: autokeras.StructuredDataRegressor

cifar10 dataset 
60,000 32×32 pixel images divided into 10 classes.

0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck


"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.style.use('dark_background')

# from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator
# from keras.layers import BatchNormalization
# from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.datasets import cifar10
from keras.utils import normalize, to_categorical

import autokeras as ak


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape)  #50K

#Let us just use 1K images for demonstration purposes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=49000, random_state=42)

print(X_train.shape)  #1K
print(y_train.shape)  #1K


X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Documentation: https://autokeras.com/image_classifier/
clf = ak.ImageClassifier(max_trials=5) #MaxTrials - max. number of keras models to try
clf.fit(X_train, y_train, epochs=1)


#Evaluate the classifier on test data
_, acc = clf.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

# get the final best performing model
model = clf.export_model()
print(model.summary())

#Save the model
model.save('cifar_model.h5')








