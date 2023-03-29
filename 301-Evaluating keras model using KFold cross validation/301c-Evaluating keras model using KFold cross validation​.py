# https://youtu.be/s5tNevBIg4Y

"""
Binary classification using keras - with cross_val_score method

KFOLD is a model validation technique.

Cross-validation between multiple folds allows us to evaluate the model performance. 

KFold library in sklearn provides train/test indices to split data in train/test sets. 
Splits dataset into k consecutive folds (without shuffling by default).
Each fold is then used once as a validation while the k - 1 remaining folds 
form the training set.

Split method witin KFold generates indices to split data into training and test set.

The split will divide the data into n_samples/n_splits groups. 
One group is used for testing and the remaining data used for training.
All combinations of n_splits-1 will be used for cross validation.  


Normally, we would use cross_val_score in sklearn to automatically evaluate
the model over all splits and report the crossvalidation score. But, that method
is deisgned to handle traditional sklearn models such as SVM, RF, 
gradient boosting etc. - NOT deep learning models from tensorflow or pytorch/ 

Therefore, in order to use cross_val_score, we will find a way to make our
keras model available to the function. This is done using the KerasClassifier
from tensorflow.keras.wrappers.scikit_learn

Note that the cross_val_score() function takes the dataset and cross-validation 
configuration and returns a list of scores calculated for each fold.

Wisconsin breast cancer example
Dataset link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data


"""
import sys
("Python version is", sys.version)
import sklearn
print("Scikit-learn version is: ", sklearn.__version__)
import tensorflow as tf
print("Tensorflow version is: ", tf.__version__)


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout


df = pd.read_csv("data/wisconsin_breast_cancer_dataset.csv")

print(df.describe().T)  #Values need to be normalized before fitting. 
print(df.isnull().sum())


#Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'diagnosis':'Label'})
print(df.dtypes)

#Understand the data 
#sns.countplot(x="Label", data=df) #M - malignant   B - benign


####### Replace categorical values with numbers########
df['Label'].value_counts()

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y) # M=1 and B=0

#Define x and normalize values
#Define the independent variables. Let's also drop Gender, so we can normalize other data
X = df.drop(labels = ["Label", "id"], axis=1) 

#############################################
#W e really didn't have to do this deliberately via for loop
# we can use KerasClassifier from tensorflow's sklearn wrapper to define a
# keras model that can be used in cross_val_score

###############################

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Define a function for our model so we can call it in cross_val_score
def create_model():
    model = Sequential()
    model.add(Dense(16, input_dim=30, activation='relu')) 
    model.add(Dropout(0.2))
    model.add(Dense(1)) 
    model.add(Activation('sigmoid'))  
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',             
                  metrics=['accuracy'])
    return model


# create model
my_keras_model = KerasClassifier(build_fn=create_model, 
                                 epochs=20, batch_size=16, 
                                 verbose=0)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import Pipeline

# define the pipeline to include scaling and the model. 
#This pipeline will be the input to cross_val_score, instead of the model. 
from sklearn.preprocessing import MinMaxScaler

steps = list()
steps.append(('scaler', MinMaxScaler()))
steps.append(('model', my_keras_model))
pipeline = Pipeline(steps=steps)


# Define the crossvalidation process to be used inside cross_val_score evaluation
cv = KFold(n_splits=7, random_state=42, shuffle=True)

# evaluate the model - 
scores = cross_val_score(pipeline, X, Y, scoring='accuracy', 
                         cv=cv, n_jobs=1)

for score in scores:
    print("Score for this split is: ", score)

# report performance
print('Accuracy: ', (np.mean(scores)))




