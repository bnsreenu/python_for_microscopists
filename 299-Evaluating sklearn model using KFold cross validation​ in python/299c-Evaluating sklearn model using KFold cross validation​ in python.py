# https://youtu.be/6dDet0-Drzc

"""
Evaluating sklearn model performance using KFold cross validation
(By using cross_val_score.)

Cross-validation: evaluating estimator performance
Using one of the scikit-learn estimators (SVM model)

KFOLD is a model validation technique.

Cross-validation between multiple folds allows us to evaluate the model performance. 

KFold library in sklearn provides train/test indices to split data in train/test sets. 
Splits dataset into k consecutive folds (without shuffling by default).
Each fold is then used once as a validation while the k - 1 remaining folds 
form the training set.

We will use the cross_val_score() function to perform the evaluation. 
It takes the dataset and cross-validation configuration and returns a list of 
scores calculated for each fold.


Wisconsin breast cancer example
Dataset link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

"""
import sys
("Python version is", sys.version)
import sklearn
print("Scikit-learn version is: ", sklearn.__version__)


import pandas as pd
import seaborn as sns
 

df = pd.read_csv("data/wisconsin_breast_cancer_dataset.csv")

print(df.describe().T)  #Values need to be normalized before fitting. 
print(df.isnull().sum())
#df = df.dropna()

#Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'diagnosis':'Label'})
print(df.dtypes)

#######  How many data points for each class?
df['Label'].value_counts()

#Understand the data 
sns.countplot(x="Label", data=df) #M - malignant   B - benign

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values


#Define x and normalize values

#Define the independent variables. Let's also drop Gender, so we can normalize other data
X = df.drop(labels = ["Label", "id"], axis=1) 

#Let us first convert our X from DataFrame to Numpy
#We didn't do this earlier as Scaler transformation does it automatically
X_array=X.to_numpy()


##########################################################
## THE WRONG WAY or NOT SO RIGHT WAY of using cross_val_score

#Cross validation score using the built-in function in sklearn
# Think about how do we scale values if the split is being automatically done 
# during crossvalidation? - Answer further below - under the RIGHT WAY
###################################################

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn import svm

# prepare the cross-validation procedure
cv = KFold(n_splits=7, random_state=42, shuffle=True)
# create model
model = svm.SVC(kernel='linear', C=1, random_state=42)

####This is the wrong part ##########
#Pepare/scale the data - But, is this the right way?
#Shouldn't we be scaling train and test separately?
scaler = MinMaxScaler()
scaler.fit(X_array)
X_array = scaler.transform(X_array)

# evaluate model
scores = cross_val_score(model, X_array, y, scoring='accuracy', cv=cv, n_jobs=-1)

for score in scores:
    print("Accuracy for this fold is: ", score)

# Mean accuracy
print(' Mean accuracy over all folds is: ', (np.mean(scores)))


###################################################################
## THE RIGHT WAY of using cross_val_score
#Proper way of performing cross validation is by including scaling as part of the
# crossvaidation pipeline
################################################################
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import Pipeline

#Define our input again, non-scaled input
X_array=X.to_numpy()

# create model
model = svm.SVC(kernel='linear', C=1, random_state=42)

### The right part ###
# define the pipeline to include scaling and the model. 
#This pipeline will be the input to cross_val_score, instead of the model. 
steps = list()
steps.append(('scaler', MinMaxScaler()))
steps.append(('model', model))
pipeline = Pipeline(steps=steps)


# prepare the cross-validation procedure
cv = KFold(n_splits=7, random_state=42, shuffle=True)


# evaluate model
scores2 = cross_val_score(pipeline, X_array, y, scoring='accuracy', cv=cv, n_jobs=-1)

for score in scores2:
    print("Accuracy for this fold is: ", score)

# Mean accuracy
print(' Mean accuracy over all folds is: ', (np.mean(scores2)))

