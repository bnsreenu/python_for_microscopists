# https://youtu.be/cOos6wRMpAU

"""
Picking the best model and corresponding hyperparameters
using cross validation inside a Gridsearch

The grid search provided by GridSearchCV exhaustively generates candidates 
from a grid of parameter values specified with the param_grid parameter
Example:
    param1 = {}
    param1['classifier__n_estimators'] = [10, 50, 100, 250]
    param1['classifier__max_depth'] = [5, 10, 20]
    param1['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
    param1['classifier'] = [RandomForestClassifier(random_state=42)]

The GridSearchCV instance when “fitting” on a dataset, all the possible 
combinations of parameter values are evaluated and the best combination is retained.

cv parameter can be defined for the cross-validation splitting strategy.

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

#Understand the data 
sns.countplot(x="Label", data=df) #M - malignant   B - benign

df['Label'].value_counts()

#Define the dependent variable that needs to be predicted (labels)
y = df["Label"].values

#Define the independent variables. Let's also drop Gender, so we can normalize other data
X = df.drop(labels = ["Label", "id"], axis=1) 


#Import all the models that you want to consider (include in the Grid search)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

#Import other useful libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# Initiaze the hyperparameters for each dictionary
# each having a key as ‘classifier’ and value as estimator object. 
#The hyperparameter keys should start with the word classifier separated 
# by ‘__’ (double underscore)

# Define parameters for Random Forest 
param1 = {}
param1['classifier__n_estimators'] = [10, 50, 100, 250]
param1['classifier__max_depth'] = [5, 10, 20]
param1['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param1['classifier'] = [RandomForestClassifier(random_state=42)]
#Total 48 parameters to test (4 * 3 * 4)

# Define parameters for support vector machine (SVC)
param2 = {}
param2['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
param2['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param2['classifier'] = [SVC(random_state=42)]
#Total 20 parameters to test (5 * 4)

# Define parameters for Logistic regression
param3 = {}
param3['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
param3['classifier__penalty'] = ['l1', 'l2']
param3['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param3['classifier'] = [LogisticRegression(random_state=42)]
#Total 40 parameters to test (5 * 2 * 4)

# Define parameters for K neighbors
param4 = {}
param4['classifier__n_neighbors'] = [2,5,10,25,50]
param4['classifier'] = [KNeighborsClassifier()]
#Total 5 parameters to test (5)

# Define parameters for Gradient boosting
param5 = {}
param5['classifier__n_estimators'] = [10, 50, 100, 250]
param5['classifier__max_depth'] = [5, 10, 20]
param5['classifier'] = [GradientBoostingClassifier(random_state=42)]
#Total 12 parameters to test (4 * 3)

# define the pipeline to include scaling and the model. 
# Prepare the pipeline for the 1st model, others will be loaded appropriately
#during the Grid Search
#This pipeline will be the input to cross_val_score, instead of the model. 
steps = list()
steps.append(('scaler', MinMaxScaler()))
steps.append(('classifier', RandomForestClassifier(random_state=42)))
pipeline = Pipeline(steps=steps)

#Capture all parameter dictionaries as a list
params = [param1, param2, param3, param4, param5]
# Total parameters for all 5 models = 48+20+40+5+12 = 125


#Grid search - including cross validation
grid = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, scoring='roc_auc').fit(X, y)

#Gridsearch object (in our case 'grid') stores all the information about
#the best model and corresponding hyperparameters. 
# print the best parameters...
print(grid.best_params_)

# print best score for the best model (in our case roc_auc score)
print(grid.best_score_)

# Stats for each test - we have a total 125 tests
means = grid.cv_results_['mean_test_score']
params_summary = grid.cv_results_['params']

#Capture all data into a Data Frame
import pandas as pd
df = pd.DataFrame(list(zip(means, params_summary)), columns=['Mean Score', 'Parmeters'])
