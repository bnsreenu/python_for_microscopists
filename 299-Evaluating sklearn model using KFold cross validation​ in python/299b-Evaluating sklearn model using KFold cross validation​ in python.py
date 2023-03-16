# https://youtu.be/6dDet0-Drzc

"""
Evaluating sklearn model performance using KFold cross validation
(By manually enumerating K-folds.)

Cross-validation: evaluating estimator performance
Using one of the scikit-learn estimators (SVM model)

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


In this example, we will first split data into multiple folds and then train the 
model on each fold. We will manually iterate through the training over each fold. 


Wisconsin breast cancer example
Dataset link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

"""
import sys
("Python version is", sys.version)
import sklearn
print("Scikit-learn version is: ", sklearn.__version__)


#Let us start by looking at the normal way we address this problem, 
# without cross validation check

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


#Now, let us look at k-fold 
#Manually fitting the model by iterating over each fold
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics, svm
from sklearn.model_selection import KFold

skf = KFold(n_splits=7, random_state=42, shuffle=True)

#In our example, we have 569 data points. If n_splits=7, we have the data
#divided into 569/7 = 82 groups. All groups are used for cross validation. 
#Note that the size of the test set will be (total_data_size/n_splits = 569/7=82)

#Let us first convert our X from DataFrame to Numpy
#We didn't do this earlier as Scaler transformation does it automatically
X_array=X.to_numpy()

#Study how the data gets presented n number of times
data_splits_object=skf.split(X_array, y)

#Run the following 3 lines together multiple times to see how different data gets presented
my_split_data=next(data_splits_object)
print("Train indices are: ", my_split_data[0][0:10])
print("Test indices are: ", my_split_data[1][0:10]) #Print first 10 values

# empty lists to store predicted and ground truth values from all folds
predicted_y = []
expected_y = []

#Define the scaler for scaling within the loop
scaler = MinMaxScaler()

# Fit and predict iteratively on all splits
for train_index, test_index in skf.split(X_array, y):
    x_train, x_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #Scale inputs
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Define and fit the classifier (model)
    classifier = svm.SVC(kernel='linear', C=1, random_state=42) #C is kept 1. Can be more than 1 if you want nested KFold splits. 
    classifier.fit(x_train, y_train)
    
    # print accuracy for each fold
    predicted_this_fold = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, predicted_this_fold)
    print("Accuracy from this fold is: " + accuracy.__str__())

    # store the prediction in a list
    predicted_y.extend(predicted_this_fold)

    # store the ground truth for this specific fold
    expected_y.extend(y_test)

# print accuracy
accuracy = metrics.accuracy_score(expected_y, predicted_y)
print('\n', "Accuracy from all folds is: " + accuracy.__str__())

#########################################################
