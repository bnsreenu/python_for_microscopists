# https://youtu.be/6dDet0-Drzc

"""

Binary classification using keras 

This is the normal way most of us approach the problem of binary classification
using sklearn (SVM). In this example, we will split our
data set the normal way into train and test groups. 

In the next python file, we will learn to divide data using K Fold splits.
We will iterate through each split to train and evaluate our model. 

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


#################################################################
#Define x and normalize/scale values

#Define the independent variables. Let's also drop Gender, so we can normalize other data
X = df.drop(labels = ["Label", "id"], axis=1) 


#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=20) #Try random 1, 20 and 42

#Always remember to scale/normalize values after splitting.
#Otherwise some information from your test set will leak into the training process. 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn import metrics, svm
# create the model
model = svm.SVC(kernel='linear', C=1, random_state=42)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)
print("Accuracy: " + accuracy.__str__())


####################################################################
