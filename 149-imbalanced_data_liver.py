

#https://youtu.be/VQuJvGTzBgw

"""
@author: Sreenivas Bhattiprolu


https://www.kaggle.com/uciml/indian-liver-patient-records
Prediction of Liver Disease using Random Forest classifier
and balancing imbalanced data

This data set contains 416 liver patient records and 167 non liver patient records 
collected from North East of Andhra Pradesh, India. 
The "Dataset" column is a class label used to divide groups into 
liver patient (liver disease) or not (no disease). 
This data set contains 441 male patient records and 142 female patient records.

Any patient whose age exceeded 89 is listed as being of age "90".

Based on chemical compounds(bilrubin,albumin,protiens,alkaline phosphatase) 
present in human body and tests like SGOT , SGPT the outcome mentioned whether 
person is patient ie needs to be diagnosed or not.

Columns:

Age of the patient
Gender of the patient
Total Bilirubin
Direct Bilirubin
Alkaline Phosphotase
Alamine Aminotransferase
Aspartate Aminotransferase
Total Protiens
Albumin
Albumin and Globulin Ratio
Dataset: field used to split the data into two sets (patient with liver disease, or no disease)
"""

import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
 
df = pd.read_excel("indian_liver_patient.xlsx")

print(df.describe().T)  #Values need to be normalized before fitting. 


print(df.isnull().sum())
#df = df.dropna()

print(df['Albumin_and_Globulin_Ratio'].mean())
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(0.947)
print(df.isnull().sum())

#Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'Dataset':'Label'})
print(df.dtypes)

#Understand the data - # Value 1 = Liver disease and 2 is no disease
sns.countplot(x="Label", data=df)
sns.countplot(x="Label", hue="Gender", data=df)

sns.distplot(df['Age'], kde=False)

plt.figure(figsize=(20,10)) 
sns.countplot(x = 'Age', data = df, order = df['Age'].value_counts().index)


sns.scatterplot(x="Label", y="Albumin", data=df)  #Seems no trend between labels 1 and 2
sns.scatterplot(x="Label", y="Albumin_and_Globulin_Ratio", data=df)  #Seems no trend between labels 1 and 2
sns.scatterplot(x="Albumin", y="Albumin_and_Globulin_Ratio", data=df)  #Seems no trend between labels 1 and 2

#sns.pairplot(df, hue='Gender')
     
corr=df.corr()
plt.figure(figsize=(20,12)) 
sns.heatmap(corr,cmap="Blues",linewidths=.5, annot=True)
#May be Gender and total protien not big factors influencing the label

#Replace categorical values with numbers
df['Gender'].value_counts()

categories = {"Male":1, "Female":2}
df['Gender'] = df['Gender'].replace(categories)


#Define the dependent variable that needs to be predicted (labels)
Y = df["Label"].values

#Define the independent variables. Let's also drop Gender, so we can normalize other data
X = df.drop(labels = ["Label", "Gender"], axis=1) 

from keras.utils import normalize
X = normalize(X, axis=1)

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#################################################################
# Technique 1: Appropriate Model Selection
#Logistic regression - Can be very slow. 
#SVM  - can also be extremely slow. 
#Random Forest - ideal for imbalanced datasets. 
############################################################
# from sklearn.linear_model import LogisticRegression
# model_logistic = LogisticRegression(max_iter=50).fit(X, Y)
# prediction_test_LR = model_logistic.predict(X_test)

# from sklearn import metrics
# print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test_LR))


# from sklearn.svm import SVC
# model_SVM = SVC(kernel='linear')
# model_SVM.fit(X_train, y_train)

# prediction_test_SVM = model_SVM.predict(X_test)
# print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test_SVM))


#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier(n_estimators = 25, random_state = 42)

# Train the model on training data
model_RF.fit(X_train, y_train)

#importances = list(model_RF.feature_importances_)
features_list = list(X.columns)
feature_imp = pd.Series(model_RF.feature_importances_, index=features_list).sort_values(ascending=False)
print(feature_imp)


#Test prediction on testing data. 
prediction_test_RF = model_RF.predict(X_test)

#ACCURACY METRICS
print("********* METRICS FOR IMBALANCED DATA *********")
#Let us check the accuracy on test data
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test_RF))


(unique, counts) = np.unique(prediction_test_RF, return_counts=True)
print(unique, counts)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_test_RF)
print(cm)

#Print individual accuracy values for each class, based on the confusion matrix
print("With Lung disease = ", cm[0,0] / (cm[0,0]+cm[1,0]))
print("No disease = ",   cm[1,1] / (cm[0,1]+cm[1,1]))

#Note the low accuracy for the important class (201 label)

#Right metric is ROC AUC
#Starting version 0.23.1 you can report this for multilabel problems. 
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
from sklearn.metrics import roc_auc_score  #Version 0.23.1 of sklearn

print("ROC_AUC score for imbalanced data is:")
print(roc_auc_score(y_test, prediction_test_RF))


#https://www.scikit-yb.org/en/latest/api/classifier/rocauc.html
from yellowbrick.classifier import ROCAUC

roc_auc=ROCAUC(model_RF)  #Create object
roc_auc.fit(X_train, y_train)
roc_auc.score(X_test, y_test)
roc_auc.show()

#############################################################################
# Handling Imbalanced data
###########################################

# Technique 2 Up-sample minority class
from sklearn.utils import resample
print(df['Label'].value_counts())

#Separate majority and minority classes
df_majority = df[df['Label'] == 1]
df_minority = df[df['Label'] == 2]

# Upsample minority class and other classes separately
# If not, random samples from combined classes will be duplicated and we run into
#same issue as before, undersampled remians undersampled.
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=416,    # to match average class
                                 random_state=42) # reproducible results
 

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
print(df_upsampled['Label'].value_counts())

Y_upsampled = df_upsampled["Label"].values

#Define the independent variables
X_upsampled = df_upsampled.drop(labels = ["Label", "Gender"], axis=1) 
X_upsampled = normalize(X_upsampled, axis=1)


#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train_upsampled, X_test_upsampled, y_train_upsampled, y_test_upsampled = train_test_split(X_upsampled, 
                                                                                            Y_upsampled, 
                                                                                            test_size=0.2, 
                                                                                            random_state=20)

#Train again with new upsamples data
model_RF_upsampled = RandomForestClassifier(n_estimators = 25, random_state = 42)

# Train the model on training data
model_RF_upsampled.fit(X_train_upsampled, y_train_upsampled)
prediction_test_RF_upsampled = model_RF_upsampled.predict(X_test_upsampled)

print("********* METRICS FOR BALANCED DATA USING UPSAMPLING *********")

print ("Accuracy = ", metrics.accuracy_score(y_test_upsampled, prediction_test_RF_upsampled))

cm_upsampled = confusion_matrix(y_test_upsampled, prediction_test_RF_upsampled)
print(cm_upsampled)

print("With Lung disease =  = ", cm_upsampled[0,0] / (cm_upsampled[0,0]+cm_upsampled[1,0]))
print("No lung disease = ",  cm_upsampled[1,1] / (cm_upsampled[0,1]+cm_upsampled[1,1]))


print("ROC_AUC score for balanced data using upsampling is:")
print(roc_auc_score(y_test_upsampled, prediction_test_RF_upsampled))


from yellowbrick.classifier import ROCAUC

roc_auc=ROCAUC(model_RF_upsampled)
roc_auc.fit(X_train_upsampled, y_train_upsampled)
roc_auc.score(X_test_upsampled, y_test_upsampled)
roc_auc.show()


#Copy training code here from above #######################
########################################################################
#############################################################################

# Technique 6: Generate synthetic data (SMOTE and ADASYN)
# SMOTE: Synthetic Minority Oversampling Technique
#ADASYN: Adaptive Synthetic
# https://imbalanced-learn.org/stable/over_sampling.html?highlight=smote
# pip install imblearn
# SMOTE may not be the best choice all the time. It is one of many things
#that you need to explore. 

from imblearn.over_sampling import SMOTE, ADASYN

X_smote, Y_smote = SMOTE().fit_resample(X, Y)  #Beware, this takes some time based on the dataset size
#X_adasyn, Y_adasyn = ADASYN().fit_resample(X, Y)

X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, 
                                                                            Y_smote, 
                                                                            test_size=0.2, 
                                                                            random_state=42)


(unique, counts) = np.unique(Y, return_counts=True)
print("Original data: ", unique, counts)
(unique2, counts2) = np.unique(Y_smote, return_counts=True)
print("After SMOTE: ", unique2, counts2)
#(unique3, counts3) = np.unique(Y_adasyn, return_counts=True)
#print("After ADASYN: ", unique3, counts3)

model_SMOTE = RandomForestClassifier(n_estimators = 25, random_state = 42)
model_SMOTE.fit(X_train_smote, y_train_smote)

prediction_test_smote = model_SMOTE.predict(X_test_smote)

print ("Accuracy = ", metrics.accuracy_score(y_test_smote, prediction_test_smote))

print(roc_auc_score(y_test_smote, prediction_test_smote))


from yellowbrick.classifier import ROCAUC
roc_auc=ROCAUC(model_SMOTE)
roc_auc.fit(X_train_smote, y_train_smote)
roc_auc.score(X_test_smote, y_test_smote)
roc_auc.show()
