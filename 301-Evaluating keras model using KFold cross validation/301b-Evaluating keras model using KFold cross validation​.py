# https://youtu.be/s5tNevBIg4Y

"""
Binary classification using keras - with manual enumeration over the 
K-fold splits.

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

Therefore, we will first split data into multiple folds and then train the 
model on each fold. We can manually iterate through the training over each fold. 


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




#############################################################

#Cross validation training - 
#cross_val_score in sklearn is deisgned to handle traditional sklearn models 
# such as SVM, RF, gradient boosting etc. - NOT deep learning models from 
#tensorflow or pytorch/ 

#First split data into multiple folds and then train the model on each fold. 
#We can manually iterate through the training over each fold. 

#Training the model of each fold and saving all models
###############################################################

#Splitting data into multiple folds and training model on each fold every time. 
from sklearn.model_selection import KFold
cv = KFold(n_splits=7, shuffle=True, random_state=42)

# K-fold Cross Validation model evaluation
fold_no = 1
acc_per_fold = [] #save accuracy from each fold

#Let us first convert our X from DataFrame to Numpy
#We didn't do this earlier as Scaler transformation does it automatically
X_array=X.to_numpy()

#Train the model for each split (fold)
#We will define the model inside the for loop as we want to initialize and 
#compile it for each fold. 
#We also need to scale values for each fold inside the loop
from sklearn.preprocessing import MinMaxScaler

for train, test in cv.split(X_array, Y):

    print('   ')
    print(f'Training for fold {fold_no} ...')

    #Scale data
    scaler = MinMaxScaler()
    train_X = X_array[train]
    test_X = X_array[test]
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    
    
    #Define the model - inside the loop so it trains from scratch for each fold
    #If defined outside, each fold training starts at where it left off at the previous fold
    #calling it as model2 instead of model to make sure no information from our
    #previous example is carried over (without restarting the kernel)
    model2 = Sequential()
    model2.add(Dense(16, input_dim=30, activation='relu')) 
    model2.add(Dropout(0.2))
    model2.add(Dense(1)) 
    model2.add(Activation('sigmoid'))  
    model2.compile(loss='binary_crossentropy',
                  optimizer='adam',             
                  metrics=['accuracy'])
    
    
  # Fit data to model
    history = model2.fit(train_X, Y[train],
                  batch_size=8,
                  epochs=20,
                  verbose=1)
    #Save model trained on each fold.
    model2.save('models/model_fold_'+str(fold_no)+'.h5')   

    # Evaluate the model - report accuracy and capture it into a list for future reporting
    scores = model2.evaluate(test_X, Y[test], verbose=0)
    acc_per_fold.append(scores[1] * 100)
    
    fold_no = fold_no + 1

for acc in acc_per_fold:
    print("accuracy for this fold is: ", acc)

