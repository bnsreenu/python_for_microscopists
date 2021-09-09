#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=BAiMKBrFntc


"""

Same as linear regression using cells example except here we divide the dataset 
into training and test datasets.

"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('other_files/cells.csv')
print(df)

#plt.xlabel('time')
#plt.ylabel('cells')
#plt.scatter(df.time, df.cells,color='red',marker='+')

#For linear regression, Y=the value we want to predict
#X= all independent variables upon which Y depends. 
#3 steps for linear regression....
#Step 1: Create the instance of the model
#Step 2: .fit() to train the model or fit a linear model
#Step 3: .predict() to predict Y for given X values. 

#Now let us define our x and y values for the model.
#x values will be time column, so we can define it by dropping cells
#x can be multiple independent variables which we will discuss in a different tutorial
#this is why it is better to drop the unwanted columns rather than picking the wanted column
#y will be cells column, dependent variable that we are trying to predict. 

x_df = df.drop('cells', axis='columns')
#Or you can pick columns manually. Remember double brackets.
#Single bracket returns as series whereas double returns pandas dataframe which is what the model expects.
#x_df=df[['time']]
print(x_df.dtypes)  #Prints as object when you drop cells or use double brackets [[]]
#Prints as float64 if you do only single brackets, which is not the right type for our model. 
y_df = df.cells

#print(x_df)
#print(y_df)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.4, random_state=10)
#random_state can be any integer and it is used as a seed to randomly split dataset.
#By doing this we work with same test dataset evry time, if this is important.
#random_state=None splits dataset randomly every time



#TO create a model instance 

reg = linear_model.LinearRegression()  #Create an instance of the model.
reg.fit(X_train,y_train)   #Train the model or fits a linear model

print(reg.score(X_train, y_train))  #Prints the R^2 value, a measure of how well
#observed values are replicated by themodel. 


prediction_test = reg.predict(X_test)    
print(y_test, prediction_test)
print("Mean sq. errror between y_test and predicted =", np.mean((prediction_test-y_test)**2)) 
# A MSE value of about 8 is not bad compared to average # cells about 250.

#Residual plot
plt.scatter(prediction_test, prediction_test-y_test)
plt.hlines(y=0, xmin=200, xmax=300)

#Plot would be useful for lot of data points
