#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=2Bgma44OAF4


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

#TO create a model instance 

reg = linear_model.LinearRegression()  #Create an instance of the model.
reg.fit(x_df,y_df)   #Train the model or fits a linear model

print(reg.score(x_df,y_df))  #Prints the R^2 value, a measure of how well
#observed values are replicated by the model. 

#Test the model by Predicting cells for some values reg.predict()
print("Predicted # cells...", reg.predict([[2.3]]))

# Y = m * X + b (m is coefficient and b is intercept)
#Get the intercept and coefficient values

b = reg.intercept_
m = reg.coef_

#Manually verify the above calculation
print("From maual calculation, cells = ", (m*2.3 + b))


#Now predict cells for a list of times by reading time values from a csv file
cells_predict_df = pd.read_csv("other_files/cells_predict.csv")
print(cells_predict_df.head())

predicted_cells = reg.predict(cells_predict_df)
print(predicted_cells)

#Add the new predicted cells values as a new column to cells_predict_df dataframe
cells_predict_df['cells']=predicted_cells
print(cells_predict_df)

cells_predict_df.to_csv("other_files/predicted_cells.csv")


##############################
#
#####################################################

#Using Seaborn for plotting and linregress from scipy stats library

import pandas as pd
df = pd.read_csv('other_files/cells.csv')

import seaborn as sns
sns.set(style='darkgrid')
sns.lmplot(x='time', y='cells', data=df, order=1)

#If you want equation, not possible to display in seaborn but you can get it the
#regular way using scipy stats module. 
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(df['time'],df['cells'])
print(slope, intercept)

#Compare the slope and intercept reported with m and b values from above.
#Should be the same.





