#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=Q_7JaAp4emM

"""
Multiple Linear Regression uses several explanatory variables to predict the outcome of a response variable.
For example, if you have 10,000 images that need to be segmented and you are using manual approach.
Let us say you have 10 people performing the analysis and each of them perform
at different levels based on the time of the day and how much coffee they had.
In addition the result may depend on the age of the person and may also on the gender. 
There are a lot of variables and multiple linear regression is designed to create a model 
based on all these variables. 

Test Excel file contains 5 columns, user number, time of the day for analysis, 
# cups of coffee the person drank, age, and #images analyzed.
We need to predict the number of images analyzed using other variables.

#images analyzed = a * time + b * coffee + c * age + d

"""

import pandas as pd

df = pd.read_excel('other_files/images_analyzed.xlsx')
print(df.head())

#A few plots in Seaborn to understand the data

import seaborn as sns


sns.lmplot(x='Time', y='Images_Analyzed', data=df, hue='Age')  #Scatterplot with linear regression fit and 95% confidence interval
sns.lmplot(x='Coffee', y='Images_Analyzed', data=df, hue='Age', order=2)
#Looks like too much coffee is not good... negative effects

#sns.lmplot(x='Age', y='Images_Analyzed', data=df, hue='Age')

import numpy as np
from sklearn import linear_model

#Create Linear Regression object
reg = linear_model.LinearRegression()

#Now let us call fit method to train the model using independent variables.
#And the value that needs to be predicted (Images_Analyzed)

reg.fit(df[['Time', 'Coffee', 'Age']], df.Images_Analyzed) #Indep variables, dep. variable to be predicted

#Model is ready. Let us check the coefficients, stored as reg.coef_.
#These are a, b, and c from our equation. 
#Intercept is stored as reg.intercept_
print(reg.coef_, reg.intercept_)

#All set to predict the number of images someone would analyze at a given time
print(reg.predict([[13, 2, 23]]))
