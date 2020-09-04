
# https://youtu.be/bqBRET7tbiQ
"""
#Heart disease
The effect that the independent variables biking and smoking 
have on the dependent variable heart disease 
#Dataset link:
https://cdn.scribbr.com/wp-content/uploads//2020/02/heart.data_.zip?_ga=2.217642335.893016210.1598387608-409916526.1598387608

NOTE: #Linear regression uses ordinary least squares as optimizer

"""


import numpy as np

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('data/heart_data.csv')
print(df.head())

df = df.drop("Unnamed: 0", axis=1)
#A few plots in Seaborn to understand the data


#sns.lmplot(x='biking', y='heart.disease', data=df)  
#sns.lmplot(x='smoking', y='heart.disease', data=df)  


x_df = df.drop('heart.disease', axis=1)

#x_df = x_df.drop("smoking", axis=1) #Single variable (Biking)


y_df = df['heart.disease']


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_df)
x = scaler.transform(x_df)

#x = X.to_numpy()
y = y_df.to_numpy()

# #Bring data back to pandas DF for plotting
# df_for_plot = pd.DataFrame(x, columns=['biking', 'smoking'])
# df_for_plot['heart.disease'] = y

# sns.scatterplot(x='biking', y='heart.disease', data=df_for_plot)
# sns.scatterplot(x='smoking', y='heart.disease', data=df_for_plot)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn import linear_model

#Create Linear Regression object
model = linear_model.LinearRegression()
#Linear regression uses ordinary least squares as optimizer
#Now let us call fit method to train the model using independent variables.
#And the value that needs to be predicted (Images_Analyzed)

model.fit(X_train, y_train) #Indep variables, dep. variable to be predicted
print(model.score(X_train, y_train))  #Prints the R^2 value, a measure of how well


prediction_test = model.predict(X_test)    
print(y_test, prediction_test)
print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)


#Model is ready. Let us check the coefficients, stored as reg.coef_.
#These are a, b, and c from our equation. 
#Intercept is stored as reg.intercept_
print(model.coef_, model.intercept_)

#All set to predict the number of images someone would analyze at a given time
#print(model.predict([[13, 2, 23]]))
