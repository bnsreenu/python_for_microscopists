
# https://youtu.be/bqBRET7tbiQ
"""
#Heart disease
The effect that the independent variables biking and smoking 
have on the dependent variable heart disease 
#Dataset link:
https://cdn.scribbr.com/wp-content/uploads//2020/02/heart.data_.zip?_ga=2.217642335.893016210.1598387608-409916526.1598387608

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


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, InputLayer

#############################################

# Build the network
# output = input - no training

# model = Sequential()
# model.add(InputLayer(input_shape=(X_train.shape[1], )))
# model.compile(loss='mean_squared_error', optimizer='adam')
# print(model.summary())

############################################
#0 hidden layers - linear equation
#Identical results to linear regression 

from keras.optimizers import SGD
opt = SGD(learning_rate=0.1)

model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1], )))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=opt)
print(model.summary())

###########################################
#1 hidden layer
#0 hidden layers - linear equation

# model = Sequential()
# model.add(InputLayer(input_shape=(X_train.shape[1], )))
# model.add(Dense(4))
# model.add(Dense(1)) 
# model.compile(loss='mean_squared_error', optimizer='adam')
# print(model.summary())
############################################

history = model.fit(X_train, y_train ,verbose=1, epochs=50, 
                    validation_data=(X_test, y_test))

# Predict

weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]


prediction_test = model.predict(X_test)    
print(y_test, prediction_test)
print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

