# https://youtu.be/s5tNevBIg4Y

"""

Binary classification using keras 

This is the normal way most of us approach the problem of binary classification
using neural networks (deep learning). In this example, we will split our
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
import tensorflow as tf
print("Tensorflow version is: ", tf.__version__)


#Our standard way of defining and training a model - with no KFold CV
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout


df = pd.read_csv("data/wisconsin_breast_cancer_dataset.csv")

print(df.describe().T)  #Values need to be normalized before fitting. 
print(df.isnull().sum())
#df = df.dropna()

#Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'diagnosis':'Label'})
print(df.dtypes)

#Understand the data 
#sns.countplot(x="Label", data=df) #M - malignant   B - benign
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

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

#Always remember to scale/normalize values after splitting.
#Otherwise some information from your test set will leak into the training process. 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)  #scaler converts our dataframe into arrays
X_test = scaler.transform(X_test)


#Let us define a Simple network
# Appropriate architecture for the challenge
model = Sequential()
model.add(Dense(16, input_dim=30, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(1)) 
model.add(Activation('sigmoid'))  
model.compile(loss='binary_crossentropy',
              optimizer='adam',             #also try adam
              metrics=['accuracy'])
print(model.summary())

# # fit the model
history = model.fit(X_train, y_train ,verbose=1, epochs=20, batch_size=64,
                     validation_data=(X_test, y_test))

_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")


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


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)

