# https://youtu.be/PG4XGqUeYnM

"""
Using....
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

df = pd.read_csv("data/wisconsin_breast_cancer_dataset.csv")

print(df.describe().T)  #Values need to be normalized before fitting. 
print(df.isnull().sum())
#df = df.dropna()

#Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'Diagnosis':'Label'})
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
#################################################################
#Define x and normalize values

#Define the independent variables. Let's also drop Gender, so we can normalize other data
X = df.drop(labels = ["Label", "ID"], axis=1) 


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


####################################################################
#Simple network 1
# Appropriate architecture for the challenge

model = Sequential()
model.add(Dense(16, input_dim=30, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(1)) 
model.add(Activation('sigmoid'))  
model.compile(loss='binary_crossentropy',
              optimizer='adam',             #also try adam
              metrics=['accuracy'])

 #model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

###########################################################
#Tensorboard callback

from tensorflow.keras.callbacks import TensorBoard
file_name = 'my_saved_model'  #Give unique name each time, otherwise results get concatenated
#Try running multiple times and see the output in tensorboard
#If file name is changed, you can select the specific model in TensorBoard

tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))

#After training open Anaconda command prompt and type the following:
# python -m tensorboard.main --logdir=logs/
# Gives the URL that you copy and paste in a browser.

# # evaluate the model
history = model.fit(X_train, y_train ,verbose=1, epochs=50, batch_size=64,
                     validation_data=(X_test, y_test), callbacks=[tensorboard])



#Fit with no early stopping or other callbacks
#history = model.fit(X_train, y_train ,verbose=1, epochs=50, batch_size=64,
#                    validation_data=(X_test, y_test))

# Predict

_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")


# prediction_test = model.predict(X_test)    
# print(y_test, prediction_test)
# print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)


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

# from keras.models import load_model
# saved_model = load_model('models/best_model.h5')

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)

