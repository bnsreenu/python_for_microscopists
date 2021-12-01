# https://youtu.be/Rx7pPuosoLk

"""

Training a keras model by enumerating epochs and batches using for loops
@author: Sreenivas Bhattiprolu

cifar10 dataset 
60,000 32×32 pixel images divided into 10 classes.
0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
"""

import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.datasets import cifar10
from keras.utils import normalize, to_categorical
import time
from numpy.random import randn
from numpy.random import randint
### Normalize inputs
#WHat happens if we don't normalize inputs?
# ALso we may have to normalize depending on the activation function

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Let us take a subset of images for faster training.
rand_train_images = randint(0, X_train.shape[0], 5000)
rand_test_images = randint(0, X_test.shape[0], 1000)

X_train = X_train[rand_train_images]
y_train = y_train[rand_train_images]
X_test = X_test[rand_test_images]
y_test = y_test[rand_test_images]

X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



######################################################################
"""
Define the model using sequential or functional API. 

"""

activation = 'sigmoid'
model = Sequential()
model.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (32, 32, 3)))
model.add(BatchNormalization())

model.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())

model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization()) 
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128, activation = activation, kernel_initializer = 'he_uniform'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary()) 
###################################################
#Copy the model so we can later train it using for loops.
model1 = model
######################################################################
#Traditional training using model.fit

start = time.time()

history = model.fit(
        X_train, y_train,
        epochs = 10,
        batch_size=16,
        validation_data = (X_test, y_test)
)

end = time.time()
print("Time to train the model normal way", end - start)


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


################################################
"""
#Training using epoch and batch enumeration
"""
#############################################

#Fetch a batch of images for training
def fetch_batch(X, y, batch_size, batch):
    start = batch*batch_size
    
    X_batch = X[start:start+batch_size, :, :]
    y_batch = y[start:start+batch_size, :]
    
    return X_batch, y_batch

#Also try fetching using a random batch of images.
# def fetch_random_batch(X, y, batch_size):
#     # choose random images
#     rand_images = randint(0, X.shape[0], batch_size)
#     # select the random images and assign it to X
#     X = X[rand_images]
#     # get image labels labels
#     y = y[rand_images]
#     return X, y


batch_size = 16
loss_history = []
val_loss_history = []
acc_history = []
val_acc_history = []
n_epochs = 10


from tqdm import tqdm
from sklearn.utils import shuffle

start1 = time.time()
for epoch in range(n_epochs):
    X, y = shuffle(X_train, y_train, random_state = epoch**2)
    for batch in tqdm(range(len(X_train) //batch_size)):
    
        X_batch, y_batch = fetch_batch(X, y, batch_size, batch)
        # X_batch, y_batch = fetch_random_batch(X, y, batch_size)
        loss, acc = model1.train_on_batch(X_batch, y_batch)
    
    loss_history.append(loss)
    acc_history.append(acc)
    
    # Run validtion at the end of each epoch.
    y_pred = model1.predict(X_test)
    val_loss, val_acc = model1.evaluate(X_test, y_test)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
        
        
    print('Epoch: %d, Train Loss %.3f, Train Acc. %.3f, Val Loss %.3f, Val Acc. %.3f' %
			(epoch+1, loss, acc, val_loss, val_acc))
       
end1 = time.time()
print("Time to train the model using for loops", end1 - start1)

#plot the training and validation accuracy and loss at each epoch

epochs = range(1, len(loss_history) + 1)

plt.plot(epochs, loss_history, 'y', label='Training loss')
plt.plot(epochs, val_loss_history, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.plot(epochs, acc_history, 'y', label='Training acc')
plt.plot(epochs, val_acc_history, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()