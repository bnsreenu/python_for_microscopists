# https://youtu.be/dnxlqgvaCt4
"""
defining_model_using_keras_functional_API

Can define keras models using the sequential method - preferred for simple sequential models

OR

define using the functional API - for non-linear models.
"""

from tensorflow.keras.utils import plot_model

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.datasets import cifar10
from keras.utils import normalize, to_categorical

"""
Defining the model using the Sequential method. 
"""

model = Sequential()

model.add(Conv2D(32, 3, activation = 'relu', input_shape = (32, 32, 3)))
model.add(Conv2D(32, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, activation = 'relu'))
model.add(Conv2D(64, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary()) 

# plot graph
plot_model(model, to_file='sequential_model.png')

##########################################################################

"""
Defining the exact same model using functional API, instead of sequential. 

"""
from keras.layers import Input
from keras.models import Model

#Must define a stand alone input layer specifying the shape.
input_img = Input(shape=(32,32,3))  

#For each layer, you need to specify where the input is coming from
#here, the input is coming from the input_img layer. 
conv1_1 = Conv2D(32, kernel_size=3, activation='relu')(input_img)
conv1_2 = Conv2D(32, kernel_size=3, activation='relu')(conv1_1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

conv2_1 = Conv2D(64, kernel_size=3, activation='relu')(pool1)
conv2_2 = Conv2D(64, kernel_size=3, activation='relu')(conv2_1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

flat = Flatten()(pool2)

hidden1 = Dense(128, activation='relu')(flat)
output = Dense(10, activation='softmax')(hidden1)

#Create the model using the Model class from keras. 
#Need to specify inputs and outputs. 
model_func = Model(inputs=input_img, outputs=output)

model_func.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])
print(model_func.summary()) 

# plot graph
plot_model(model_func, to_file='functional_model.png')


#Compare both models... should be identical....

########################################################
"""
Then, why use functional API?

- Provides a more flexible way of defining models. 
- Especially useful when defining models with multiple inputs and outputs. 


"""

#A bit complex model
# Model with multiple Inputs

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

# first input to the model
input1 = Input(shape=(64,64,1))
conv11 = Conv2D(32, kernel_size=4, activation='relu')(input1)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
conv12 = Conv2D(16, kernel_size=4, activation='relu')(pool11)
pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
flat1 = Flatten()(pool12)

# second input model
input2 = Input(shape=(32,32,3))
conv21 = Conv2D(32, kernel_size=4, activation='relu')(input2)
pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
conv22 = Conv2D(16, kernel_size=4, activation='relu')(pool21)
pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
flat2 = Flatten()(pool22)

# merge inputs
merge = concatenate([flat1, flat2])

# dense layers
hidden = Dense(10, activation='relu')(merge)
output = Dense(1, activation='sigmoid')(hidden)

model = Model(inputs=[input1, input2], outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='multi_input_model.png')

####################################################################
"""
Real use case: Discriminator in a Conditional GAN

"""
from keras.layers import Embedding
from keras.layers import Reshape
from keras.layers import Concatenate
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.optimizers import Adam

in_shape=(32,32,3)
n_classes = 10 #e.g., cifar or MNIST

# label input
in_label = Input(shape=(1,))  #Shape 1
li = Embedding(n_classes, 50)(in_label) #Shape 1,50
n_nodes = in_shape[0] * in_shape[1]  #32x32 = 1024. 
li = Dense(n_nodes)(li)  #Shape = 1, 1024
li = Reshape((in_shape[0], in_shape[1], 1))(li)  #32x32x1


# image input
in_image = Input(shape=in_shape) #32x32x3
# concat label as a channel
merge = Concatenate()([in_image, li]) #32x32x4 (4 channels, 3 for image and the other for labels)

#We will combine input label with input image and supply as inputs to the model. 
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge) #16x16x128
fe = LeakyReLU(alpha=0.2)(fe)
fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe) #8x8x128
fe = LeakyReLU(alpha=0.2)(fe)
# flatten feature maps
fe = Flatten()(fe)  #8192  (8*8*128=8192)
fe = Dropout(0.4)(fe)
# output
out_layer = Dense(1, activation='sigmoid')(fe)  #Shape=1

	# define model
##Combine input label with input image and supply as inputs to the model. 
model_disc = Model([in_image, in_label], out_layer)
	# compile model
opt = Adam(lr=0.0002, beta_1=0.5)
model_disc.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


print(model_disc.summary())

plot_model(model_disc, to_file='discriminator_model.png')

