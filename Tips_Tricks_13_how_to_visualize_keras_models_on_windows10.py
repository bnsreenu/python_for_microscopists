# https://youtu.be/dunU57850uc
"""
Python tips and tricks - 13
How to plot keras models using plot_model on Windows10

We use the plot_model library:​
from tensorflow.keras.utils import plot_model​

Plot_model requires Pydot and graphviz libraries.​

To install Graphviz: ​
Download and install the latest version exe​
https://gitlab.com/graphviz/graphviz/-/releases ​

To check the installation,​
go to the command prompt and enter: dot -V​

Open Anaconda prompt for the ​desired environment ​

pip install pydot​
pip install graphviz​

"""

from tensorflow.keras.utils import plot_model

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(1))
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='simple_model.png')

##################################################################
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
plot_model(model, to_file='complex_model.png')