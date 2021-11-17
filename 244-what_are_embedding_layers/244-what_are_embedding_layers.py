# https://youtu.be/nam2zR7p7Os

"""
What is Embedding layer in keras?
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import tensorflow as tf


#Each value in the input_array (of input dimension) is mapped to a vector 
#of a defined size (output_dimension).
#Think of it as principal components where you map one dimension to another diemsnion(s).
#Obviously, here we are doing PCA. Just mapping one vector to another with some initialization. 
# So a 1 X 10 array gives 1 X 10 X 50 array (input_dim=1x10, output_dim=50).
# Embeddings can be randomly initialized on initialized using 0s or 1s. 

model = Sequential()
model.add(Embedding(10, 50, embeddings_initializer="ones")) #Try "uniform" and "ones" initializer

#Array of size 10, --> should output size (10,1,50)
input_array = np.array([0,1,2,3,4,5,6,7,8,9])
output_array = model.predict(input_array)

#Array of size (1,10) -->should output size (1,10,50)
input_array2 = np.expand_dims(np.array([0,1,2,3,4,5,6,7,8,9]), axis=0)
output_array2 = model.predict(input_array2)

##############################################
#Where in the world do we use embedding layer?

#########################################
"""
Text processing (example: word2vec)
Consider the following example sentences...

Hello, How are you doing - [0, 1, 2, 3, 4] (Hello is 0, are is 1, etc.)
Hello, How are you feeling - [0, 1, 2, 3, 5] (Feeling is 5)

Now, we want to train a network with first layer as embedding layer. 

"""

model = Sequential()
model.add(Embedding(6, 2, embeddings_initializer="uniform", input_length=5))
# In the above layer, we have 6 as the vocabulary contains 6 words. 
# 2 represents the size of the embedding vector. 
# input length is 5 as our sentences have 5 words length as input sequences.
input_array3 = np.expand_dims(np.array([0,1,2,3,4,5]), axis=0)
output_array3 = model.predict(input_array3) #Size (1,6,2)

#Now we have each input encoded as a vector of size 2. 
print(output_array3[0]) #First row for 0, second row for 1 and so on....

#Remember that these weights are part of the model training, so they get updated. 

#########################################
#Conditional GAN where we provide labels as input.
#We use embedding layer as the first layer to represent each class as a vector.
#e.g., cifar10 dataset has 10 classes. We want these 10 classes to be represented
#by vectors, each, that can be trained as part of discriminator training. 
#Let us say the vector would have a size 50. Then we can add an embedding layer
#of size 50. 

model = Sequential()
model.add(Embedding(10, 50, embeddings_initializer="uniform")) #Try "uniform" and "ones" initializer
model.compile('rmsprop', 'mse')

input_array = np.array([0,1,2,3,4,5,6,7,8,9])
output_array = model.predict(input_array)
print(output_array)
