
# https://youtu.be/cTrAlg0OWUo
"""
@author: Sreenivas Bhattiprolu
"""

import tensorflow as tf

#To see if you have any active GPUs accessible to tensorflow
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")



#For TF 1.0
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))


##For TF 2.0 try this
#import tensorflow as tf
#tf.config.gpu.set_per_process_memory_fraction(0.75)
#tf.config.gpu.set_per_process_memory_growth(True)
