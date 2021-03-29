# https://youtu.be/syJZxDtLujs
"""
Note: Importing segmentation models library may give you generic_utils 
error on TF2.x

If you get an error about generic_utils...

change 
keras.utils.generic_utils.get_custom_objects().update(custom_objects) 
to 
keras.utils.get_custom_objects().update(custom_objects) 
in 
.../lib/python3.7/site-packages/efficientnet/__init__.py 

For Anaconda users:
Use this code to find out the location of site-packages directory 
under your current environment in anaconda. 

from distutils.sysconfig import get_python_lib
print(get_python_lib())


For Colab users:
You can click on the direct link provided on Colab for __init__.py and edit it.
Remember to restart the runtime after editing the file. 

Alternatively you can work with Tensorflow 1.x that doesn't throw
the generic_utils error. 
In google colab, add this as your first line.
%tensorflow_version 1.x
(Or just create a new environment in your local IDE to use TF1.x)
"""

#For Anaconda users, run the following to find out the location of site-packages 
# on your system. 
from distutils.sysconfig import get_python_lib
print(get_python_lib())