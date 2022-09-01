# # https://youtu.be/MF2AYo0SO6s
"""
Rename all files in a folder to numbers
"""

import os
import glob
import random

images_dir = 'renamed_to_numbers/images/'
annot_dir = 'renamed_to_numbers/annots/'


imagepath_list = glob.glob(os.path.join(images_dir, '*.jpg'))
random.Random(42).shuffle(imagepath_list)  #Setting the seed is important to make sure we get same sorting for images and annotations

padding = len(str(len(imagepath_list)))  #Number of digits to add for file number


for n, filepath in enumerate(imagepath_list, 1):
    os.rename(filepath,
              os.path.join(images_dir, '{:>0{}}.jpg'.format(n,padding)))
              
annotpath_list = glob.glob(os.path.join(annot_dir, '*.xml'))
random.Random(42).shuffle(annotpath_list)

for m, filepath in enumerate(annotpath_list, 1):
    os.rename(filepath,
              os.path.join(annot_dir, '{:>0{}}.xml'.format(m,padding)))
              