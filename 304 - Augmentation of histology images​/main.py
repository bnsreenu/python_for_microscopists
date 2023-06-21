# https://youtu.be/SuDtHqtC5OE
"""

Random Stain Normalization and Augmentation (RandStainNA) is a hybrid framework 
designed to fuse stain normalization and stain augmentation to generate more realistic stain variations. 
It incorporates randomness to SN by automatically sorting out a random virtual template from
pre-estimated stain style distributions. More specifically, from the perception of
SN’s viewpoint, stain styles ‘visible’ to the deep neural network are enriched
in the training stage. Meanwhile, from the perception from the SA’s viewpoint,
RandStainNA imposes a restriction on the distortion range and consequently,
only a constrained practicable range is ‘visible’ to CNN. T

https://github.com/yiqings/RandStainNA

https://arxiv.org/abs/2206.12694

This python file calls methods from the randstainna file. 
Also, you need to provide a yaml file with approrite stats. that act as the template
image for color transformation. The yaml file can be generated using the 
datasets_statistics_V1.0.py file in the preprocess-statistics directory.

"""

import os
from randstainna import RandStainNA
import cv2


# Setting: is_train = False
randstainna = RandStainNA(
    yaml_file = './preprocess-statistics/output/random_images.yaml',
    std_hyper = 0.0,
    distribution = 'normal',
    probability = 1.0,
    is_train = False
)

dir_path = 'data/original/'
img_list = os.listdir(dir_path)


save_dir_path = 'data/augmented'
if not os.path.exists(save_dir_path):
    os.mkdir(save_dir_path)

for img_path in img_list:
    img = randstainna(cv2.imread(dir_path+img_path))
    save_img_path = save_dir_path + '/{}'.format(img_path.split('/')[-1])
    cv2.imwrite(save_img_path,img)


