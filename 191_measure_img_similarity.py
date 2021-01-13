# https://youtu.be/16s3Pi1InPU
"""
Comparing images using ORB/SIFT feature detectors
and structural similarity index. 

@author: Sreenivas Bhattiprolu
"""


from skimage.metrics import structural_similarity
import cv2

#Works well with images of different dimensions
def orb_sim(img1, img2):
  # SIFT is no longer available in cv2 so using ORB
  orb = cv2.ORB_create()

  # detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  # define the bruteforce matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  #perform matches. 
  matches = bf.match(desc_a, desc_b)
  #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
  similar_regions = [i for i in matches if i.distance < 50]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)


#Needs images to be same dimensions
def structural_sim(img1, img2):

  sim, diff = structural_similarity(img1, img2, full=True)
  return sim

img00 = cv2.imread('images/monkey_distorted.jpg', 0)
img01 = cv2.imread('images/monkey_rotated.jpg', 0)

img1 = cv2.imread('images/BSE.jpg', 0)  # 714 x 901 pixels
img2 = cv2.imread('images/BSE_noisy.jpg', 0)  # 714 x 901 pixels
img3 = cv2.imread('images/BSE_smoothed.jpg', 0)  # 203 x 256 pixels
img4 = cv2.imread('images/different_img.jpg', 0)  # 203 x 256 pixels

orb_similarity = orb_sim(img1, img3)  #1.0 means identical. Lower = not similar

print("Similarity using ORB is: ", orb_similarity)
#Resize for SSIM
from skimage.transform import resize
img5 = resize(img3, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)

ssim = structural_sim(img1, img5) #1.0 means identical. Lower = not similar
print("Similarity using SSIM is: ", ssim)
