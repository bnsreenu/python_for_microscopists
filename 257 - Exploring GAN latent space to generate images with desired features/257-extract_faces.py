# https://youtu.be/iuQ_f3W5Ttk
"""

Load images, extract faces and save them to a new directory.

Dataset: https://www.kaggle.com/jessicali9530/celeba-dataset

Haarcascade models...
https://github.com/opencv/opencv/tree/master/data/haarcascades
"""

#######################################################################

#Extract faces
import cv2
import glob

face_cascade = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')

#select the path
path = "data/*.*"
img_number = 1  #Start an iterator for image number.

for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    img= cv2.imread(file, 1)  #now, we can read each file since we have the full path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        for (x,y,w,h) in faces:
            roi_color = img[y:y+h, x:x+w] 
        resized = cv2.resize(roi_color, (128,128))
        cv2.imwrite("faces/"+str(img_number)+".jpg", resized)
    except:
        print("No faces detected")
    
    
    img_number +=1     
