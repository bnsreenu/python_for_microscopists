# https://youtu.be/Fuve1nAdm8k
"""
Face and eye detection using opencv (Haar Cascade classificaion)

Download face and eye models:
Go to these links, click on RAW and save as... otherwise you'd be saving html files of Github page. '
    https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    
"""

#Verify on static image

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades_models/haarcascade_eye.xml')

img = cv2.imread('obamas.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#First detect face and then look for eyes inside the face.
#Multiscale refers to detecting objects (faces) at multiple scales. 
faces = face_cascade.detectMultiScale(gray, 1.3, 5) #scaleFactor = 1.3, minNeighbors = 3
#Above faces returns a list of rectangles. For Obama image we only have 1 face
#so it return a tuplr of (1,4) --> 1 represents one rectangle and 4 represents
#the x,y,w,h values that define the square.

#Obamas image with both barack and Michelle it returns a tuple of (2,4) --> 2 faces.

#For each detected face now detect eyes. 
#For emotion detection this is where we update code to identify facial emotion
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)   #Draw red bounding box around the face
    roi_gray = gray[y:y+h, x:x+w] #Original gray image but only the detected face part
    roi_color = img[y:y+h, x:x+w] #Original color image but only the detected face part. For display purposes
    eyes = eye_cascade.detectMultiScale(roi_gray) #Use the gray face image to detect eyes
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) #Draw green bounding boxes around the eyes

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


########################################################

#Apply the above logic to a live video
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades_models/haarcascade_eye.xml')

#Check if your system can detect camera and what is the source number
# cams_test = 10
# for i in range(0, cams_test):
#     cap = cv2.VideoCapture(i)
#     test, frame = cap.read()
#     print("i : "+str(i)+" /// result: "+str(test))


cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    #First detect face and then look for eyes inside the face.
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:      #Press Esc to stop the video
        break

cap.release()
cv2.destroyAllWindows()