# https://youtu.be/owiqdzha_DE
"""
pip install easyocr

https://github.com/JaidedAI/EasyOCR
"""

import easyocr
import cv2

#For the first time it downloads the models for the languages chosen below. 
#Not all languages are compatible with each other so you cannot put
#multiple languages below
#reader = easyocr.Reader(['hi', 'te', 'en'])  #Hindi, telugu, and English
#The above gives error that Telugu is only compatible with English.

#So let us just use Hindi and English 
#To use GPU you need to have CUDA configured for the pytorch library.
reader = easyocr.Reader(['hi', 'en'], gpu=False)  #Hindi, telugu, and English

img = cv2.imread('images/SBI.jpg')


results = reader.readtext(img, detail=1, paragraph=False) #Set detail to 0 for simple text output
#Paragraph=True will combine all results making it easy to capture it in a dataframe. 


#To display the text on the original image or show bounding boxes
#we need the coordinates for the text. So make sure the detail=1 above, readtext.
# display the OCR'd text and associated probability
for (bbox, text, prob) in results:
    
    #Define bounding boxes
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    
    #Remove non-ASCII characters to display clean text on the image (using opencv)
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
   
    #Put rectangles and text on the image
    cv2.rectangle(img, tl, br, (0, 255, 0), 2)
    cv2.putText(img, text, (tl[0], tl[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# show the output image
cv2.imshow("Image", img)
cv2.waitKey(0)

#####################################################################

#Read multiple images in a directory, extract text using OCR and capture 
# it in a pandas dataframe. 
import cv2
import glob
import pandas as pd

#select the path
path = "images/english/*.*"
img_number = 1 
reader = easyocr.Reader(['en'])  #English

df=pd.DataFrame()
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    img= cv2.imread(file, 0)  #now, we can read each file since we have the full path
    results = reader.readtext(img, detail=0, paragraph=True) #Set detail to 0 for simple text output
    df = df.append(pd.DataFrame({'image': file, 'detected_text': results[0]}, index=[0]), ignore_index=True)
    img_number +=1   
    
    
    
    
    
    
    