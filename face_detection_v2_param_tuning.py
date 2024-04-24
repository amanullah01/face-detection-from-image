# Parameter tuning

import cv2
import matplotlib.pyplot as pt

cascade_classifier = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

image = cv2.imread('images/people.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detect_face = cascade_classifier.detectMultiScale(gray_image,1.1, minNeighbors=3, minSize=(20, 20))

for (x,y,w,h) in detect_face:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

pt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
pt.show()
