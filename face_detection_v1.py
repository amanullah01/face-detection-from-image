import cv2
import matplotlib.pyplot as plt


# OpenCV has a lot of pre-trained classifiers for face detection, eye detection etc.
# several positive and negative samples to train the model (Viola-Jones algorithm)
cascade_classifier = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

image = cv2.imread('images/face.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)
detect_faces = cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

print(detect_faces)
for(x, y, width, height) in detect_faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 10)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.imshow(gray_image, cmap='gray')
plt.show()
