from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
from PIL import Image
import PIL.ImageOps


file = fetch_openml('mnist_784')
X = file.data
Y = file.target

count = pd.Series(Y).value_counts()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state= 9, train_size = 7500, test_size = 2500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial')
clf.fit(X_train_scaled, Y_train)
Y_predicted = clf.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_predicted)

import cv2 

capture = cv2.VideoCapture(0)
while (True):
    ret, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = frame.shape
    upperleft = int(width/2 - 55), int(height/2 - 55)
    bottomright = int(width/2 + 55), int(height/2 + 55)
    cv2.rectangle(frame, upperleft, bottomright, (0,128,0), 2)
    roi = frame[upperleft[1]: bottomright[1], upperleft[0]: bottomright[0]]
    PILImage = Image.fromarray(roi)
    imageconverted = PILImage.convert('L')
    imageconvertedresize = imageconverted.resize((28,28), Image.ANTIALIAS)
    imageconvertedresizeinverted = PIL.ImageOps.invert(imageconvertedresize)
    pixelfilter = 20
    minpixel = np.percentile(imageconvertedresizeinverted, pixelfilter)
    imagescaled = np.clip(imageconvertedresizeinverted - minpixel, 0, 255)
    maxpixel = np.max(imageconvertedresizeinverted)
    imagescaled = np.asarray(imagescaled)/maxpixel
    finalimage = np.array(imagescaled).reshape(1, 784)
    testprediction = clf.predict(finalimage)
    print(testprediction)
    cv2.imshow("recognition", frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break