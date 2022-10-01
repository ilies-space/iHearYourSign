import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from Text2Speech import speak

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
counter = 0
folder = "Data/T"

labels = ["Alif", "Baa", "Taa", "Thaa", "Jim", "Ha", "Kha", "Dal", "Dhal", "Ra", "Zay", "Sin", "Shin", "Sad", "Dad",
          "Tta", "Dhaa", "Ayn", "Ghaa", "Faa", "Kaa", "Kaf", "Laa", "Mim", "Noun", "Haa", "Waaw", "Yaa"]

arabicLater = ["أ", "ب", "ت"]

currentPredicatedLaterIndex = 0

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    if hands:
        try:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

                currentPredicatedLaterIndex = index

                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCroped", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        except:
            print("Error")

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    # Read current predicated later on press R rom keyboard
    if key == ord("r"):
        speak(arabicLater[currentPredicatedLaterIndex])
