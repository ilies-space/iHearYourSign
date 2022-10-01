import cv2
from Modules.HandTrackingModule import HandDetector
from Modules.ClassificationModule import Classifier
import numpy as np
import math
import os

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

arabicLater = ["ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك"
               , "ل", "م", "ن", "ه", "و", "ي"]

predictedCharactersCollector = []

predictionHolder = []

currentPredicatedLaterIndex = 0

def handle_prediction_result():
    global predictedCharactersCollector, predictionHolder, currentPredicatedLaterIndex
    # if the predication is the same 8 time arrow than save it otherwise restart
    predicted_character = arabicLater[currentPredicatedLaterIndex]
    time_of_recheck = 8
    print("----> ", predicted_character)
    if len(predictedCharactersCollector) == 0:
        predictedCharactersCollector.append(predicted_character)
    else:
        if len(predictedCharactersCollector) < time_of_recheck:
            if predictedCharactersCollector[-1] == predicted_character:
                predictedCharactersCollector.append(predicted_character)
                if len(predictedCharactersCollector) == time_of_recheck/2:
                    speak(predicted_character)
            else:
                predictedCharactersCollector = []
        else:
            predictionHolder.append(predicted_character)
            predictedCharactersCollector = []
            os.system("mpg321 beep.mp3")

    print("handle_prediction_result: ", prediction)

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
                handle_prediction_result()

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

        except:
            print("Error")

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(50)

    # Read current predicated later on press R rom keyboard
    if key == ord("r"):
        singleWord = ''.join(predictionHolder)
        print("TO SPEAK: -------> ",singleWord)
        speak(singleWord)
        if key == ord("c"):
            predictionHolder = [""]