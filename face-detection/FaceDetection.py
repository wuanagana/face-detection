 # Import required modules
import cv2
import math
import time
import argparse
import numpy as np
from graphviz import Source

def getFaceBox(net, frame, conf_threshold):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (frameWidth, frameHeight), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    confidences = []
    global first,counterFps

    for i in range(0, detections.shape[2]):
        print("------------------------");
        print(str(i) + "| Det[0]: " + str(detections[0, 0, i, 0]));
        print(str(i) + "| ClassID: "  + str(detections[0, 0, i, 1]));
        print(str(i) + "| Confidence: "  + str(detections[0, 0, i, 2]));
        print(str(i) + "| xMin: "  + str(detections[0, 0, i, 3]));
        print(str(i) + "| yMin: "  + str(detections[0, 0, i, 4]));
        print(str(i) + "| xMax: "  + str(detections[0, 0, i, 5]));
        print(str(i) + "| yMax: "  + str(detections[0, 0, i, 6]));
        print("------------------------");
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:

            box = detections[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
            (startX, startY, endX, endY) = box.astype("int")
            if(detections[0, 0, i, 3] < 1 and detections[0, 0, i, 4] < 1 and detections[0, 0, i, 5] < 1 and detections[0, 0, i, 6] < 1):
                               #  x       y         width            height
                bboxes.append([startX, startY, (endX - startX), (endY - startY)])
                confidences.append(confidence)
    return bboxes, confidences




faceProto = "models/nvidia_deploy.prototxt"
faceModel = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"

first = True

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

faceNet = cv2.dnn.readNetFromCaffe(faceProto, faceModel)

cap = cv2.VideoCapture("videoTest/SinglePassage.264")

padding = 20
start_time = time.time()
counterFps = 0

while  True:

    counterFps = counterFps + 1;

    # Read 1 frame
    rc,frame = cap.read()
    if rc is not True:
        break

    frame = cv2.resize(frame, (1980, 1020))

    pressedKey = cv2.waitKey(1)
    if pressedKey == ord('q'):
        break

    if counterFps == 49:

        bboxes, confidences = getFaceBox(faceNet, frame, 0.8)

        for bbox in bboxes:

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255) ,2)
            cv2.putText(frame, str(confidences),(int(bbox[0])-25, int(bbox[1])-10),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 2)
            cv2.imwrite("49Frame.jpg",frame)

cap.release()
cv2.destroyAllWindows()
