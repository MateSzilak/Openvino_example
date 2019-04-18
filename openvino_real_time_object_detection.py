# USAGE
# python openvino_real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages

import numpy as np
import argparse
import time
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera

from picamera import PiCamera

fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('./MobileNetSSD_deploy.prototxt', './MobileNetSSD_deploy.caffemodel')

# specify the target device as the Myriad processor on the NCS
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")

camera = PiCamera()
camera.resolution = (300, 300)
camera.rotation = 270
rawCapture = PiRGBArray(camera, size=(300, 300))


# loop over the frames from the video stream
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = f.array
    t1 = time.perf_counter()

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.2:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output frame
    cv2.putText(frame, fps, (300-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    f.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break



    # FPS calculation
    framecount += 1
    if framecount >= 15:
        fps       = "{:.1f} FPS".format(time1/15)
        framecount = 0
        time1 = 0
        time2 = 0
    t2 = time.perf_counter()
    elapsedTime = t2-t1
    time1 += 1/elapsedTime
    time2 += elapsedTime
    # update the FPS counter

# stop the timer and display FPS information

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()