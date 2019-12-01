# USAGE
# python openvino_real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages

import numpy as np
import argparse
import time
import cv2
import datetime

import os


fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0

cap = cv2.VideoCapture("GOPR1452.MP4")
os.makedirs("/home/pi/Python_dev/Openvino_example/GOPR1452")
file = open("/home/pi/Python_dev/Openvino_example/GOPR1452/log.txt", "w")


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




counter = 0

#initialize can
#os.system('sudo ip link set can0 type can bitrate 500000')
#os.system('sudo ifconfig can0 up')
#can0 = can.interface.Bus(channel = 'can0', bustype = 'socketcan_ctypes')

#can_counter = 0

# loop over the frames from the video stream
while(cap.isOpened()):    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    
    ret, frame = cap.read()
    if ret == True:
        t1 = time.perf_counter()
        print(1)
        frame = frame[0:580, 320:960]
        
        
    # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
        net.setInput(blob)
        detections = net.forward()

        label = ""
        startX = 0
        startY = 0
        endX = 0
        endY = 0
        confidence = 0
        StringPuffer = ''
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
                StringPuffer = StringPuffer + ' ' + str(label)+ ' ' +str(confidence)+ ' startX: ' + str(startX)+ ' startY: ' + str(startY)+ ' endX: ' + str(endX)+ ' endY: ' + str(endY)

    # show the output frame
        #cv2.putText(frame, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.imwrite('/home/pi/Python_dev/Openvino_example/GOPR1452/%d.jpg' % (counter), frame)
        counter = counter + 1
        file.write("%s\n" % StringPuffer)
        
file.close()
 