from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (416, 416)
camera.rotation = 270
rawCapture = PiRGBArray(camera)
 
# allow the camera to warmup
time.sleep(0.1)
 
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	frame = f.array
	cv2.imshow("Frame", frame)
	#f.truncate(0)
	if 'a'==input("nyomj meg egy gombot"):
       		break