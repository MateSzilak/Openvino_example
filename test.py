from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
#import numpy as np


camera = PiCamera()
camera.resolution = (416, 416)
camera.rotation = 270
rawCapture = PiRGBArray(camera)

time.sleep(0.2)# grab an image from the camera

camera.capture(rawCapture, format="bgr")
image = rawCapture.array

cv2.imshow("Image", image)
cv2.waitKey(0)


    #resized_image = cv2.resize(image,(480, 640), interpolation = cv2.INTER_CUBIC)
   # image_data = np.array(resized_image, dtype='f')

  # Normalization [0,255] -> [0,1]
   # image_data /= 255.

  # BGR -> RGB? The results do not change much
  # copied_image = image_data
  #image_data[:,:,2] = copied_image[:,:,0]
  #image_data[:,:,0] = copied_image[:,:,2]

  # Add the dimension relative to the batch size needed for the input placeholder "x"
#image_array = np.expand_dims(image, 0)  # Add batch dimension
    #print(type(image)) 
    # display the image on screen and wait for a keypress
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
   