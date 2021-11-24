import cv2
from cv2 import VideoCapture, imshow, waitKey
import numpy as np
import matplotlib.pyplot as plt
cam = VideoCapture(0)  #set the port of the camera as before

sz = 512

for i in range(100):
    retval, image = cam.read() #return a True bolean and and the image if all go right
    if i == 0:
        width, height, depth = image.shape   # Get dimensions
        left = int((width - sz)/2)
        top = int((height - sz)/2)
        right = int((width + sz)/2)
        bottom = int((height + sz)/2)

    im = image[left:right,top:bottom,0:depth]
    im = cv2.flip(im, 1)
    
    imshow(str(retval),im)
    cv2.setWindowProperty(str(retval), cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)


cam.release() #Closes video file or capturing device.