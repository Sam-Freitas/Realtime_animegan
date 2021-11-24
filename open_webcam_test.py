import cv2
from cv2 import VideoCapture, imshow, waitKey
import numpy as np
import matplotlib.pyplot as plt
cam = VideoCapture(0)  #set the port of the camera as before

sz = 512

num_frames = 1000

for i in range(num_frames):
    retval, image = cam.read() #return a True bolean and and the image if all go right

    im = cv2.flip(image, 1)
    
    imshow(str(retval),im)
    cv2.setWindowProperty(str(retval), cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)


cam.release() #Closes video file or capturing device.