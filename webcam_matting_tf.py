import cv2
from cv2 import VideoCapture, imshow, waitKey
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch

model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3") # or "resnet50"

rec = [ tf.constant(0.) ] * 4         # Initial recurrent states.
downsample_ratio = tf.constant(1)  # Adjust based on your video.

cam = VideoCapture(0)  #set the port of the camera as before

sz = 512

# src is of shape [Batch, Height, Width, Channel], not [B, C, H, W]!
# note cv2 reads images as BGR

batch_size = 4
height = width = 512
channels = 3

buffer_frames = 10
num_frames = 110

infer_array = np.zeros(shape=(batch_size,height,width,channels)).astype(np.float64)

for i in range(num_frames):

    retval, image = cam.read() #return a True bolean and and the image if all go right

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if i == (buffer_frames - 1):
        batch_counter = 0

    if i > (buffer_frames - 1):

        # if retval:
        #     image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        if i == buffer_frames:
            im_width, im_height, im_depth = image.shape   # Get dimensions
            left = int((im_width - sz)/2)
            top = int((im_height - sz)/2)
            right = int((im_width + sz)/2)
            bottom = int((im_height + sz)/2)

        im = image[left:right,top:bottom,0:im_depth]
        im = cv2.flip(im, 1)

        im = im.astype(np.float64)/255

        if batch_counter < batch_size:

            infer_array[batch_counter] = im
            batch_counter = batch_counter + 1

        else:
            new_array = np.zeros(shape=(batch_size+1,height,width,channels)).astype(np.float64)
            new_array[0:batch_size] = infer_array
            new_array[-1] = im
            infer_array = np.delete(new_array, 0, axis=0)

        for src in tf.convert_to_tensor(infer_array):  # src is of shape [B, H, W, C], not [B, C, H, W]!
            src = np.expand_dims(src,axis=0)
            out = model([src, *rec, downsample_ratio])
            fgr, pha, *rec = out['fgr'], out['pha'], out['r1o'], out['r2o'], out['r3o'], out['r4o']

        print('loop')

        # if i == (num_frames - 1):
        #     imshow(str(retval),cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        #     cv2.setWindowProperty(str(retval), cv2.WND_PROP_TOPMOST, 0)
        #     cv2.waitKey(1)
        #     print('end loops')
        # else:
        #     imshow(str(retval),cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        #     cv2.setWindowProperty(str(retval), cv2.WND_PROP_TOPMOST, 1)
        #     cv2.waitKey(1)

cam.release() #Closes video file or capturing device.