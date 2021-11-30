import cv2
from cv2 import VideoCapture, imshow, waitKey
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

cam = VideoCapture(0)  #set the port of the camera as before

rec = [ tf.constant(0.) ] * 4         # Initial recurrent states.
downsample_ratio = tf.constant(1)  # Adjust based on your video.


model = tf.keras.models.load_model('rvm_mobilenetv3_tf')
# model = tf.function(model)

# for src in YOUR_VIDEO:  # src is of shape [B, H, W, C], not [B, C, H, W]!
#     out = model([src, *rec, downsample_ratio])
#     fgr, pha, *rec = out['fgr'], out['pha'], out['r1o'], out['r2o'], out['r3o'], out['r4o']

# src is of shape [Batch, Height, Width, Channel], not [B, C, H, W]!
# note cv2 reads images as BGR

batch_size = 4
height = width = 512
channels = 3

buffer_frames = 10
num_frames = 110

# infer_array = np.zeros(shape=(batch_size,height,width,channels)).astype(np.float64)

for i in range(num_frames):

    retval, image = cam.read() #return a True bolean and and the image if all go right

    if i > (buffer_frames):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        im = image
        im = cv2.flip(im, 1)

        im = im.astype(np.float64)/255

        src = im

        out = model([src, *rec, downsample_ratio])
        fgr, pha, *rec = out['fgr'], out['pha'], out['r1o'], out['r2o'], out['r3o'], out['r4o']


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