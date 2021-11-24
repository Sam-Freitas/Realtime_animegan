import cv2
from cv2 import VideoCapture, imshow, waitKey
import numpy as np
from sys import platform
import PIL
from tqdm import tqdm
import torch

def take_center_N_pixels(in_img,im_size):

    im_width, im_height, im_depth = in_img.shape   # Get dimensions
    left = int((im_width - im_size)/2)
    top = int((im_height - im_size)/2)
    right = int((im_width + im_size)/2)
    bottom = int((im_height + im_size)/2)

    out_img = in_img[left:right,top:bottom,0:im_depth]

    return out_img

this_platform = platform

sz = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
if this_platform == 'darwin':
    import coremltools as ct
    matting_model = ct.models.model.MLModel('rvm_mobilenetv3_1280x720_s0.375_int8.mlmodel')
else:
    matting_model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3") # or "resnet50"

anime_model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2", device=device).eval()
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device, size=sz)

r1, r2, r3, r4 = None, None, None, None

cam = VideoCapture(0)  #set the port of the camera as before

batch_size = 4
height,width = 720,1280
channels = 3

camera_buffer_frames = 10
num_frames = 200

for i in tqdm(range(num_frames)):

    retval, image = cam.read() #return a True bolean and and the image if all go right

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if i == (camera_buffer_frames - 1):
        batch_counter = 0

    if i > (camera_buffer_frames - 1):

        im = cv2.flip(image, 1)

        src2 = PIL.Image.fromarray(im)

        if r1 is None:
            # Initial frame, do not provide recurrent states.
            inputs = {'src': src2}
        else:
            # Subsequent frames, provide recurrent states.
            inputs = {'src': src2, 'r1i': r1, 'r2i': r2, 'r3i': r3, 'r4i': r4}

        matting_outputs = matting_model.predict(inputs)

        fgr = matting_outputs['fgr']  # PIL.Image.
        pha = matting_outputs['pha']  # PIL.Image.

        r1 = matting_outputs['r1o']  # Numpy array.
        r2 = matting_outputs['r2o']  # Numpy array.
        r3 = matting_outputs['r3o']  # Numpy array.
        r4 = matting_outputs['r4o']  # Numpy array.
        
        out_alpha = ((np.array(pha)>128)*255).astype(np.uint8)
        out_alpha_rgb = cv2.cvtColor(out_alpha,cv2.COLOR_GRAY2RGB)

        out_img = cv2.bitwise_and(im,out_alpha_rgb)

        out_img_small = take_center_N_pixels(out_img,sz)

        in_img = PIL.Image.fromarray(out_img_small)

        # img = Image.open(...).convert("RGB")
        out = face2paint(model=anime_model, img=in_img, size=sz)

        if i == (num_frames - 1):
            imshow(str(retval),cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR))
            cv2.setWindowProperty(str(retval), cv2.WND_PROP_TOPMOST, 0)
            cv2.waitKey(1)
            print('end loops')
        else:
            imshow(str(retval),cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR))
            cv2.setWindowProperty(str(retval), cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)

cam.release() #Closes video file or capturing device.