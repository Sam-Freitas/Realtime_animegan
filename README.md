# Realtime_animegan
This is a simple learning experiment to see if i can take a webcam video and apply differnt types of Networks to the data without writing to the disk, made for MacOS

------------------------------------------------------------------------------------------

# Usage
Make sure you have pytorch-gpu and cuda properly installed with only a single webcam input 

run: 
```
python realtime_anime.py
```

# ML models and referenced code

Robust Video Matting - pytorch:

  https://github.com/PeterL1n/RobustVideoMatting
  
Animegan2 - pytorch:

  https://github.com/bryandlee/animegan2-pytorch
  
Opening a camera as a single thread:

  https://github.com/jackhanyuan/RobustVideoMatting/blob/master/inference_camera.py

## TODO

Get it working on pure tensorflow (pytorch is not supported on metal -- macOS -- right now)
