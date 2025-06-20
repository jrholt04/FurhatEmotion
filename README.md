# FurhatEmotion

## Set Up
1. create a conda env that is running python 3.9: conda create -n <envname> python=3.9
2. install deep face: pip install deepface
3. install open cv: pip install opencv-python
4. install tf-keras: pip install tf-keras
5. install zero-mq: pip install pyzmq
6. install vader: pip install vaderSentiment

## 🔍 YOLOv3 Weights Required

The file `yolov3.weights` is **not included** in this repository because of its large size.

To run the detection successfully, please:

1. Download the YOLOv3 weights from the official source:
   [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

2. Place the downloaded file in the following directory: faceSkillDetectionServer

## Git
1. run this to not track your ip in the launch.json: git update-index --assume-unchanged faceSkillDetectionServer/launch.json
