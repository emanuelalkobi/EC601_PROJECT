#import matplotlib.pyplot as plt
#import numpy as np

from darkflow.net.build import TFNet


import cv2

options = {"model": "cfg/yolo_custom.cfg",
           "load": "bin/yolo.weights",
           "batch": 8,
           "epoch": 120,
           "gpu": 1.0,
           "train": True,
           "annotation": "./screen_shots_ready/annotations/",
           "dataset": "./screen_shots_ready/images/"}

# options = {"model": "cfg/yolo_custom.cfg",
#            "load": "bin/yolo.weights",
#            "batch": 8,
#            "epoch": 100,
#            #"gpu": 1.0,
#            "train": True,
#            "annotation": "./soccer_ball_data/annotations/",
#            "dataset": "./soccer_ball_data/images/"}

tfnet = TFNet(options)

tfnet.train()
