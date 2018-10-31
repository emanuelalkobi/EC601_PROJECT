import matplotlib.pyplot as plt
import numpy as np
from darkflow.net.build import TFNet
import cv2
import pprint as pp
import sys

def boxing(original_img , predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
        
        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
        
    return newImage

#loading model
options = {"model": "my_cfg/cfg/yolo_custom.cfg",
           "load": -1,
           "gpu": 1.0}
tfnet2 = TFNet(options)
tfnet2.load_from_ckpt()

print(sys.argv[1])
name=str.split(sys.argv[1],'.')
original_img = cv2.imread(sys.argv[1])
results = tfnet2.return_predict(original_img)
cv2.imwrite(name[0]+'_result.'+name[1],boxing(original_img, results))
print("finished tracking")
