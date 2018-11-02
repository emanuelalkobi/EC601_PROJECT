import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from darkflow.net.build import TFNet
import cv2

def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.1:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
            #newImage = cv2.putText(newImage, label, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 230, 0), 1, cv2.LINE_AA)



    return newImage

def main():
    options = {"model": "cfg/yolo_custom.cfg",
               "load": -1
               # "gpu": 1.0
               }

    tfnet2 = TFNet(options)

    tfnet2.load_from_ckpt()

    results_list = []

    for i in range(1, 292):
        original_img = cv2.imread(os.getcwd() + "/screen_shots_ready/images/FIFA" + str(i).zfill(3) + ".jpg")
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        results = tfnet2.return_predict(original_img)
        if results:
            results_list.append([i, original_img, results])
        print(i, results)

    for i in results_list:
        #fig, ax = plt.subplots(figsize=(20, 20))
        #ax.imshow(boxing(i[0], i[1]))
        new_image = boxing(i[1], i[2])
        cv2.imwrite(os.getcwd() + '/testing_results/' + str(i[0]) + '.jpg', new_image)

main()

