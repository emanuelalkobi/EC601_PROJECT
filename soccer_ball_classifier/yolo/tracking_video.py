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



def main():
    #loading model
    options = {"model": "my_cfg/cfg/yolo_custom.cfg",
               "load": -1,
               "gpu": 1.0}
    tfnet2 = TFNet(options)
    tfnet2.load_from_ckpt()
    if (len(sys.argv)!=2):
        print("please only insert a path to a video file")
        exit(1)
    try:
        with open(sys.argv[1]) as file:
            pass
    except IOError as e:
        print("Unable to open file") #Does not exist OR no read permissions
        exit(1)

    print(sys.argv[1])
    name=str.split(sys.argv[1],'.')
    cap = cv2.VideoCapture(sys.argv[1])
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(name[0]+'_result.'+name[1],fourcc, 20.0, (int(width), int(height)))
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
         
            # Display the resulting frame
            cv2.imshow('Frame',frame)
            results = tfnet2.return_predict(frame)
            new_frame = boxing(frame, results)
            out.write(new_frame) 
             # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
 
        # Break the loop
        else: 
            break
 
    # When everything done, release the video capture object
    cap.release()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("finished tracking")

main()
