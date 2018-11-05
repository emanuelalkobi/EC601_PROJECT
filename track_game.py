from __future__ import division
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
CWD_PATH = os.getcwd()

import detect_color as detect_color
import get_info as get_info
#colors to detect as shirts supported
colors=['red','yellow','green','blue']

NUM_CLASSES=90
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# Note: Model used for SSDLite_Mobilenet_v2
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

#check argument number
if ((len(sys.argv) < 2)):
    print("Did not receive a video file to analyze")
    sys.exit(0)

#check if vide to analyze exsist
filename= get_info.get_video_file(sys.argv[1])


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

category_index = label_map_util.create_category_index(categories)

#get teams names and colors
team1=get_info.get_name(1)
color1=get_info.get_color(1,colors)
team2=get_info.get_name(2)
color2=get_info.get_color(2,colors)
print("Teams information\n")
print("Team number 1 name :\n",team1)
print("Team number 1 colors :\n",color1)
print("Team number 2 name :\n",team2)
print("Team number 2 colors :\n",color2)

#intializing the web camera device
out = cv2.VideoWriter('ssoccer_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,360))

cap = cv2.VideoCapture(filename)

# Running the tensorflow session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   counter = 0
   print("--------------------------Start tracking video--------------------------\n")
   while (True):
   
      ret, image_np = cap.read()
      counter += 1
      if ret:
          h = image_np.shape[0]
          w = image_np.shape[1]

      if not ret:
        break
      if counter % 1 == 0:
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          class_to_delete=[]
          for i, x in np.ndenumerate(classes):
              if (x!=1 and x!=37):    #classe 1 person class 37 sports ball
                  class_to_delete.append(i[1])
          classes=np.delete(classes,class_to_delete,1)
          scores=np.delete(scores,class_to_delete,1)
          boxes=np.delete(boxes,class_to_delete,1)
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=3,
              min_score_thresh=0.3)
          frame_number = counter
          loc = {}
          for n in range(len(scores[0])):
             if scores[0][n] > 0.60:
                # Calculate position
                ymin = int(boxes[0][n][0] * h)
                xmin = int(boxes[0][n][1] * w)
                ymax = int(boxes[0][n][2] * h)
                xmax = int(boxes[0][n][3] * w)

                # Find label corresponding to that class
                for cat in categories:
                    if cat['id'] == classes[0][n]:
                        label = cat['name']

                ## extract every person
                if label == 'person':
                    #crop them
                    crop_img = image_np[ymin:ymax, xmin:xmax]
                    color = detect_color.detect_team(crop_img,color1,color2)
                    if color != 'not_sure':
                        coords = (xmin, ymin)
                        if color == color1:
                             loc[coords] = team1
                        elif color==color2:
                            loc[coords] = team2
                if label == 'sports ball':
                    coords = (xmin, ymin)
                    loc[coords] = 'Soccer Ball'
        ## print color next to the person
          for key in loc.keys():
            text_pos = str(loc[key])
            cv2.putText(image_np, text_pos, (key[0], key[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2) # Text in black
      
      cv2.imshow('image', image_np)
      out.write(image_np)
       
      if cv2.waitKey(10) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          break
out.release()


