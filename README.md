# *EC601_PROJECT-Soccer Offside Detection*

Product Name: 

     Soccer Offside Detection System
     
Mission Definition:

     Automatically detecting offside from game videos with visual based tools
     
Target Customer: 

     Soccer referees or football associations
     
Running command:
     
     python3 track_game.py -i <input video> -o <result video> 
     
     


## Game Detection

Track_game.py is the python file that track the soccer game :

* Detecting players:
------------------------------------------------------------------------------------------------------------------------------
For detecting the players we used Tensorflow Object Detection API.We used the pre trained model over the COCO dataset.
COCO data set is a large-scale object detection dataset. 

*    COCO data set properties :
     - Object segmentation
     - Recognition in context
     - 330K images (>200K labeled)
     - 1.5 million object instances
     - 80 object categories
     - Object segmentation
     - 91 stuff categories
     
*    In our project we care about 2 classes only :
     - person
     - sports ball
     



Our code is based on the next website:

            https://towardsdatascience.com/analyse-a-soccer-game-using-tensorflow-object-detection-and-opencv-e321c230e8f2
    
    
* Detecting soccer ball:
------------------------------------------------------------------------------------------------------------------------------
In order to detect the soccer ball that it is a harder task we use the next 2 methods in this order(this means that when the a method failed to detect the soccer ball we are trying to detect it using the next method):

1.SSD_mobilenet_COCO model



2.Motion tracking

The motion tracking algorith is based on the next project with modifications to detect a soccer ball:
     
     https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
     

track_game.py combines the result of these 2 methods for detecting the soccer ball.

We implemented a Template matching algorithm too but we remove it as we did not get better results with this algoritm.
The template matching algorithm is based on the link below:
     
     https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
     
     eg. 'python track_game.py -i fifa_videos/angle2.mp4 -o angle2_ball3.mp4 -t template/ball3.png'

![gif](https://github.com/emanuelalkobi/EC601_PROJECT/blob/master/results/gif.gif)

