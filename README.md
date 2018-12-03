# *EC601_PROJECT-Soccer Offside Detection*

Product Name: 

     Soccer Offside Detection System
     
Mission Definition:

     Automatically detecting offside from game videos with visual based tools
     
Target Customer: 

     Soccer referees or football associations
     
Running command:
     
     python3 track_game.py -i <input video> -o <result video> 
     
     


*Game Detection*

Track_game.py is the current result:

For detecting the players we used the SSD_mobilenet_COCO. It is based on a similar project:

            https://towardsdatascience.com/analyse-a-soccer-game-using-tensorflow-object-detection-and-opencv-e321c230e8f2
            

In order to detect the soccer ball that it is a harder task we use the next 3 methods in this order(this means that when the a method failed to detect the soccer ball we are trying to detect it using the next method):

1.SSD_mobilenet_COCO model

2.Template matching
The template matching algorithm is based on the link below:
     
     https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
     
     eg. 'python track_game.py -i fifa_videos/angle2.mp4 -o angle2_ball3.mp4 -t template/ball3.png'

3.Motion tracking

     The motion tracking algorith is based on the next project with modifications to detect a soccer ball:
     
               https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

track_game.py combines the result of these 3 methods for detecting the soccer ball.

![gif](https://github.com/emanuelalkobi/EC601_PROJECT/blob/master/results/gif.gif)

