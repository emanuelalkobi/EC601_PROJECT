# EC601_PROJECT
Soccer Offside Detection
Product Name: 

     Soccer Offside Detection System
     
Mission Definition:

     Automatically detecting offside from game videos with visual based tools
     
Target Customer: 

     Soccer referees or football associations
     
Running command:
     
     python3 track_game.py -i <input video> -o <result video> -t <template image>
     
     

     

------------------------------------------------------------------------------------------
Timeline

Sprint1(9/10 - 9/26): Established our mission and user stories

Sprint2(9/26 - 10/06): Applied a model to track the players

Sprint3(10/07 - 11/10): To build a model for tracking soccer ball

Sprint4(11/11 - 12/13): Combine the whole code together and track all the necessary parts.write the algorith to detect the offside based on the detections.


------------------------------------------------------------------------------------------
*Game Detection*

Track_game.py is the main file and his aim is to detect the players and the soccer ball.

For detecting the players we used the SSD_mobilenet_COCO. It is based on a similar project:

            https://towardsdatascience.com/analyse-a-soccer-game-using-tensorflow-object-detection-and-opencv-e321c230e8f2
            

In order to detect the soccer ball that it is a harder task we use the next 3 methods in this order(this means that when the a method failed to detect the soccer ball we are trying to detect it using the next method):

1.SSD_mobilenet_COCO model

2.Template matching

     The template matching algorithm is based on the link below:
     https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
     
     eg. 'python track_game.py -i fifa_videos/angle2.mp4 -o angle2_ball3.mp4 -t template/ball3.png'
     To save time, angle2_ball3.mp4 has already been saved here.   

3.Motion tracking

     The motion tracking algorith is based on the next project with modifications to detect a soccer ball:
     
               https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/




------------------------------------------------------------------------------------------

At soccer ball classifier there are 3 different classifiers for detecting a soccer ball.

This classifiers are for studying purpose only.


 #yolo

 This classifier use the yolo and darkflow algorith in order to detect items in a image.
 We train it in order to detect soccer balls at images and videos.
 there are 2 files:

      1.track_image.py <image path>
      2.track_video.py <video path>
      
      the output of both algorithms is a new video/image with a result after their name with the output of the algorithm
      need to run the python script from the sub folder soccer_ball_classifier/yolo/
      
      based on:
      https://towardsdatascience.com/yolov2-to-detect-your-own-objects-soccer-ball-using-darkflow-a4f98d5ce5bf
      

#yolo with FIFA game

This classifier was also trained with yolo and darkflow, but with images of FIFA video games. Detailed instructions can be found inside the directory.


#inception

This classifier use the ssd_inception algorithm to detect soccer ball in images.

     1. Put the images you want to test into 'testimg' folder.
     2. Run 'inception_test.py'.
     3. The output of this algorithms is stored in 'testresult' folder.


      
