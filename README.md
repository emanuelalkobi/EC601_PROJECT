# EC601_PROJECT
Soccer Offside Detection
Product Name: 

     Soccer Offside Detection System
     
Mission Definition:

     Automatically detecting offside from game videos with visual based tools
     
Target Customer: 

     Soccer referees or football associations
     

------------------------------------------------------------------------------------------
Timeline

Sprint1(9/10 - 9/26): Established our mission and user stories

Sprint2(9/26 - 10/06): Applied a model to track the players

Sprint3(10/07 - 11/10): To build a model for tracking soccer ball

Sprint4(11/11 - 12/13): Combine the whole code together and track all the necessary parts.write the algorith to detect the offside based on the detections.


------------------------------------------------------------------------------------------
*Player Detection*

track_game.py is for detecting the players. It is based on a similar project:

            https://towardsdatascience.com/analyse-a-soccer-game-using-tensorflow-object-detection-and-opencv-e321c230e8f2
            
The "SSD_mobilenet_COCO" model is used, which is in the object_detection object.
Type the following to run it:
     
     python3 track_game.py

It will run with soccer_small.mp4

------------------------------------------------------------------------------------------
At soccer ball classifier there are 3 different classifiers for detecting a soccer ball.


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
