# EC601_PROJECT
Soccer Offside Detection
**Product Name:** 

     Soccer Offside Detection System
     
**Mission Definition:**

     Automatically detecting offside from game videos with visual based tools
     
**Target Customer: **

     Soccer referees or football associations
     

------------------------------------------------------------------------------------------
**Timeline**

Sprint2(9/26 - 10/06): review and investigate background ,look for algorithms for tracking.

Sprint3(10/07 - 11/10): build a classifier for a soccer ball.combine to get the best classifier

Sprint4(11/11 - 12/13): run the whole code together and track all the necessary parts.write the algorith to detect the offside based on the detections.


------------------------------------------------------------------------------------------
**track players**
track_game.py is based on:

            https://towardsdatascience.com/analyse-a-soccer-game-using-tensorflow-object-detection-and-opencv-e321c230e8f2
            
we are trcking only persons and soccer ball objects and then use opencv to identify between 2 teams.

track_game is using soccer_small.mp4 video.

in order to run download all the repository and run track_game.py. 

------------------------------------------------------------------------------------------
At soccer ball classifier there are 3 different classifiers for detecting a soccer ball.

**yolo**

This classifier use the yolo and darkflow algorith in order to detect items in a image.
We train it in order to detect soccer balls at images and videos.
there are 2 files:

     1.track_image.py <image path>
     2.track_video.py <video path>
     
     the output of both algorithms is a new video/image with a result after their name with the output of the algorithm


