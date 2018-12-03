------------------------------------------------------------------------------------------

We also tried 3 other machine learning methods to detect the ball. The results were staisfying since the ball is too small and blurry. We didn't end up using any of the models, but we still put them in this repo for studying purpose.

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
