This directory is for training the soccer ball detection model with FIFA game images.

The source video can be found in my google drive:

    https://drive.google.com/open?id=1arG0N1wlVeXdQNVQTlmBKuA3zPbNTRqA
  
screen_shots.py is for making screen shots of the video for training. I manually deleted the images which contains no soccer ball labeled the soccer balls with LabelImg. The images and annotations can be found in 'screen_shots_ready'

Then I trained the model with YOLO and darkflow by following this tutorial:

    https://github.com/deep-diver/Soccer-Ball-Detection-YOLOv2
    

To train the model yourself, you need to download and install darkflow on your computer:

    git clone https://github.com/thtrieu/darkflow.git
    cd darkflow
    python3 setup.py build_ext --inplace
    pip install -e .
 
Please also download the directory 'bin' from my google drive and put it in your project directory. It contains the pre-trained yolo_weights file.

Then you can start the training by running

    python3 train_yolo.py
    
The model I've trained can also be found in the directory 'ckpt' in my google drive.

