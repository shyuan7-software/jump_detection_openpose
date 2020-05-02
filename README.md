# Jump detection
## How to run:
### Dependency:
pip install keras  
pip install tensorflow  
pip install numpy  
pip install opencv-python  
pip install matplotlib  
pip install googledrivedownloader  

### Run:
python generate_figure.py  [video_path]  [body-landmark_directory]  

eg.  
python generate_figure.py  test_video1.mp4  test_video1  
test_video1.mp4 is a MP4 video file  
test_video1 is a directory containing all body-landmark json-files of test_video1.mp4, generated by OpenPose

### Check results:
After running generate_figure.py, check your results at the 'OUTPUT' directory

## Code explanation:
### 1. load_data.py:
This python file is for loading dataset, by loading clips,
 decoding JSON files to load dataset from "./dataset" directory
### 2. cnn_model.py, rnn_model.py, ensemble.py:
The python files are to train the neural network model, for fast loading dataset,
it uses numpy.load() function to load dataset.  
### 3. download_GGdrive.py:  
This python files will download a video and its landmark files from Google drive, and call generate_figure.py to run model on the downloaded video and landmark files.
### 4. evaluate.py:
This python file is to load trained model and test dataset, and then evaluate the performance of the model again.
### 5. generate_figure.py:
This python file is the main function of this project, this file has two input:
a video, and a directory that stores landmark files. This file will output a json file and a figure, as follows:  
![alt text](https://github.com/shyuan7-software/images/blob/master/generate_figure_result.png)

## Directory:
### 1. ./model/:
This directory stores the trained models.    
Under this directory, "model/Final_submission/Ensemble_model_trainable_256B_relu_reg/Ensemble_model_trainable_256B_relu_reg_best.h5" file is the best trained model I can get.


### 2. ./sample/ (not in github):
This is for test purpose, under this directory, there are 13 videos (I shoot myself jumping) and their landmark files, you can easily
test my code with this directory.  
[sample](https://drive.google.com/file/d/1TO9qZnFNA0U0Kj7CNqGJTSWKzFSaqOC5/view?usp=sharing)


## Improvement of submission-2:
1. Improved the quality of the dataset, by deleting videos from HMDB51 (the videos from HMDB51 always consists more than one person)
2. Disregarded the original video input, and only focus on the body-landmarks input
3. Improved the sample rate from 10 frames per clip to 32 frames per clip
4. Updated the model, with the [paper](https://arxiv.org/abs/1704.02581)
5. Added more sample videos for test purpose

## Improving direction on the third submission:
Main tasks:   
&#8195;Design a new (CNN based) model that takes the original video as input  
&#8195;Train the new model, so that it can keep 85% or higher accuracy  
  
Bonus points:   
&#8195;Add videos of old people jumping/not jumping to dataset;   
&#8195;Add more sample videos;   
&#8195;Connect the new (CNN based) model with the current model (which takes body-landmarks as input), try to keep the accuracy higher than 85%;

## Improvement of submission-3:
1. Designed a CNN based model, which takes orignial RGB video frames as input, based on the [paper](https://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf)
2. Designed a CNN based model, which takes the skeleton data as input, based on the [paper](https://arxiv.org/pdf/1704.07595.pdf)
3. Trained the model which takes skeleton data as input,and it has higher than 95% accuarcy

## Improving direction on the fourth submission:
Main tasks:   
&#8195;Fuse the CNN based model and the RNN based model, to achieve a better performance in real life application  
&#8195;Adjust the hyperparameter for the CNN based model and the RNN based model, so they will have an optimal performance  
&#8195;Do experiment with CNN which takes RGB video frame as input, to know whether CNN can extract the feature of the distance between feets and ground. If yes, include it to the model.
  
Bonus points:   
&#8195;Add videos of old people jumping/not jumping to the dataset;   
&#8195;Add more sample videos;   

## Improvement of submission-final:
1. Expanded the dataset, by filming myself jumping
2. Implemented data augmentation, by rotating videos 90, 180, 270 degrees
3. Implemented 3D ResNet model  
4. Tried to use 2D CNN model to classify images of jumping and standing
5. Tried to include temporal CNN layer into the model
6. Used ensemble learning technique to combine the previsou two models, and got a best model which has 97% accuracy