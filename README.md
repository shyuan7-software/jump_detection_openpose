# Jump detection
## Code explanation:
### 1. load_data.py:
This python file is for loading dataset, by loading clips,
 decoding JSON files to load dataset from "./dataset" directory
### 2. train_model.py:
This python file is to train the neural network model, for fast loading dataset,
it uses numpy.load() function to load dataset from "./10image" directory
### 3. evaluate.py:
This python file is to load trained model and test dataset, and then evaluate the performance of the model again.
### 4. generate_figure.py:
This python file is the main function of this project, this file has two input:
a video, and a directory that stores landmark files. This file will output a json file and a figure, as follows:  
![alt text](https://github.com/shyuan7-software/images/blob/master/generate_figure_result.png)

## Directory:
### 1. ./model/:
This directory stores the trained models.    
Under this directory, model/submission2/windowsize_COMP_TWO_STREAM_GRU_3layer512_25_20.5dropout_64batch_32image_100epoch_noHMDB_best.h5 file is the best trained model I can get.

### 2. ./dataset/ (not in github):
This directory stores all the dataset in forms of clips and landmark json files  
The dataset is too large to be uploaded to github, please access it through:  
[dataset](https://drive.google.com/file/d/1shPnXQeDR2yWOFankFqCrR7JyakFenSl/view?usp=sharing) 

### 3. ./32image/ (not in github):
This directory stores all the dataset in forms of .NPY files  
The dataset is too large to be uploaded to github, please access it through:  
[32image](https://drive.google.com/drive/folders/1b11D5WAf7ELt4FV2HNGCvS3ZCKkNhYJS?usp=sharing)

### 4. ./sample/ (not in github):
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