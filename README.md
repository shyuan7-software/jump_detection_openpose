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
This directory stores the output from train_model.py when training.  
Under this directory, Dense128-32_GRU32_CNN16-32-32_GRU16_16batch_10image_50epoch_best.h5 file is the best trained model I can get.

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


## Improvement done to the next submission(submission-2):
1. Improve the quality of the dataset, by deleting data from HMDB (the video from HMDB always consists more than one person), best: 0.8405172413793104
2. To be update tomorrow...