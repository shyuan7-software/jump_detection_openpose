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
[image](https://github.com/shyuan7-software/images/blob/master/generate_figure_result.png)  

## Directory:
### 1. ./video_track_model/:
This directory stores the output from train_model.py when training.  
Under this directory, _best.h5 file is the best model I can train.

### 2. ./dataset/ (not in github):
This directory stores all the dataset in forms of clips and landmark json files  
The dataset is too large to be uploaded to github, please access it through:  
[dataset](https://drive.google.com/drive/folders/1vUYK2-X1HWBWLH3C1e4IYcaMAN_CzRjg?usp=sharing) 

### 3. ./10image/ (not in github):
This directory stores all the dataset in forms of .NPY files  
The dataset is too large to be uploaded to github, please access it through:  
[10image](https://drive.google.com/drive/folders/1V6PB5sE8K8jLW1UnnoBPDZVhQO-tLXAh)

### 4. ./sample/ (not in github):
This is for test purpose, under this directory, there are 6 videos (I shoot myself jumping) and their landmarks, you can easily
test my code with this directory.  
[sample](https://drive.google.com/drive/folders/1St1RiO6kB9MlPiOF6ItiIuPoCYV1uahQ?usp=sharing)