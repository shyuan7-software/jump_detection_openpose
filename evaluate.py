from keras.models import load_model

from load_data import load_dataset, load_videos_tracks, load_all_videos
import h5py
import os
import sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_image=10
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model_name = 'video_track_model/Dense128-32_GRU32_CNN16-32-32_GRU16_16batch_10image_50epoch_best.h5'
test_videos, test_tracks, test_lables = load_all_videos("dataset/clips/test/",
                                                             "dataset/landmarks/test/", 10)
'''test_videos = np.load(str(num_image) + "image/test_videos.npy")
test_tracks = np.load(str(num_image) + "image/test_tracks.npy")
test_lables = np.load(str(num_image) + "image/test_lables.npy")'''
print(test_videos.shape, test_tracks.shape, test_lables.shape)
model = load_model(model_name)
print(model_name)

test_loss, test_acc = model.evaluate([test_videos, test_tracks], test_lables, batch_size=1)
print('test_acc:', test_acc)
