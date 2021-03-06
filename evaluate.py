# Test the model on test dataset

from keras.models import load_model
from load_data import load_dataset, load_videos_tracks, load_all_videos, get_dataset_diff_based_CNN
import h5py
import os
import sys
import numpy as np
from load_data import WeightedSum, get_temp_seq, get_spat_seq

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


num_image=32 # number of frames sampled from a 3 seconds clip
model_name = 'model/Final_submission/Ensemble_model_trainable_256B_relu_reg/Ensemble_model_trainable_256B_relu_reg_best.h5'
dataset_path = "gitignore/npy/Final_dataset_npy/"

# test_videos, test_tracks, test_lables = load_all_videos("dataset/clips/test/",
#                                                              "dataset/landmarks/test/", num_image)

# Load test data and labels
test_videos = np.load(dataset_path + "/test_videos.npy")
test_tracks = np.load(dataset_path + "/test_tracks.npy")
test_lables = np.load(dataset_path + "/test_lables.npy")
# test_coords, test_motions = get_dataset_diff_based_CNN(test_tracks, num_image)

(temp_seq_larm, temp_seq_rarm, temp_seq_trunk, temp_seq_lleg, temp_seq_rleg) = get_temp_seq(test_tracks, num_image)
spat_seq = get_spat_seq(test_tracks, num_image)
coords, motions = get_dataset_diff_based_CNN(test_tracks, num_image)
print(test_tracks.shape, test_lables.shape)
test_data = [coords, motions, temp_seq_larm, temp_seq_rarm, temp_seq_trunk, temp_seq_lleg, temp_seq_rleg, spat_seq]

# print(test_coords.shape, test_motions.shape)

model = load_model(model_name, custom_objects={"WeightedSum": WeightedSum})
model.summary()
# model = load_model(model_name)
print(model_name)

test_loss, test_acc = model.evaluate(test_data, test_lables, batch_size=1)
# test_loss, test_acc = model.evaluate([test_coords, test_motions], test_lables)
print('test_acc:', test_acc)
