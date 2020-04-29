# This model is an ensemble of cnn_model and rnn_model (check cnn_model.py and rnn_model.py)
import os
import keras
from keras.engine import Layer
import matplotlib.pyplot as plt
import numpy as np
from keras import Input, Model, regularizers
from keras.engine.saving import load_model
from keras.layers import Flatten, Dense, Dropout, concatenate, Conv2D, \
    MaxPooling2D, GRU
from keras.utils import plot_model
import math
import random

from load_data import data_aug, get_temp_seq, get_spat_seq, get_dataset_diff_based_CNN, WeightedSum
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

plt.switch_backend("agg")
dataset_path = "gitignore/npy/Final_dataset_npy/"
num_image = 32
chain_seq = [1,2,3,4,3,2,
        1,5,6,7,6,5,
        1,0,1,8,
        9,10,11,23,22,24,11,10,9,8,
        12,13,14,21,19,20,14,13,12,8,
        1]
def load_np_data():
    train_tracks = np.load(dataset_path + "/train_tracks.npy")
    train_lables = np.load(dataset_path + "/train_lables.npy")

    valid_tracks = np.load(dataset_path + "/valid_tracks.npy")
    valid_lables = np.load(dataset_path + "/valid_lables.npy")

    test_tracks = np.load(dataset_path + "/test_tracks.npy")
    test_lables = np.load(dataset_path + "/test_lables.npy")
    return (train_tracks, train_lables), (valid_tracks, valid_lables), (test_tracks, test_lables)

# 1. prepare for data
(train_tracks, train_labels), (valid_tracks, valid_labels), (test_tracks, test_labels) = load_np_data()
# APPEND: DATA AUG
train_videos, train_tracks, train_labels = data_aug(np.zeros(shape=(len(train_labels), num_image, 1, 1, 3)), train_tracks, train_labels)
valid_videos, valid_tracks, valid_labels = data_aug(np.zeros(shape=(len(valid_labels), num_image, 1, 1, 3)), valid_tracks, valid_labels)
test_videos, test_tracks, test_labels = data_aug(np.zeros(shape=(len(test_labels), num_image, 1, 1, 3)), test_tracks, test_labels)

print('Train:', train_tracks.shape, train_labels.shape)
print('Valid:', valid_tracks.shape, valid_labels.shape)
print('Test:', test_tracks.shape, test_labels.shape)


# 1.1 RNN data
def preprocess_data(tr_tracks, va_tracks, te_tracks, num):
    tr_ts = get_temp_seq(tr_tracks,num)
    va_ts = get_temp_seq(va_tracks,num)
    te_ts = get_temp_seq(te_tracks,num)

    tr_ss = get_spat_seq(tr_tracks,num)
    va_ss = get_spat_seq(va_tracks,num)
    te_ss = get_spat_seq(te_tracks,num)

    return (tr_ts, va_ts, te_ts), (tr_ss, va_ss, te_ss)


(train_temp_seqs, valid_temp_seqs, test_temp_seqs), \
(train_spat_seq, valid_spat_seq, test_spat_seq) \
    = preprocess_data(train_tracks, valid_tracks, test_tracks, num_image)
print('Train:', [temp_seq.shape for temp_seq in train_temp_seqs], train_spat_seq.shape, train_labels.shape)
print('Valid:', [temp_seq.shape for temp_seq in valid_temp_seqs], valid_spat_seq.shape, valid_labels.shape)
print('Test:', [temp_seq.shape for temp_seq in test_temp_seqs], test_spat_seq.shape, test_labels.shape)
(train_temp_larm, train_temp_rarm, train_temp_trunk, train_temp_lleg, train_temp_rleg) = train_temp_seqs
(valid_temp_larm, valid_temp_rarm, valid_temp_trunk, valid_temp_lleg, valid_temp_rleg) = valid_temp_seqs
(test_temp_larm, test_temp_rarm, test_temp_trunk, test_temp_lleg, test_temp_rleg) = test_temp_seqs

# 1.2 CNN data
train_coords, train_motions = get_dataset_diff_based_CNN(train_tracks, num_image)
print(train_coords.shape, train_motions.shape)

valid_coords, valid_motions = get_dataset_diff_based_CNN(valid_tracks, num_image)
print(valid_coords.shape, valid_motions.shape)

test_coords, test_motions = get_dataset_diff_based_CNN(test_tracks, num_image)
print(test_coords.shape, test_motions.shape)


train_data = [train_coords, train_motions, train_temp_larm, train_temp_rarm, train_temp_trunk, train_temp_lleg,
              train_temp_rleg, train_spat_seq]
valid_data = [valid_coords, valid_motions, valid_temp_larm, valid_temp_rarm, valid_temp_trunk, valid_temp_lleg,
              valid_temp_rleg, valid_spat_seq]
test_data = [test_coords, test_motions, test_temp_larm, test_temp_rarm, test_temp_trunk, test_temp_lleg, test_temp_rleg,
             test_spat_seq]

batch_size = 256
epochs = 100
path = 'model/'
model_name = 'Ensemble_model'
window_size = num_image // 4
GRU_path = 'model/Final_submission/AUGM_Continued_GRU/AUGM_Continued_best.h5'
CNN_path = 'model/Final_submission/AUGM_CNN_0.5D_256B/AUGM_CNN_0.5D_256B_best.h5'
GRU_model = load_model(GRU_path, custom_objects={"WeightedSum": WeightedSum})
CNN_model = load_model(CNN_path)
GRU_model.trainable = True
CNN_model.trainable = True

coord_input = Input(shape=(32, 25, 2), name='coord_input')
motion_input = Input(shape=(31, 25, 2), name='motion_input')
temporal_larm_input = Input(shape=(num_image, 2 * 3), name='temporal_larm_seq')
temporal_rarm_input = Input(shape=(num_image, 2 * 3), name='temporal_rarm_seq')
temporal_trunk_input = Input(shape=(num_image, 2 * 3), name='temporal_trunk_seq')
temporal_lleg_input = Input(shape=(num_image, 2 * 6), name='temporal_lleg_seq')
temporal_rleg_input = Input(shape=(num_image, 2 * 6), name='temporal_rleg_seq')
spatial_input = Input(shape=(len(chain_seq), 2 * window_size), name='spatial_seq')

x = CNN_model([coord_input, motion_input])
y = GRU_model([temporal_larm_input, temporal_rarm_input, temporal_trunk_input, temporal_lleg_input, temporal_rleg_input, spatial_input])

output = concatenate([x, y])
output = Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu')(output)
output = Dropout(0.5)(output)
output = Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu')(output)
output = Dropout(0.5)(output)
output = Dense(1, activation='sigmoid')(output)
model = Model([coord_input,
               motion_input,
               temporal_larm_input,
               temporal_rarm_input,
               temporal_trunk_input,
               temporal_lleg_input,
               temporal_rleg_input,
               spatial_input], output)
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
os.mkdir(path + model_name)
path += model_name + '/'
# plot_model(model, show_shapes=True, to_file=path + model_name + '_model.png')
# 3. train network with saving the best model
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath=path + model_name + '_best.h5',
        monitor='val_loss',
        save_best_only=True,
    )
]

history = model.fit(train_data,
          train_labels,
          epochs=epochs,
          batch_size=batch_size,
          callbacks=callbacks_list,
          validation_data=(valid_data, valid_labels))
# 4. see validation loss and acc
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epochs = range(1, len(loss_values) + 1)
plt.clf()
plt.plot(epochs, loss_values, 'r.-', label='Training loss')
plt.plot(epochs, val_loss_values, 'y*-', label='Validation loss')
plt.plot(epochs, acc, 'g.-', label='Training acc')
plt.plot(epochs, val_acc, 'b*-', label='Validation acc')
plt.title('Training and validation loss/accuracy')
plt.xlabel('Epochs')
plt.ylabel('loss/accuracy')
plt.legend()
plt.show()
plt.savefig(path + model_name + '_figure.png')

# 5. predict on test
f = open(path + "output.txt", 'a')
test_loss, test_acc = model.evaluate(test_data, test_labels, batch_size=batch_size)
print('Final test_acc:', test_acc)
print('Final test_loss:', test_loss)

f.write('Final test_acc:' + str(test_acc))
f.write('\n')
f.write('Final test_loss:' + str(test_loss))
f.write('\n')
model.save(path + model_name + '_final.h5')

best_model = load_model(path + model_name + '_best.h5',custom_objects={'WeightedSum':WeightedSum})
test_loss, test_acc = best_model.evaluate(test_data, test_labels, batch_size=batch_size)
print('Best test_acc:', test_acc)
print('Best test_loss:', test_loss)

f.write('Best test_acc:' + str(test_acc))
f.write('\n')
f.write('Best test_loss:' + str(test_loss))
f.write('\n')
f.close()