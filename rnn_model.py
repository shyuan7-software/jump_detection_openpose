# The model is based on https://arxiv.org/abs/1704.02581

import os
import keras
import matplotlib.pyplot as plt
import numpy as np
# from keras.utils import plot_model
from keras import Input, Model
from keras.engine.saving import load_model
from keras.layers import Dense, GRU, concatenate, Conv2D, BatchNormalization, MaxPooling2D

from load_data import WeightedSum, get_spat_seq, get_temp_seq

# Uncomment the line below to make GPU unavaliable
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define the custom layer
# tf.disable_v2_behavior()

# The chain_seq is the order of traversing skeleton joints
chain_seq = [1, 2, 3, 4, 3, 2,
             1, 5, 6, 7, 6, 5,
             1, 0, 1, 8,
             9, 10, 11, 23, 22, 24, 11, 10, 9, 8,
             12, 13, 14, 21, 19, 20, 14, 13, 12, 8,
             1]

plt.switch_backend("agg")


# Load dataset through loading .npy files, much faster than loading original video and body landmark files
def load_np_data(dataset_path):
    train_videos = np.load(dataset_path + "/train_videos.npy")
    train_tracks = np.load(dataset_path + "/train_tracks.npy")
    train_lables = np.load(dataset_path + "/train_lables.npy")

    valid_videos = np.load(dataset_path + "valid_videos.npy")
    valid_tracks = np.load(dataset_path + "/valid_tracks.npy")
    valid_lables = np.load(dataset_path + "/valid_lables.npy")

    test_videos = np.load(dataset_path + "/test_videos.npy")
    test_tracks = np.load(dataset_path + "/test_tracks.npy")
    test_lables = np.load(dataset_path + "/test_lables.npy")
    return (train_videos, train_tracks, train_lables), (valid_videos, valid_tracks, valid_lables), (
        test_videos, test_tracks, test_lables)


# 0. intialization
num_image = 32  # number of frames, sampled from each clip
model_name = 'rnn_model'
path = "model/"  # the dir where you want to save your output and trained-model
dataset_path = "gitignore/npy/Final_dataset_npy/"
batch_size = 64
epochs = 1
dropout = 0.5
RNN_size = 512
window_size = num_image // 4  # the size of window in spatial input, as described in https://arxiv.org/abs/1704.02581
# 1. prepare for data

'''(train_videos, train_tracks, train_lables), (valid_videos, valid_tracks, valid_lables), (
    test_videos, test_tracks, test_lables) = load_data.load_dataset(
    video_dir="dataset/clips/", landmark_dir="dataset/landmarks/", num_image=num_image)'''

(train_videos, train_tracks, train_labels), (valid_videos, valid_tracks, valid_labels), (
    test_videos, test_tracks, test_labels) = load_np_data(dataset_path)


# Transfering landmarks data to temporal input and spatial input
def preprocess_data(train_tracks, valid_tracks, test_tracks, num_image):
    train_temp_seqs = get_temp_seq(train_tracks, num_image)
    valid_temp_seqs = get_temp_seq(valid_tracks, num_image)
    test_temp_seqs = get_temp_seq(test_tracks, num_image)

    train_spat_seq = get_spat_seq(train_tracks, num_image)
    valid_spat_seq = get_spat_seq(valid_tracks, num_image)
    test_spat_seq = get_spat_seq(test_tracks, num_image)

    return (train_temp_seqs, valid_temp_seqs, test_temp_seqs), (train_spat_seq, valid_spat_seq, test_spat_seq)


(train_temp_seqs, valid_temp_seqs, test_temp_seqs), \
(train_spat_seq, valid_spat_seq, test_spat_seq) \
    = preprocess_data(train_tracks, valid_tracks, test_tracks, num_image)

print('Train:', [temp_seq.shape for temp_seq in train_temp_seqs], train_spat_seq.shape, train_labels.shape)
print('Valid:', [temp_seq.shape for temp_seq in valid_temp_seqs], valid_spat_seq.shape, valid_labels.shape)
print('Test:', [temp_seq.shape for temp_seq in test_temp_seqs], test_spat_seq.shape, test_labels.shape)
(train_temp_larm, train_temp_rarm, train_temp_trunk, train_temp_lleg, train_temp_rleg) = train_temp_seqs
(valid_temp_larm, valid_temp_rarm, valid_temp_trunk, valid_temp_lleg, valid_temp_rleg) = valid_temp_seqs
(test_temp_larm, test_temp_rarm, test_temp_trunk, test_temp_lleg, test_temp_rleg) = test_temp_seqs

# 2. build network
# 2.1 temporal input
# 2.1.1
temporal_larm_input = Input(shape=(num_image, 2 * 3), name='temporal_larm_seq')
x1 = GRU(RNN_size // 8, implementation=1, recurrent_activation='hard_sigmoid', return_sequences=True, dropout=0.2,
         recurrent_dropout=dropout)(temporal_larm_input)

# 2.1.2
temporal_rarm_input = Input(shape=(num_image, 2 * 3), name='temporal_rarm_seq')
x2 = GRU(RNN_size // 8, implementation=1, recurrent_activation='hard_sigmoid', return_sequences=True, dropout=0.2,
         recurrent_dropout=dropout)(temporal_rarm_input)

# 2.1.3
temporal_trunk_input = Input(shape=(num_image, 2 * 3), name='temporal_trunk_seq')
x3 = GRU(RNN_size // 8, implementation=1, recurrent_activation='hard_sigmoid', return_sequences=True, dropout=0.2,
         recurrent_dropout=dropout)(temporal_trunk_input)

# 2.1.4
temporal_lleg_input = Input(shape=(num_image, 2 * 6), name='temporal_lleg_seq')
x4 = GRU(RNN_size // 4, implementation=1, recurrent_activation='hard_sigmoid', return_sequences=True, dropout=0.2,
         recurrent_dropout=dropout)(temporal_lleg_input)

# 2.1.5
temporal_rleg_input = Input(shape=(num_image, 2 * 6), name='temporal_rleg_seq')
x5 = GRU(RNN_size // 4, implementation=1, recurrent_activation='hard_sigmoid', return_sequences=True, dropout=0.2,
         recurrent_dropout=dropout)(temporal_rleg_input)

x = concatenate([x1, x2, x3, x4, x5], 2)
x = GRU(RNN_size, implementation=1, recurrent_activation='hard_sigmoid', return_sequences=True, dropout=0.2,
        recurrent_dropout=dropout)(x)
x = GRU(RNN_size, implementation=1, recurrent_activation='hard_sigmoid', return_sequences=False, dropout=0.2,
        recurrent_dropout=dropout)(x)
x = Dense(1, activation='sigmoid')(x)
# 2.2 spatial_input
spatial_input = Input(shape=(len(chain_seq), 2 * window_size), name='spatial_seq')
y = GRU(RNN_size, implementation=1, recurrent_activation='hard_sigmoid', return_sequences=True, dropout=0.2,
        recurrent_dropout=dropout)(spatial_input)
y = GRU(RNN_size, implementation=1, recurrent_activation='hard_sigmoid', return_sequences=True, dropout=0.2,
        recurrent_dropout=dropout)(y)
y = GRU(RNN_size, implementation=1, recurrent_activation='hard_sigmoid', return_sequences=False, dropout=0.2,
        recurrent_dropout=dropout)(y)
y = Dense(1, activation='sigmoid')(y)
# 2.3 output
output = WeightedSum(0.9)([x, y])

model = Model([temporal_larm_input,
               temporal_rarm_input,
               temporal_trunk_input,
               temporal_lleg_input,
               temporal_rleg_input, spatial_input], output)
model.summary()
config = model.get_config()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
os.mkdir(path + model_name)
path += model_name + '/'
# plot_model(model, show_shapes=True, to_file=path+model_name+'_model.png')
# 3. train network with saving the best model
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath=path + model_name + '_best.h5',
        monitor='val_loss',
        save_best_only=True,
    )
]

history = model.fit(
    [train_temp_larm, train_temp_rarm, train_temp_trunk, train_temp_lleg, train_temp_rleg, train_spat_seq],
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks_list,
    # verbose=2,
    validation_data=(
        [valid_temp_larm, valid_temp_rarm, valid_temp_trunk, valid_temp_lleg, valid_temp_rleg, valid_spat_seq],
        valid_labels))
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

test_loss, test_acc = model.evaluate(
    [test_temp_larm, test_temp_rarm, test_temp_trunk, test_temp_lleg, test_temp_rleg, test_spat_seq], test_labels,
    verbose=2, batch_size=16)
print('Final test_acc:', test_acc)
print('Final test_loss:', test_loss)
model.save(path + model_name + '_final.h5')
f.write('Final test_acc:' + str(test_acc))
f.write('\n')
f.write('Final test_loss:' + str(test_loss))
f.write('\n')
model.save(path + model_name + '_final.h5')

best_model = load_model(path + model_name + '_best.h5', custom_objects={'WeightedSum': WeightedSum})
test_loss, test_acc = best_model.evaluate(
    [test_temp_larm, test_temp_rarm, test_temp_trunk, test_temp_lleg, test_temp_rleg, test_spat_seq], test_labels,
    verbose=2, batch_size=16)
print('Best test_acc:', test_acc)
print('Best test_loss:', test_loss)
f.write('Best test_acc:' + str(test_acc))
f.write('\n')
f.write('Best test_loss:' + str(test_loss))
f.write('\n')
f.close()
