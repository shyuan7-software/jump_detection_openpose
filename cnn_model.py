# The model is based on https://arxiv.org/pdf/1704.07595.pdf
import os
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import Input, Model
from keras.engine.saving import load_model
from keras.layers import Flatten, Dense, Dropout, concatenate, Conv2D, \
    MaxPooling2D
from keras.utils import plot_model
from load_data import get_dataset_diff_based_CNN
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

plt.switch_backend("agg")
path = "model/"
dataset_path = "gitignore/npy/32image_noHMDB_noEmptyFrame/"
num_image = 32


# Load dataset through loading .npy files, much faster than loading original video and body landmark files
def load_np_data():
    train_tracks = np.load(dataset_path + "/train_tracks.npy")
    train_lables = np.load(dataset_path + "/train_lables.npy")

    valid_tracks = np.load(dataset_path + "/valid_tracks.npy")
    valid_lables = np.load(dataset_path + "/valid_lables.npy")

    test_tracks = np.load(dataset_path + "/test_tracks.npy")
    test_lables = np.load(dataset_path + "/test_lables.npy")
    return (train_tracks, train_lables), (valid_tracks, valid_lables), (test_tracks, test_lables)


# 0. prepare for data

(train_tracks, train_labels), (valid_tracks, valid_labels), (test_tracks, test_labels) = load_np_data()
print('Train:', train_tracks.shape, train_labels.shape)
print('Valid:', valid_tracks.shape, valid_labels.shape)
print('Test:', test_tracks.shape, test_labels.shape)

# Transfering original body landmark dataset, to the dataset that can be used in the model of
# https://arxiv.org/pdf/1704.07595.pdf
train_coords, train_motions = get_dataset_diff_based_CNN(train_tracks, num_image)
print(train_coords.shape, train_motions.shape)

valid_coords, valid_motions = get_dataset_diff_based_CNN(valid_tracks, num_image)
print(valid_coords.shape, valid_motions.shape)

test_coords, test_motions = get_dataset_diff_based_CNN(test_tracks, num_image)
print(test_coords.shape, test_motions.shape)

# 1. intialization
batch_size = 64
epochs = 500
dropout = 0.5
model_name = 'CNN_model'

# 2. build network
# 2.1 coord_input

coord_input = Input(shape=(32, 25, 2), name='coord_input')
x = Dense(128)(coord_input)
x = Dropout(dropout)(x)
x = Conv2D(32, (2, 2), activation='relu', strides=1)(x)
x = MaxPooling2D((3, 3), strides=2)(x)
x = Dropout(dropout)(x)

x = Conv2D(32, (2, 2), activation='relu', strides=1)(x)
x = MaxPooling2D((3, 3), strides=2)(x)
x = Dropout(dropout)(x)

x = Conv2D(64, (2, 2), activation='relu', strides=1)(x)
x = MaxPooling2D((3, 3), strides=2)(x)
x = Dropout(dropout)(x)

# 2.2 motion_input
motion_input = Input(shape=(31, 25, 2), name='motion_input')
y = Dense(128)(motion_input)
y = Dropout(dropout)(y)

y = Conv2D(32, (2, 2), activation='relu', strides=1)(y)
y = MaxPooling2D((3, 3), strides=2)(y)
y = Dropout(dropout)(y)

y = Conv2D(32, (2, 2), activation='relu', strides=1)(y)
y = MaxPooling2D((3, 3), strides=2)(y)
y = Dropout(dropout)(y)

y = Conv2D(64, (2, 2), activation='relu', strides=1)(y)
y = MaxPooling2D((3, 3), strides=2)(y)
y = Dropout(dropout)(y)

output = concatenate([x, y])
output = Flatten()(output)
output = Dense(128, activation='relu')(output)
output = Dropout(dropout)(output)
# output = Dense(64, activation='relu')(output)
# output = Dropout(dropout)(output)
# output = Dense(32, activation='relu')(output)
# output = Dropout(dropout)(output)
output = Dense(1, activation='sigmoid')(output)

model = Model([coord_input, motion_input], output)
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

os.mkdir(path + model_name)
path += model_name + '/'
print(path)
plot_model(model, show_shapes=True, to_file=path + model_name + '_model.png')
# 3. train network with saving the best model
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath=path + model_name + '_best.h5',
        monitor='val_loss',
        save_best_only=True,
    )
]

history = model.fit([train_coords, train_motions],
                    train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks_list,
                    validation_data=([valid_coords, valid_motions], valid_labels))
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
test_loss, test_acc = model.evaluate([test_coords, test_motions], test_labels, batch_size=batch_size)
print('Final test_acc:', test_acc)
print('Final test_loss:', test_loss)

f.write('Final test_acc:' + str(test_acc))
f.write('\n')
f.write('Final test_loss:' + str(test_loss))
f.write('\n')
model.save(path + model_name + '_final.h5')

best_model = load_model(path + model_name + '_best.h5')
test_loss, test_acc = best_model.evaluate([test_coords, test_motions], test_labels, batch_size=batch_size)
print('Best test_acc:', test_acc)
print('Best test_loss:', test_loss)

# Save the test accuracy and loss to a text file
f.write('Best test_acc:' + str(test_acc))
f.write('\n')
f.write('Best test_loss:' + str(test_loss))
f.write('\n')
f.close()
