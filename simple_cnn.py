# This model tries to classify the images of jumping and standing
import os
import keras
from keras.engine import Layer
import matplotlib.pyplot as plt
import numpy as np
from keras import Input, Model, models, layers
from keras.engine.saving import load_model
from keras.layers import Flatten, Dense, Dropout, concatenate, Conv2D, \
    MaxPooling2D, GRU
from keras.utils import plot_model
import math
import random

from load_data import load_img_dataset

train_imgs, train_labels, valid_imgs, valid_labels, test_imgs, test_labels = load_img_dataset('gitignore/img_dataset/')
print(train_imgs.shape, train_labels.shape)
print(valid_imgs.shape, valid_labels.shape)
print(test_imgs.shape, test_labels.shape)

model_name = 'simple_cnn_small'
path = "gitignore/simple_cnn/"


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath=path + model_name + '_best.h5',
        monitor='val_loss',
        save_best_only=True,
    )
]

history = model.fit(
    train_imgs,
    train_labels,
    epochs=50,
    batch_size=8,
    callbacks=callbacks_list,
    verbose=2,
    validation_data=(
        valid_imgs,
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
test_loss, test_acc = model.evaluate(
    test_imgs, test_labels, verbose=2, batch_size=8)
print('Final test_acc:', test_acc)
print('Final test_loss:', test_loss)
model.save(path + model_name + '_final.h5')

best_model = load_model(path + model_name + '_best.h5')
test_loss, test_acc = best_model.evaluate(
    test_imgs, test_labels, verbose=2, batch_size=8)
print('Best test_acc:', test_acc)
print('Best test_loss:', test_loss)
