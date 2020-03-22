import keras
import matplotlib.pyplot as plt
#from keras.utils import plot_model
from keras import models, Input, Model, regularizers
from keras.applications import ResNet50
from keras.engine.saving import load_model
from keras.layers import Flatten, Dense, LSTM, TimeDistributed, MaxPool2D, Dropout, GRU, concatenate, Conv2D, MaxPooling2D
import os
import numpy as np
# Uncomment the line below to make GPU unavaliable
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import load_data

plt.switch_backend("agg")
def load_np_data(num_image):
    train_videos = np.load(str(num_image)+"image/train_videos.npy")
    train_tracks = np.load(str(num_image)+"image/train_tracks.npy")
    train_lables = np.load(str(num_image)+"image/train_lables.npy")

    valid_videos = np.load(str(num_image)+"image/valid_videos.npy")
    valid_tracks = np.load(str(num_image)+"image/valid_tracks.npy")
    valid_lables = np.load(str(num_image)+"image/valid_lables.npy")

    test_videos = np.load(str(num_image)+"image/test_videos.npy")
    test_tracks = np.load(str(num_image)+"image/test_tracks.npy")
    test_lables = np.load(str(num_image)+"image/test_lables.npy")
    return (train_videos, train_tracks, train_lables), (valid_videos, valid_tracks, valid_lables), (test_videos, test_tracks, test_lables)

# 0. intialization
num_image = 10
model_name = 'my_model'
path = "video_track_model/"

# 1. prepare for data

'''(train_videos, train_tracks, train_lables), (valid_videos, valid_tracks, valid_lables), (
    test_videos, test_tracks, test_lables) = load_data.load_dataset(
    video_dir="dataset/clips/", landmark_dir="dataset/landmarks/", num_image=num_image)'''

(train_videos, train_tracks, train_lables), (valid_videos, valid_tracks, valid_lables), (test_videos, test_tracks, test_lables) = load_np_data(num_image)

print('Train:', train_videos.shape, train_tracks.shape, train_lables.shape)
print('Valid:', valid_videos.shape, valid_tracks.shape, valid_lables.shape)
print('Test:', test_videos.shape, test_tracks.shape, test_lables.shape)
# 2. build network

# 2.1 CNN
# 2.1.1 pretrained CNN network
'''pretrained_CNN = TimeDistributed(ResNet50(weights="imagenet", include_top=False,
                                          input_tensor=Input(shape=(224, 224, 3))),
                                 input_shape=(num_image, 224, 224, 3))
pretrained_CNN.layer.trainable = False
video_input = Input(shape=(num_image, 224, 224, 3), name='video')
x = pretrained_CNN(video_input)
x = TimeDistributed(MaxPool2D(2, 2))(x)
x = TimeDistributed(Flatten())(x)
x = GRU(32, return_sequences=False)(x)
x = Dropout(0.5)(x)'''
# 2.1.2 customized CNN network
video_input = Input(shape=(num_image, 224, 224, 3), name='video')
x = TimeDistributed(Conv2D(16, (3, 3), activation='relu'))(video_input)
x = TimeDistributed(MaxPooling2D(2,2))(x)
x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(x)
x = TimeDistributed(MaxPooling2D(2,2))(x)
x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(x)
x = TimeDistributed(MaxPooling2D(2,2))(x)
x = TimeDistributed(Flatten())(x)
x = GRU(16, return_sequences=False)(x)
x = Dropout(0.5)(x)

# 2.2 dense network

landmark_input = Input(shape=(num_image, 25, 3), name='landmark')
y = TimeDistributed(Dense(128,kernel_regularizer=regularizers.l2(0.001), activation='relu'))(landmark_input)
y = TimeDistributed(Dense(128,kernel_regularizer=regularizers.l2(0.001), activation='relu'))(y)
y = Dropout(0.5)(y)
y = TimeDistributed(Dense(64,kernel_regularizer=regularizers.l2(0.001), activation='relu'))(y)
y = TimeDistributed(Dense(64,kernel_regularizer=regularizers.l2(0.001), activation='relu'))(y)
y = Dropout(0.5)(y)
y = TimeDistributed(Dense(32,kernel_regularizer=regularizers.l2(0.001), activation='relu'))(y)
y = TimeDistributed(Dense(32,kernel_regularizer=regularizers.l2(0.001), activation='relu'))(y)

y = TimeDistributed(Flatten())(y)
y = GRU(32, return_sequences=False)(y)
y = Dropout(0.5)(y)

# 2.3 concatenated
concatenated = concatenate([x, y])
output = Dense(1, activation='sigmoid')(concatenated)
model = Model([video_input, landmark_input], output)
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
#plot_model(model, show_shapes=True, to_file=path+model_name+'_model.png')
# 3. train network with saving the best model
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath=path+model_name+'_best.h5',
        monitor='val_acc',
        save_best_only=True,
    )
]

history = model.fit([train_videos, train_tracks],
                    train_lables,
                    epochs=50,
                    batch_size=16,
                    callbacks=callbacks_list,
                    verbose=2,
                    validation_data=([valid_videos, valid_tracks], valid_lables))
# 4. see validation loss and acc
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'r.-', label='Training loss')
plt.plot(epochs, val_loss_values, 'y*-', label='Validation loss')
plt.plot(epochs, acc, 'g.-', label='Training acc')
plt.plot(epochs, val_acc, 'b*-', label='Validation acc')
plt.title('Training and validation loss/accuracy')
plt.xlabel('Epochs')
plt.ylabel('loss/accuracy')
plt.legend()
plt.show()
plt.savefig(path+model_name+'_figure.png')

# 5. predict on test
# 5. predict on test
test_loss, test_acc = model.evaluate([test_videos, test_tracks], test_lables, verbose=2, batch_size=16)
print('Final test_acc:', test_acc)
model.save(path+model_name+'_final.h5')

best_model = load_model(path+model_name+'_best.h5')
test_loss, test_acc = best_model.evaluate([test_videos, test_tracks], test_lables, verbose=2, batch_size=16)
print('Best test_acc:', test_acc)
