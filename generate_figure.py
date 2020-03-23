import json
import os
import sys
from math import ceil

from load_data import decode_json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import shutil
import cv2
import sys

# X = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87]
# Y = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87]

# video_path = 'mini_dataset/clips/test/jump/jump4.mp4'
# landmark_path = 'mini_dataset/landmarks/test/jump/jump4.mp4'
# model_path = 'video_track_best.h5'

plt.switch_backend("agg")


def generate_clips_tracks(video_path, landmark_path, num_image):
    landmark_path += '/'
    video = cv2.VideoCapture(video_path)
    image_per_frame = []
    while video.isOpened():
        success, image = video.read()
        if not success:
            break
        if image.all() != None:
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image_per_frame.append(image)
    files = os.listdir(landmark_path)
    landmark_per_frame = [landmark_path + file for file in files]
    frame_rate = round(video.get(5))
    clips_per3s = []
    tracks_per3s = []
    clips_num = ceil(len(image_per_frame) / (frame_rate * 3))
    empty_img = np.zeros((224, 224, 3), np.uint8)
    empty_landmark = [[0, 0, 0]] * 25
    for clip_id in range(clips_num):
        cur_clip = []
        cur_track = []
        for i in range(clip_id * (frame_rate * 3), min(len(image_per_frame), (clip_id + 1) * (frame_rate * 3)),
                       (frame_rate * 3) // num_image):
            cur_clip.append(image_per_frame[i])
            cur_track.append(decode_json(landmark_per_frame[i]))
            if len(cur_clip) == num_image: break
        while len(cur_clip) < num_image:
            cur_clip.append(empty_img)
            cur_track.append(empty_landmark)
        clips_per3s.append(cur_clip)
        tracks_per3s.append(cur_track)

    return np.array(clips_per3s), np.array(tracks_per3s)


def generate_figure(video_path, landmark_path, model_path, num_image):
    clips, tracks = generate_clips_tracks(video_path, landmark_path, num_image)
    if 'output' not in os.listdir('./'):
        os.mkdir('output')
    print(clips.shape, tracks.shape)
    videoname = video_path.split("/")[-1]
    model = load_model(model_path)
    predictions = model.predict([clips, tracks], batch_size=1)
    Y = []
    for (i, p) in enumerate(predictions):
        Y.append(p[0])
    X = [3 * x for x in range(0, len(clips))]
    fig = plt.figure()
    plt.bar(X, Y, 3, align='edge', ec='c', ls='-.', lw=1, color='#EECFA1', tick_label=X)
    plt.tick_params(labelsize=6)

    for (x, y) in zip(X, Y):
        plt.text(x, y + 0.01, str(round(y, 2)) + '%', fontsize=6)
    plt.xlabel("Time(s)")
    plt.ylabel("Possibility(%)")
    plt.title("Jump detection")
    plt.savefig('./output/' + videoname + '.png', dpi=300)
    plt.show()
    # json
    temp_dict = {'jump': []}
    for i in range(len(clips)):
        temp_dict['jump'].append({str(X[i]) + 's to ' + str(X[i] + 3) + 's': str(Y[i])})
    json_file = './output/' + videoname + '.json'
    with open(json_file, 'w') as f:
        json.dump(temp_dict, f)
    print('The result is located in ./output/' + videoname + '.json and ./output/' + videoname + '.png')


# video_path: where is your video?
# landmark_path: where is the video's corresponding landmark directory?
if __name__ == '__main__':
    v_path = sys.argv[1]  # 'sample/no_jump.mp4'
    l_path = sys.argv[2]  # 'sample/no_jump'
    generate_figure(video_path=v_path, landmark_path=l_path,
                    model_path='video_track_model/Dense128-32_GRU32_CNN16-32-32_GRU16_16batch_10image_50epoch_best.h5',
                    num_image=10)