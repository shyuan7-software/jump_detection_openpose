# Author: Shaohua Yuan
# Email: shyuan@tamu.edu
# In order to load dataset, please format the file directory as following:
# dataset_dir --> train --> jump   --> (lots of video, the name of video should contain "jump")
#                          others --> (lots of video, the name of video should not contain "jump")
#            --> valid --> jump   --> (lots of video, the name of video should contain "jump")
#                          others --> (lots of video, the name of video should not contain "jump")
#            --> test  --> jump   --> (lots of video, the name of video should contain "jump")
#                          others --> (lots of video, the name of video should not contain "jump")
# Then, just run load_all_dataset(dataset_dir), it will return train, valid, and test dataset in order
import json as js
import os
import random

import cv2
import numpy as np
from keras.engine import Layer


# video_path- "dataset/landmarks/test/jump/jump1.mp4/jump1_000000000000_keypoints.json"
# Function: decode a landmark json file
def decode_json(path):
    with open(path, 'r') as jsonfile:
        json_dict = js.load(jsonfile)
    if len(json_dict['people']) == 0:
        landmark = [0, 0, 0] * 25
    else:
        person = json_dict['people'][0]
        landmark = person['pose_keypoints_2d']
    res = []
    for i in range(len(landmark) // 3):
        res.append(landmark[i * 3:i * 3 + 3])
    return res


# test decode_json()
'''
landmarks = decode_json("./dataset/landmarks/test/jump/jump1.mp4/jump1_000000000000_keypoints.json")
print(len(landmarks), type(landmarks))
'''


# video_path- "dataset/landmarks/test/jump/jump1.mp4"
# num_image- 30 json files will be decoded and returned
# Function: load a number of landmark json files
def video_to_landmarks(landmark_path, num_image):
    landmark_path += '/'
    json_names = sorted(os.listdir(landmark_path))
    landmarks = []
    frame_num = len(json_names)
    if frame_num == 0:
        return [], False
    #sample_rate = math.ceil(frame_num / num_image)
    sample_rate = frame_num / num_image
    if sample_rate == 0:
        sample_rate += 1
    for frame_id in range(frame_num):
        if len(landmarks) == num_image:
            break
        #if frame_id % sample_rate == 0:
        if int(sample_rate * len(landmarks)) <= frame_id:
            # print(frame_id)
            landmarks.append(decode_json(landmark_path + json_names[frame_id]))
    empty_landmark = [[0, 0, 0]] * 25
    while len(landmarks) < num_image:
        landmarks.append(empty_landmark)
    return np.array(landmarks), True


# test video_to_landmarks
'''
landmarks, res = video_to_landmarks("dataset/landmarks/test/jump/jump1.mp4", 30)
print(landmarks.shape)
for i in landmarks:
    print(len(i), i)

'''


# video_path- "dataset/landmarks/test/jump/jump1.mp4"
# num_image- 30 frames will be returned
# Function: Convert a video to lots of images

def video_to_imgaes(video_path, num_image):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(video_path)
        return [], False
    frame_num = video.get(7)
    # sample_rate = math.ceil(frame_num / num_image)
    sample_rate = frame_num / num_image
    if sample_rate == 0:
        sample_rate += 1
    images = []
    while video.isOpened():
        frameId = video.get(1)
        success, image = video.read()
        if not success or len(images) == num_image:
            break
        if image.all() != None:
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        # if frameId % sample_rate == 0:
        if int(sample_rate * len(images)) <= frameId:
            # print(frameId)
            images.append(image)
            # cv2.namedWindow("Image")
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)

    empty_img = np.zeros((224, 224, 3), np.uint8)
    while len(images) < num_image:
        while len(images) < num_image:
            images.append(empty_img)
    return np.array(images), True


# video_dir - "dataset/clips/train/others/"
# landmark_dir - "dataset/landmarks/train/others/"
# Function: Load all videos under "videos_dir" into memory as images, and at the same time label them according to the file name
def load_videos_tracks(videos_dir, landmark_dir, num_image):
    dirs = os.listdir(videos_dir)
    videos = []
    body_tracks = []
    labels = []
    for file in dirs:
        print(videos_dir + file)
        video, res1 = video_to_imgaes(videos_dir + file, num_image)
        body_track, res2 = video_to_landmarks(landmark_dir + file, num_image)
        if not res1 or not res2:
            continue
        videos.append(video)
        body_tracks.append(body_track)
        if "jump" in file:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(videos), np.array(body_tracks), np.array(labels)


# test load_videos():
'''
train_datas, train_tracks, train_labels = load_videos_tracks("dataset/clips/train/others/",
                                                             "dataset/landmarks/train/others/", 10)
print(train_datas.shape)
print(train_tracks.shape)
print(train_labels.shape)

'''


# Function: Load a kind of dataset(video, track, label), train, test, or valid
# video_dir - "dataset/clips/train/"
# landmark_dir - "dataset/landmarks/train/"
#               under video_dir and landmark_dir, there must be jump and others dirs
def load_all_videos(video_dir, landmark_dir, num_image):
    videoset_dir_0 = video_dir + "jump/"  # class jump
    trackset_dir_0 = landmark_dir + "jump/"
    videoset_dir_1 = video_dir + "others/"  # class others
    trackset_dir_1 = landmark_dir + "others/"

    jump_videos, jump_tracks, jump_lables = load_videos_tracks(videoset_dir_0, trackset_dir_0, num_image)
    others_videos, others_tracks, others_labels = load_videos_tracks(videoset_dir_1, trackset_dir_1, num_image)
    videos = np.concatenate((jump_videos, others_videos), 0)
    tracks = np.concatenate((jump_tracks, others_tracks), 0)
    labels = np.concatenate((jump_lables, others_labels), 0)
    index = [i for i in range(len(videos))]
    random.shuffle(index)
    videos = videos[index]
    tracks = tracks[index]
    labels = labels[index]
    return videos, tracks, labels


# Test load_all_videos
'''
train_videos, train_tracks, train_lables = load_all_videos("dataset/clips/train/", "dataset/landmarks/train/", 30)
print(train_videos.shape, train_tracks.shape, train_lables.shape)
'''


# Function: Load all train, valid, and test dataset
# video_dir - "dataset/clips/"
# landmark_dir - "dataset/landmarks/"
def load_dataset(video_dir, landmark_dir, num_image):
    train_video_dir, train_landmark_dir = video_dir + "train/", landmark_dir + "train/"
    valid_video_dir, valid_landmark_dir = video_dir + "valid/", landmark_dir + "valid/"
    test_video_dir, test_landmark_dir = video_dir + "test/", landmark_dir + "test/"
    train_videos, train_tracks, train_lables = load_all_videos(train_video_dir, train_landmark_dir, num_image)
    valid_videos, valid_tracks, valid_lables = load_all_videos(valid_video_dir, valid_landmark_dir, num_image)
    test_videos, test_tracks, test_lables = load_all_videos(test_video_dir, test_landmark_dir, num_image)

    return ((train_videos, train_tracks, train_lables),
            (valid_videos, valid_tracks, valid_lables),
            (test_videos, test_tracks, test_lables))


# test load_all_dataset()
'''
(train_videos, train_tracks, train_lables), (valid_videos, valid_tracks, valid_lables), (test_videos, test_tracks, test_lables) =load_dataset("dataset/clips/", "dataset/landmarks/", 30)
print(train_videos.shape)
print(train_tracks.shape)
print(train_lables.shape)
print(valid_videos.shape)
print(valid_tracks.shape)
print(valid_lables.shape)
print(test_videos.shape)
print(test_tracks.shape)
print(test_lables.shape)
'''
'''(train_videos, train_tracks, train_lables), (valid_videos, valid_tracks, valid_lables), (
    test_videos, test_tracks, test_lables) \
    = load_dataset(video_dir="dataset/clips/", landmark_dir="dataset/landmarks/", num_image=30)

np.save("30image/train_videos_30image", train_videos)
np.save("30image/train_tracks_30image", train_tracks)
np.save("30image/train_lables_30image", train_lables)
print(train_videos.shape, train_tracks.shape, train_lables.shape)

np.save("30image/valid_videos_30image", valid_videos)
np.save("30image/valid_tracks_30image", valid_tracks)
np.save("30image/valid_lables_30image", valid_lables)
print(valid_videos.shape, valid_tracks.shape, valid_lables.shape)

np.save("30image/test_videos_30image", test_videos)
np.save("30image/test_tracks_30image", test_tracks)
np.save("30image/test_lables_30image", test_lables)
print(test_videos.shape, test_tracks.shape, test_lables.shape)'''

def get_temp_seq_part(tracks, part_indexs,num_image):
    # part_indexs:
    # left arm:     [2, 3, 4]
    # right arm:    [5, 6, 7]
    # trunk:        [0, 1, 8]
    # left leg:     [9, 10, 11, 24, 22, 23]
    # right leg:    [12, 13, 14, 21, 19, 20]
    temp_seq_part = []
    for i in range(len(tracks)):
        batch = []
        for j in range(num_image):
            image = []
            for k in part_indexs:
                image.append(tracks[i][j][k][0:2])
            batch.append(image)
        temp_seq_part.append(batch)
    temp_seq_part = np.array(temp_seq_part)
    temp_seq_part = temp_seq_part.reshape(len(tracks), num_image, len(part_indexs) * 2)
    return temp_seq_part


def get_temp_seq(tracks,num_image):
    # part_indexs:
    # left_arm:     [2, 3, 4]
    # right_arm:    [5, 6, 7]
    # trunk:        [0, 1, 8]
    # left_leg:     [9, 10, 11, 24, 22, 23]
    # right_leg:    [12, 13, 14, 21, 19, 20]
    temp_seq_larm = get_temp_seq_part(tracks, [2, 3, 4],num_image)
    temp_seq_rarm = get_temp_seq_part(tracks, [5, 6, 7],num_image)
    temp_seq_trunk = get_temp_seq_part(tracks, [0, 1, 8],num_image)
    temp_seq_lleg = get_temp_seq_part(tracks, [9, 10, 11, 24, 22, 23],num_image)
    temp_seq_rleg = get_temp_seq_part(tracks, [12, 13, 14, 21, 19, 20],num_image)
    return (temp_seq_larm,
            temp_seq_rarm,
            temp_seq_trunk,
            temp_seq_lleg,
            temp_seq_rleg)


def get_spat_seq(tracks,num_image):
    '''
    chain sequence:
    1.left hand to right hand:
        4,3,2,5,6,7
    2.head to hip:
         0, 1, 8
    3.left foot to right foot:
        23, 22, 24, 11, 10, 9, 12, 13, 14, 21, 19, 20
    '''
    chain_seq = [1, 2, 3, 4, 3, 2,
                 1, 5, 6, 7, 6, 5,
                 1, 0, 1, 8,
                 9, 10, 11, 23, 22, 24, 11, 10, 9, 8,
                 12, 13, 14, 21, 19, 20, 14, 13, 12, 8,
                 1]
    spat_seq = []
    for i in range(len(tracks)):
        batch = []
        for k in chain_seq:
            joints = []
            for j in range(num_image):
                joints.append(tracks[i][j][k][0:2])
            batch.append(joints)
        spat_seq.append(batch)
    spat_seq = np.array(spat_seq)
    spat_seq = spat_seq.reshape(len(tracks), len(chain_seq), num_image * 2)
    return spat_seq

class WeightedSum(Layer):
    def __init__(self, a, **kwargs):
        self.a = a
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, model_outputs):
        return self.a * model_outputs[0] + (1 - self.a) * model_outputs[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {'a': self.a}
        base_config = super(WeightedSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))