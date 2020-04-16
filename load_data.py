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
    # sample_rate = math.ceil(frame_num / num_image)
    sample_rate = frame_num / num_image
    if sample_rate == 0:
        sample_rate += 1
    for frame_id in range(frame_num):
        if len(landmarks) == num_image:
            break
        # if frame_id % sample_rate == 0:
        if int(sample_rate * len(landmarks)) <= frame_id:
            # print(frame_id)
            landmark = decode_json(landmark_path + json_names[frame_id])
            if not np.all(landmark == 0):
                landmarks.append(landmark)
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
        # video, res1 = video_to_imgaes(videos_dir + file, num_image)
        video, res1 = [], True
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


#
#
# (train_videos, train_tracks, train_lables), (valid_videos, valid_tracks, valid_lables), (
#     test_videos, test_tracks, test_lables) \
#     = load_dataset(video_dir="gitignore/dataset_noHMDB/clips/", landmark_dir="gitignore/dataset_noHMDB/landmarks/",
#                    num_image=16)
#
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/train_videos", train_videos)
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/train_tracks", train_tracks)
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/train_lables", train_lables)
# print(train_videos.shape, train_tracks.shape, train_lables.shape)
#
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/valid_videos", valid_videos)
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/valid_tracks", valid_tracks)
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/valid_lables", valid_lables)
# print(valid_videos.shape, valid_tracks.shape, valid_lables.shape)
#
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/test_videos", test_videos)
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/test_tracks", test_tracks)
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/test_lables", test_lables)
# print(test_videos.shape, test_tracks.shape, test_lables.shape)


def get_temp_seq_part(tracks, part_indexs, num_image):
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


def get_temp_seq(tracks, num_image):
    # part_indexs:
    # left_arm:     [2, 3, 4]
    # right_arm:    [5, 6, 7]
    # trunk:        [0, 1, 8]
    # left_leg:     [9, 10, 11, 24, 22, 23]
    # right_leg:    [12, 13, 14, 21, 19, 20]
    temp_seq_larm = get_temp_seq_part(tracks, [2, 3, 4], num_image)
    temp_seq_rarm = get_temp_seq_part(tracks, [5, 6, 7], num_image)
    temp_seq_trunk = get_temp_seq_part(tracks, [0, 1, 8], num_image)
    temp_seq_lleg = get_temp_seq_part(tracks, [9, 10, 11, 24, 22, 23], num_image)
    temp_seq_rleg = get_temp_seq_part(tracks, [12, 13, 14, 21, 19, 20], num_image)
    return (temp_seq_larm,
            temp_seq_rarm,
            temp_seq_trunk,
            temp_seq_lleg,
            temp_seq_rleg)


def get_spat_seq(tracks, num_image):
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
    window_size = num_image // 4
    spat_seq = []
    for i in range(len(tracks)):
        batch = []
        for k in chain_seq:
            joints = []
            for j in range(num_image // 2 - window_size // 2, num_image // 2 + window_size // 2):
                joints.append(tracks[i][j][k][0:2])
            batch.append(joints)
        spat_seq.append(batch)
    spat_seq = np.array(spat_seq)
    spat_seq = spat_seq.reshape(len(tracks), len(chain_seq), window_size * 2)
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


def optical_flow(video_path, num_image):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(video_path)
        return [], False
    frame_num = cap.get(7)
    sample_rate = frame_num / num_image
    if sample_rate == 0:
        sample_rate += 1
    flows = []
    frameId = cap.get(1)
    ret, first_frame = cap.read()
    first_frame = cv2.resize(first_frame, (224, 224), interpolation=cv2.INTER_AREA)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if len(flows) // 2 == num_image:
            break
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_x = flow[..., 1]
        flow_y = flow[..., 0]
        if int(sample_rate * (len(flows) // 2)) <= frameId:
            # print(frameId)
            flows.append(flow_x)
            flows.append(flow_y)
            cv2.imshow("frame", frame)
            cv2.imshow("x", flow_x)
            cv2.imshow("y", flow_y)
            cv2.waitKey()
            # if cv2.waitKey(100) & 0xFF == ord('q'):
            #     break
        prev_gray = gray
        frameId = cap.get(1)
    # print(len(flows))
    empty_flow = np.zeros((224, 224, 1), np.uint8)
    while len(flows) // 2 < num_image:
        flows.append(empty_flow)
        flows.append(empty_flow)
    cap.release()
    cv2.destroyAllWindows()
    res = np.stack(flows, axis=2)
    # for i in range(20):
    #     cv2.imshow("y", res[...,i])
    #     cv2.waitKey(1000)
    return res, True


# optical_flow('gitignore/dataset/clips/test/jump/jump104.mp4', 10)
def single_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(video_path)
        return [], False
    frame_num = cap.get(7)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num // 2)
    success, frame = cap.read()
    # cv2.imshow('b', frame)
    # cv2.waitKey(1000)
    cap.release()
    cv2.destroyAllWindows()
    return cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA), True


# single_frame('gitignore/dataset/clips/test/jump/jump104.mp4')

def load_flows_frame(videos_dir, num_image):
    dirs = os.listdir(videos_dir)
    flows = []
    frames = []
    labels = []
    for file in dirs:
        print(videos_dir + file)
        flow, res1 = optical_flow(videos_dir + file, num_image)
        frame, res2 = single_frame(videos_dir + file)
        if not res1 or not res2:
            continue
        flows.append(flow)
        frames.append(frame)
        if "jump" in file:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(flows), np.array(frames), np.array(labels)


# a, b, c = load_flows_frame('gitignore/dataset/clips/test/jump/', 2)
# print(a.shape, b.shape, c.shape)

def load_all_flows_frame(video_dir, num_image):
    videoset_dir_0 = video_dir + "jump/"  # class jump
    videoset_dir_1 = video_dir + "others/"  # class others
    others_flows, others_frames, others_labels = load_flows_frame(videoset_dir_1, num_image)
    jump_flows, jump_frames, jump_lables = load_flows_frame(videoset_dir_0, num_image)

    flows = np.concatenate((jump_flows, others_flows), 0)
    frames = np.concatenate((jump_frames, others_frames), 0)
    labels = np.concatenate((jump_lables, others_labels), 0)
    index = [i for i in range(len(labels))]
    random.shuffle(index)
    flows = flows[index]
    frames = frames[index]
    labels = labels[index]
    return flows, frames, labels


# a, b, c = load_all_flows_frame('gitignore/dataset/clips/train/', 10)
# print(a.shape, b.shape, c.shape)

def load_dataset_CNN(video_dir, num_image):
    train_video_dir = video_dir + "train/"
    valid_video_dir = video_dir + "valid/"
    test_video_dir = video_dir + "test/"
    train_flows, train_frames, train_lables = load_all_flows_frame(train_video_dir, num_image)
    valid_flows, valid_frames, valid_lables = load_all_flows_frame(valid_video_dir, num_image)
    test_flows, test_frames, test_lables = load_all_flows_frame(test_video_dir, num_image)

    return ((train_flows, train_frames, train_lables),
            (valid_flows, valid_frames, valid_lables),
            (test_flows, test_frames, test_lables))


#
#
# ((train_flows, train_frames, train_lables), \
#  (valid_flows, valid_frames, valid_lables), \
#  (test_flows, test_frames, test_lables)) \
#     = load_dataset_CNN(video_dir="./gitignore/dataset_noHMDB/clips/", num_image=10)
#
# np.save("gitignore/npy/10imgae_noHMDB_cnn/train_flows", train_flows)
# np.save("gitignore/npy/10imgae_noHMDB_cnn/train_frames", train_frames)
# np.save("gitignore/npy/10imgae_noHMDB_cnn/train_lables", train_lables)
# print(train_flows.shape, train_frames.shape, train_lables.shape)
#
# np.save("gitignore/npy/10imgae_noHMDB_cnn/valid_flows", valid_flows)
# np.save("gitignore/npy/10imgae_noHMDB_cnn/valid_frames", valid_frames)
# np.save("gitignore/npy/10imgae_noHMDB_cnn/valid_lables", valid_lables)
# print(valid_flows.shape, valid_frames.shape, valid_lables.shape)
#
# np.save("gitignore/npy/10imgae_noHMDB_cnn/test_flows", test_flows)
# np.save("gitignore/npy/10imgae_noHMDB_cnn/test_frames", test_frames)
# np.save("gitignore/npy/10imgae_noHMDB_cnn/test_lables", test_lables)
# print(test_flows.shape, test_frames.shape, test_lables.shape)

def draw_skeleton_pic(landmark, scale=224):
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    edges = np.zeros((scale, scale))
    edges[0, 1] = edges[1, 8] = edges[0, 15] = edges[0, 16] = edges[15, 17] = edges[16, 18] = 1
    edges[1, 2] = edges[2, 3] = edges[3, 4] = edges[1, 5] = edges[5, 6] = edges[6, 7] = 2
    edges[8, 9] = edges[9, 10] = edges[10, 11] = edges[11, 24] = edges[11, 22] = edges[22, 23] = \
        edges[8, 12] = edges[12, 13] = edges[13, 14] = edges[14, 21] = edges[14, 19] = edges[19, 20] = 3
    canvas = np.zeros((scale, scale, 3), np.uint8)
    for i in range(25):
        for j in range(i + 1, 25):
            if edges[i, j] > 0:
                x1, y1, c1 = int(landmark[i, 0] * scale), int(landmark[i, 1] * scale), landmark[i, 2]
                x2, y2, c2 = int(landmark[j, 0] * scale), int(landmark[j, 1] * scale), landmark[j, 2]
                # if x1 == y1 == 0 or x2 == y2 == 0:
                if c1 == 0 or c2 == 0:
                    continue
                cv2.line(canvas, (x1, y1), (x2, y2), colors[int(edges[i, j] - 1)], thickness=3)
    # cv2.imshow("Canvas", canvas)
    # cv2.waitKey(0)
    # print(type(canvas))
    return canvas


def draw_ST_skeleton_pic(video, scale=224):
    white = (255, 255, 255)

    canvas = np.zeros((scale, scale * len(video), 3), np.uint8)
    for i in range(len(video)):
        canvas[0:scale, scale * i:scale * (i + 1)] = draw_skeleton_pic(video[i])
        if i > 0:
            for j in range(25):
                if j in [15, 16, 17, 18]:
                    continue
                cur_x, cur_y, cur_c = scale * i + int(scale * video[i, j, 0]), int(scale * video[i, j, 1]), video[
                    i, j, 2]
                prv_x, prv_y, prv_c = scale * (i - 1) + int(scale * video[i - 1, j, 0]), int(
                    scale * video[i - 1, j, 1]), video[i - 1, j, 2]
                if cur_c == 0 or prv_c == 0:
                    continue
                cv2.line(canvas, (cur_x, cur_y), (prv_x, prv_y), white, thickness=1)
    # cv2.namedWindow("Canvas", cv2.WINDOW_FREERATIO)
    # cv2.imshow("Canvas", canvas)
    # cv2.waitKey(0)
    return canvas


def tracks_to_videos(tracks, num_image):
    videos = []
    for i in range(len(tracks)):
        video = []
        for j in range(num_image):
            frame = draw_skeleton_pic(tracks[i, j])
            video.append(frame)
        videos.append(video)
    return np.array(videos)


def tracks_to_STimages(tracks):
    STimages = []
    for i in range(len(tracks)):
        STimage = draw_ST_skeleton_pic(tracks[i])
        STimages.append(STimage)
    return np.array(STimages)


# valid_STimages = tracks_to_STimages(valid_tracks)
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/valid_STimages", valid_STimages)
# print(valid_STimages.shape)
#
# test_STimages = tracks_to_STimages(test_tracks)
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/test_STimages", test_STimages)
# print(test_STimages.shape)
#
# train_STimages = tracks_to_STimages(train_tracks)
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/train_STimages", train_STimages)
# print(train_STimages.shape)
#
#
# train_skeleton_videos = tracks_to_videos(train_tracks, 16)
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/train_skeleton_videos", train_skeleton_videos)
# print(train_skeleton_videos.shape)
#
# test_skeleton_videos = tracks_to_videos(test_tracks, 16)
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/test_skeleton_videos", test_skeleton_videos)
# print(test_skeleton_videos.shape)
#
# valid_skeleton_videos = tracks_to_videos(valid_tracks, 16)
# np.save("gitignore/npy/16image_noHMDB_noEmptyFrame/valid_skeleton_videos", valid_skeleton_videos)
# print(valid_skeleton_videos.shape)
# videos = np.load("gitignore/npy/16image_noHMDB_noEmptyFrame/test_skeleton_videos.npy")
# labels = np.load("gitignore/npy/16image_noHMDB_noEmptyFrame/test_lables.npy")
# for i in range(len(videos)):
#     if labels[i] == 0:
#         for j in range(16):
#             cv2.imshow('.',videos[i,j])
#             cv2.waitKey(0)

dataset_path = "gitignore/npy/32image_noHMDB_noEmptyFrame/"
num_image = 32


def load_np_data():
    train_tracks = np.load(dataset_path + "/train_tracks.npy")
    train_lables = np.load(dataset_path + "/train_lables.npy")

    valid_tracks = np.load(dataset_path + "/valid_tracks.npy")
    valid_lables = np.load(dataset_path + "/valid_lables.npy")

    test_tracks = np.load(dataset_path + "/test_tracks.npy")
    test_lables = np.load(dataset_path + "/test_lables.npy")
    return (train_tracks, train_lables), (valid_tracks, valid_lables), (test_tracks, test_lables)


# (train_tracks, train_labels), (valid_tracks, valid_labels), (test_tracks, test_labels) = load_np_data()
# print(train_tracks.shape)

def get_diff(prv_landmark, cur_landmark):
    diff = []
    for i in range(len(prv_landmark)):
        prv_x, prv_y, prv_c = prv_landmark[i]
        cur_x, cur_y, cur_c = cur_landmark[i]
        dx = cur_x - prv_x
        dy = cur_y - prv_y
        if cur_c == 0 or prv_c == 0:
            diff.append([0, 0])
        else:
            diff.append([dx, dy])
    return np.array(diff)


def get_cart_coord(landmark):
    cart_coord = []
    for i in range(len(landmark)):
        x, y, c = landmark[i]
        cart_coord.append([x, y])
    return np.array(cart_coord)


def get_dataset_diff_based_CNN(tracks, num_image):
    motions = []
    coords = []
    for i in range(len(tracks)):
        coord = []
        motion = []
        for j in range(num_image):
            landmark = tracks[i, j]
            coord.append(get_cart_coord(landmark))
            if j > 0:
                motion.append(get_diff(tracks[i, j - 1], landmark))
        motions.append(motion)
        coords.append(coord)
    return np.array(coords), np.array(motions)


# train_coords, train_motions = get_dataset_diff_based_CNN(train_tracks, num_image)
# np.save(dataset_path + 'train_coords', train_coords)
# np.save(dataset_path + 'train_motions', train_motions)
# print(train_coords.shape, train_motions.shape)
#
# valid_coords, valid_motions = get_dataset_diff_based_CNN(valid_tracks, num_image)
# np.save(dataset_path + 'valid_coords', valid_coords)
# np.save(dataset_path + 'valid_motions', valid_motions)
# print(valid_coords.shape, valid_motions.shape)
#
# test_coords, test_motions = get_dataset_diff_based_CNN(test_tracks, num_image)
# np.save(dataset_path + 'test_coords', test_coords)
# np.save(dataset_path + 'test_motions', test_motions)
# print(test_coords.shape, test_motions.shape)
