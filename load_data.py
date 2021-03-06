# Author: Shaohua Yuan
# Email: shyuan@tamu.edu

import json as js
import math
import os
import random
import cv2
import numpy as np
from keras.engine import Layer

# video_path- "dataset/landmarks/test/jump/jump1.mp4/jump1_000000000000_keypoints.json"
# Function: decode a landmark json file
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression


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
        video, res1 = video_to_imgaes(videos_dir + file, num_image)
        # video, res1 = [], True
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
    valid_videos, valid_tracks, valid_lables = load_all_videos(valid_video_dir, valid_landmark_dir, num_image)
    test_videos, test_tracks, test_lables = load_all_videos(test_video_dir, test_landmark_dir, num_image)
    train_videos, train_tracks, train_lables = load_all_videos(train_video_dir, train_landmark_dir, num_image)

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



# Transfering body landmarks to a temporal sequence, which is described in https://arxiv.org/abs/1704.02581
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


# Transfering body landmarks to temporal sequences, which is described in https://arxiv.org/abs/1704.02581
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


# Transfering body landmarks to a spatial sequence, which is described in https://arxiv.org/abs/1704.02581
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
    # The chain_seq is the order of traversing skeleton joints
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


# A customized layer for fusing temporal stream and spatial stream, which is described in https://arxiv.org/abs/1704.02581
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


# Generate a number of optical flow frames for a video
# If generated successfully, return a list of optial flow frames and True
# Else return a empty list and False
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

# Sample one frame from one video,
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

# Load dataset for the input of the model, which is described in
# https://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf
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

# Load train/valid/test dataset
# https://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf
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

# Load all dataset for the model decribed in
# # https://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf
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


# Rotate skeleton image:
def rotate90(landmark, k):
    new_landmark = []
    x0, y0 = 0.5, 0.5  # the center of image
    for i in range(25):
        x1, y1, c1 = landmark[i, 0], landmark[i, 1], landmark[i, 2]
        x2 = math.cos(k*(-math.pi / 2)) * (x1 - x0) - math.sin(k*(-math.pi / 2)) * (y1 - y0) + x0
        y2 = math.sin(k*(-math.pi / 2)) * (x1 - x0) + math.cos(k*(-math.pi / 2)) * (y1 - y0) + y0
        new_landmark.append([x2, y2, c1])
    return np.array(new_landmark)


# Draw the picture of human body, based on body landmark
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


# Draw human body of all frames on a convas, based on body landmarks
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


# Transfering body landmarks dataset to the dataset that contains the image drawed based on body landmarks
def tracks_to_videos(tracks, num_image):
    videos = []
    for i in range(len(tracks)):
        video = []
        for j in range(num_image):
            frame = draw_skeleton_pic(tracks[i, j])
            video.append(frame)
        videos.append(video)
    return np.array(videos)


# Transfering body landmarks dataset to the dataset that contains the image drawed based on body landmarks
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

# dataset_path = "gitignore/npy/32image_noHMDB/"
# num_image = 32


# Load dataset through loading .npy files, much faster than loading original video and body landmark files
def load_np_data(path):
    train_tracks = np.load(path + "/train_tracks.npy")
    train_lables = np.load(path + "/train_lables.npy")

    valid_tracks = np.load(path + "/valid_tracks.npy")
    valid_lables = np.load(path + "/valid_lables.npy")

    test_tracks = np.load(path + "/test_tracks.npy")
    test_lables = np.load(path + "/test_lables.npy")
    return (train_tracks, train_lables), (valid_tracks, valid_lables), (test_tracks, test_lables)


# (train_tracks, train_labels), (valid_tracks, valid_labels), (test_tracks, test_labels) = load_np_data()
# print(train_tracks.shape)

# Get difference between two frames of body landmarks, used in the model of
# https://arxiv.org/pdf/1704.07595.pdf
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


# Get Cartesian Coordinates, based on body landmark, used in the model of
# https://arxiv.org/pdf/1704.07595.pdf
def get_cart_coord(landmark):
    cart_coord = []
    for i in range(len(landmark)):
        x, y, c = landmark[i]
        cart_coord.append([x, y])
    return np.array(cart_coord)


# Transfering original body landmark dataset, to the dataset that can be used in the model of
# https://arxiv.org/pdf/1704.07595.pdf
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


# (train_tracks, train_labels), (valid_tracks, valid_labels), (test_tracks, test_labels) = load_np_data()
# print(train_tracks.shape)
# old_labels = test_labels
# new_labels = to_categorical(old_labels)
# for i in range(len(new_labels)):
#     print(old_labels[i], new_labels[i])


def label_aug(label):
    return [label, label, label, label]


def track_aug(track):
    rotated_tracks = [track]
    for k in [1, 2, 3]:
        new_track = []
        for frame_id in range(len(track)):
            new_track.append(rotate90(track[frame_id], k=k))
        rotated_tracks.append(new_track)
    return rotated_tracks


def video_aug(video):
    rotated_videos = [video]
    for k in [1,2,3]:
        new_video = []
        for frame_id in range(len(video)):
            new_video.append(np.rot90(video[frame_id], k=k))
        rotated_videos.append(new_video)
    return rotated_videos

# Data Augmentation by rotating images 90, 180, 270. Therefore, the original dataset will be increased to 4 times larger than before
def data_aug(videos, tracks, labels):
    size = len(tracks)
    new_videos = []
    new_tracks = []
    new_labels = []
    for i in range(size):
        new_videos.extend(video_aug(videos[i]))
        new_tracks.extend(track_aug(tracks[i]))
        new_labels.extend(label_aug(labels[i]))
    new_videos = np.array(new_videos)
    new_tracks = np.array(new_tracks)
    new_labels = np.array(new_labels)
    index = [i for i in range(len(new_videos))]
    random.shuffle(index)
    new_videos = new_videos[index]
    new_tracks = new_tracks[index]
    new_labels = new_labels[index]
    return new_videos, new_tracks, new_labels


# (train_tracks, train_labels), (valid_tracks, valid_labels), (test_tracks, test_labels) = load_np_data(dataset_path)
# test_videos = np.load(dataset_path+'/test_videos.npy')
# print(test_tracks.shape)
# test_videos, test_tracks, test_labels = data_aug(test_videos, test_tracks, test_labels)
# print(test_tracks.shape)、

# Load one type of images, like jump or others
def load_one_type_imgs(imgs_dir):
    files = os.listdir(imgs_dir)
    imgs = []
    for file in files:
        img = cv2.imread(imgs_dir+'/'+file)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        imgs.append(img)
    return imgs

# Load one type of dataset, like train, valid, or test
def load_imgs(path):
    imgs = []
    labels = []
    jump_imgs = load_one_type_imgs(path+'/jump/')
    jump_lbes = [1] * len(jump_imgs)
    others_imgs = load_one_type_imgs(path + '/others/')
    others_lbes = [0] * len(others_imgs)
    imgs.extend(jump_imgs)
    imgs.extend(others_imgs)
    labels.extend(jump_lbes)
    labels.extend(others_lbes)
    imgs=np.array(imgs)
    labels = np.array(labels)
    index = [i for i in range(len(labels))]
    random.shuffle(index)
    imgs = imgs[index]
    labels = labels[index]
    return imgs, labels


# Load all img dataset
def load_img_dataset(path):
    train_imgs, train_labels = load_imgs(path+'/train/')
    valid_imgs, valid_labels = load_imgs(path + '/valid/')
    test_imgs, test_labels = load_imgs(path + '/test/')
    return train_imgs, train_labels,valid_imgs, valid_labels, test_imgs, test_labels

if __name__ == '__main__':
    # (train_videos, train_tracks, train_lables), (valid_videos, valid_tracks, valid_lables), (
    #     test_videos, test_tracks, test_lables) \
    #     = load_dataset(video_dir="gitignore/Final_dataset/clips/", landmark_dir="gitignore/Final_dataset/landmarks/",
    #                    num_image=32)
    #
    # np.save("gitignore/npy/Final_dataset_npy/train_videos", train_videos)
    # np.save("gitignore/npy/Final_dataset_npy/train_tracks", train_tracks)
    # np.save("gitignore/npy/Final_dataset_npy/train_lables", train_lables)
    # print(train_videos.shape, train_tracks.shape, train_lables.shape)
    #
    # np.save("gitignore/npy/Final_dataset_npy/valid_videos", valid_videos)
    # np.save("gitignore/npy/Final_dataset_npy/valid_tracks", valid_tracks)
    # np.save("gitignore/npy/Final_dataset_npy/valid_lables", valid_lables)
    # print(valid_videos.shape, valid_tracks.shape, valid_lables.shape)
    #
    # np.save("gitignore/npy/Final_dataset_npy/test_videos", test_videos)
    # np.save("gitignore/npy/Final_dataset_npy/test_tracks", test_tracks)
    # np.save("gitignore/npy/Final_dataset_npy/test_lables", test_lables)
    # print(test_videos.shape, test_tracks.shape, test_lables.shape)
    '''
    train_videos = np.load("gitignore/npy/Final_dataset_npy/train_videos.npy")
    train_labels = np.load("gitignore/npy/Final_dataset_npy/train_lables.npy")
    train_tracks = np.load("gitignore/npy/Final_dataset_npy/train_tracks.npy")'''



