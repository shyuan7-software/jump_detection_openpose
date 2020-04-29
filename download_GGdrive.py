# import requests
#
# def download_file_from_google_drive(id, destination):
#     URL = "https://docs.google.com/uc?export=download"
#
#     session = requests.Session()
#
#     response = session.get(URL, params = { 'id' : id }, stream = True)
#     token = get_confirm_token(response)
#
#     if token:
#         params = { 'id' : id, 'confirm' : token }
#         response = session.get(URL, params = params, stream = True)
#
#     save_response_content(response, destination)
#
# def get_confirm_token(response):
#     for key, value in response.cookies.items():
#         if key.startswith('download_warning'):
#             return value
#
#     return None
#
# def save_response_content(response, destination):
#     CHUNK_SIZE = 32768
#
#     with open(destination, "wb") as f:
#         for chunk in response.iter_content(CHUNK_SIZE):
#             if chunk: # filter out keep-alive new chunks
#                 f.write(chunk)
#
# if __name__ == "__main__":
#     file_id = '13hvZruDwDsk9-AvTuiYOHn8Yj9Tywxv8'
#     destination = './landmark_sample.zip'
#     download_file_from_google_drive(file_id, destination)
import os
import sys

from google_drive_downloader import GoogleDriveDownloader as gdd

from generate_figure import generate_figure_ensemble


def download_video(video_name):
    name2id_video = {'sofa.mp4': '12ns9sHm1AW6VNKckpn7KnFBL0q0iP2YJ',
                     'back_full_power.mp4': '1k-k9vXFjeEan6ajhihnJKhV7xWSvkBwb',
                     'back_small_power.mp4': '1Oms3bft0drBK_s26x-lhfX79XehmcJl7',
                     'bed.mp4': '1wDeM28hxOWk0HHvu-b7bbAwYU2B43CN4',
                     'front_full_power.mp4': '154bw5AR8gSzzZZsstsv3Tfiq2vi5pzKB',
                     'front_small_power.mp4': '15ogdyB6Z06KQIH_wfa9nUdwFyTfiV6aW',
                     'intermedia.mp4': '1cI_YZlOYHrFiROt0bY2by1jW8UokWnWh',
                     'keep_jump.mp4': '1_KQxBQK6AsT3m44v9oncI-cOh5a-ezhK',
                     'no_jump.mp4': '1bc7mjJMDaAmW8IfcXcxhXDXRKoz7zfTx',
                     'random.mp4': '1iMfd4vEtu2BU5KXsv1YKYVP6hWYQgYYi',
                     'raw_video.mp4': '1f2V6B0c5tZeX-jV9tESyZlPKiE8Eqp12',
                     'side_full_power.mp4': '1EFB3shNa1YNJEC492U818aZE6YSe8n4b',
                     'side_small_power.mp4': '1wPcAqbgs7hZnMN1QF5D2Y1y13svJivO1'
                     }
    gdd.download_file_from_google_drive(file_id=name2id_video[video_name],
                                        dest_path='./' + video_name,
                                        unzip=False, overwrite=True)
    return True


def download_landmarks(landmark_name):
    name2id_landmark = {'sofa': '1lSxDsANKpwwZBSe7rlVL3WSuyuHnyWM_',
                        'back_full_power': '1JFhE1PooRrRGyfcbjOXwCMvNp0kwdLJ2',
                        'back_small_power': '1mvMuiNga7ABVJwnbB0kmGyIJ_gBlHSZe',
                        'bed': '1mGgQdE9EzVsEyJGMG-RfubmiW-O3ra7z',
                        'front_full_power': '1fzp3V1tx6DTpX5LjqF7rtLKlPPh0deDe',
                        'front_small_power': '1IXAudTWSQoZIv0d0UM-wfutFY5DvLIQ5',
                        'intermedia': '1PP19WVyGczNOkTFjvt-rUg8zHE9K1XKD',
                        'keep_jump': '1Kq85Xr4kiDj586y-bHbJBoutWJSbLTe0',
                        'no_jump': '14eY6bOt4fNRUVpD16jYKdyg5aWz2e4eq',
                        'random': '16DdzFdLgBtOT5Op7_PYosgx5aRTIJDjJ',
                        'raw_video': '10t-MVogQfDks-xvxw6au8ocHYe9aUAdq',
                        'side_full_power': '1a9HgvOBqx5eQzKu-szqxxFCxzWgKLmgT',
                        'side_small_power': '1k7Jo7EQUleGUTDEaehdjwtpFwm0PdV0J'
                        }
    gdd.download_file_from_google_drive(file_id=name2id_landmark[landmark_name],
                                        dest_path='./' + landmark_name + '.zip',
                                        unzip=True, overwrite=True)
    return True


def download(video_name, landmark_name):
    res1 = download_video(video_name)
    res2 = download_landmarks(landmark_name)
    return res1 and res2


if __name__ == "__main__":
    video_name = sys.argv[1]  # 'no_jump.mp4'
    # video_name = 'no_jump.mp4'  # 'no_jump.mp4'
    landmark_name = os.path.splitext(video_name)[0]
    res = download(video_name, landmark_name)
    if res:
        generate_figure_ensemble(video_path='./' + video_name, landmark_path='./' + landmark_name,
                                 model_path='model/Final_submission/Ensemble_model_trainable_256B_relu_reg/Ensemble_model_trainable_256B_relu_reg_best.h5',
                                 num_image=32)
