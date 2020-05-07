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
                     'side_small_power.mp4': '1wPcAqbgs7hZnMN1QF5D2Y1y13svJivO1',
                     '3600_2.mkv':'1vG3iofKLCLPaMY231Ow_ANSzTyS4o7MI',
                     '3700_1.mkv':'1nL1Sy0NrLk_qxfcJ-mcnUc6xk1ybHdYt',
                     '3800_3.mkv':'1aMEsNW4t1X4TP3Ikj9lCjIuuOP4WN1jQ',
                     '3900_2.mkv':'12JVsOuyQxureJqMxPOy_sJSjVYbJ5veH',
                     '4000_1.mkv':'1ErcaVkec4jvXiIwQKhmmA-SPMM26n1EB',
                     '4000_2.mkv':'1cjrdLz9zjRbfPd43m-8uEs4t4WcYSHu0',
                     '4000_3.mkv':'1UIVO_IKa9n6MSwNieAigf3ZapqmUSMuh',
                     '4000_4.mkv':'1k6O7sORO7MR-i6ZeCcxcy7zDaEj5Z0f7',
                     '4100_1.mkv':'1OrPypJrySicR7Ymnw6-dU8y2sc4jjspD',
                     '4200_3.mkv':'1tMdyT32bQwqEW-GcIHJyk7OasEWW0YMa',
                     '4300_3.mkv':'1nryMkBcFd8yrjDyZ8dTM7pQHrhrlApxB',
                     '4400_1.mkv':'1GkJMJLg1Sxqh3w9aix5gCH4m4_9z56kn'
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
                        'side_small_power': '1k7Jo7EQUleGUTDEaehdjwtpFwm0PdV0J',
                        '3600_2': '1FcjpvvJKRd2SAPTblmpPQ6_HMHkZmZay',
                        '3700_1': '19UDo2gi4GaoxY5cCo3b1Hn6cOngYTGmV',
                        '3800_3': '1VEVzxWMT_GIY4MyiifXRkzJx8kYooDg4',
                        '3900_2': '1NkUgTNJZOA6rVCNYa_QcKHhqiMYotytF',
                        '4000_1': '10jXtwUt7Q8z1Ox2hdblYC9tSa3H615At',
                        '4000_2': '1rEsthHdpDeNugechNvsHytBxZdN02AdV',
                        '4000_3': '1I-1REbTWSfM-7Xea0tf9IjGcTezOPHUF',
                        '4000_4': '1FS__iZ1JalzC3LL0HNw5TFXNgEenYBNW',
                        '4100_1': '1dS879WAyO1QmEQSNBhLuSs3M0HwuJRHy',
                        '4200_3': '1u5p-cYZUZNXVJXWfBvmPZf3Jm3sxswSi',
                        '4300_3': '14X131qctCHbEeUzhtPuOgLL97yo-kgXN',
                        '4400_1': '10gxUkIcZ2HW-TO_avaMrAsZ4FWJGvMSs'
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
    landmark_name = os.path.splitext(video_name)[0]  # ‘no_jump’
    res = download(video_name, landmark_name)
    if res:
        os.system('python generate_figure.py ' + video_name + ' ' + landmark_name)
