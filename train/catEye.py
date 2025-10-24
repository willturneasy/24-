import math
import os
import shutil
import time
import uuid

import cv2
from ultralytics import YOLO

from train import make_path_exit
from trainClassify import get_model_pre_str


def get_point_and_conf(key_point, shape=(640, 640), min_conf=0.5):
    """
    获取点位和可信度
    :param min_conf:
    :param key_point:
    :param shape:
    :return:
    """
    x_coord, y_coord = key_point[0], key_point[1]
    if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
        if len(key_point) == 3:
            conf = key_point[2]
            if conf < min_conf:
                return None
            return float(x_coord), float(y_coord), float(conf)
    return None


def get_eye_attr(eye_center, eyes_distance, image, half=1):
    if half > 1:
        eyes_distance = eyes_distance / half
    eye_top, eye_left, eye_bottom, eye_right = int(eye_center[1] - eyes_distance / 3), \
        int(eye_center[0] - eyes_distance / 2), \
        int(eye_center[1] + eyes_distance / 3), \
        int(eye_center[0] + eyes_distance / 2)
    cv2.imwrite(f'E:/数据集/未分类数据集/classify/{uuid.uuid1()}.jpg', image[eye_top:eye_bottom, eye_left:eye_right])


if __name__ == '__main__':
    # path_to_data_set = "E:/数据集/分类后的数据集"
    # pose_model = YOLO("models/yolov8n-pose.pt", task="pose")  # load a pretrained model (recommended for training)
    # dir_list = os.listdir(path_to_data_set)
    # dirs = [f for f in dir_list if os.path.isdir(os.path.join(path_to_data_set, f))]
    # for d in dirs:
    #     # print(os.path.join(path_to_data_set,f"{d}/JPEGImages"))
    #     img_path = os.path.join(path_to_data_set, f"{d}/JPEGImages")
    #     if os.path.exists(img_path):
    #         results = pose_model(img_path)
    #         for r in results:
    #             # print(r)
    #             if len(r.keypoints.data) > 0 and len(r.keypoints.data[0]) > 3:
    #                 left_eye = get_point_and_conf(r.keypoints.data[0][1], r.orig_shape)
    #                 right_eye = get_point_and_conf(r.keypoints.data[0][2], r.orig_shape)
    #                 if left_eye is not None and right_eye is not None:
    #                     length = math.sqrt((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2)
    #                     half_len = int(length / 2)
    #                     get_eye_attr(left_eye, half_len, r.orig_img)
    #                     # get_eye_attr(left_eye, half_len, img, half=2)
    #                     # get_eye_attr(left_eye, half_len, img, half=3)
    #                     # get_eye_attr(left_eye, half_len, img, half=4)
    #                     get_eye_attr(right_eye, half_len, r.orig_img)
    # count = 0
    # for root, _, files in os.walk("E:/数据集/未分类数据集/classify"):
    #     for f in files:
    #         if f.endswith(".jpg"):
    #             f = os.path.join(root, f)
    #             if os.stat(f).st_size < 860:
    #                 os.remove(f)
    #                 count += 1
    # print(count)
    # base_path = "E:/数据集/未分类数据集/classify"
    # eye_model = YOLO("runs/classify/train13/weights/best.pt", task="classify")
    # results = eye_model(base_path)
    # for r in results:
    #     shutil.move(r.path, os.path.join(base_path, f"{eye_model.names[r.probs.top1]}/"))
    # class_list = ["closeEye", "openEye"]
    # path_to_data_set = "E:/数据集/自用场景数据集/classify"
    # path_to_save = "E:/数据集/训练的数据集"
    # date = time.strftime("%Y-%m-%d")
    # for root, _, files in os.walk("runs/classify"):
    #     for f in files:
    #         if f.endswith("best.pt"):
    #             f = os.path.join(root, f)
    #             eye_model = YOLO(f, task="classify")
    #             model_name = ""
    #             for cls in class_list:
    #                 results = eye_model(os.path.join(path_to_data_set, f"{cls}/"))
    #                 all_count = len(results)
    #                 error_count = 0
    #                 for r in results:
    #                     if eye_model.names[r.probs.top1] != cls:
    #                         error_count += 1
    #                 cls_pre = int(100 * (all_count - error_count) / all_count)
    #                 print(f"{cls}:{cls_pre}")
    #                 model_name = f"{cls}_{cls_pre}_{model_name}"
    #             shutil.copy(f, os.path.join(path_to_save, f'{date}/{model_name}_{uuid.uuid4()}.pt'))
    # all_path = os.path.join(path_to_save, os.path.join("-".join(class_list), "all"))
    # make_path_exit(all_path)
    # for root, _, files in os.walk(os.path.join(path_to_save, "-".join(class_list))):
    #     for f in files:
    #         if f.endswith(".pt"):
    #             f = os.path.join(root, f)
    #             model = YOLO(f, task="classify")
    #             # if model.names.keys()
    #             class_check = True
    #             for key in model.names:
    #                 if model.names[key] not in class_list:
    #                     class_check = False
    #                     break
    #
    #             if class_check:
    #                 class_check = len(model.names.keys()) == len(class_list)
    #             if class_check:
    #                 model_name = get_model_pre_str(f, class_list, path_to_data_set)
    #                 if model_name is not None:
    #                     shutil.move(f, os.path.join(all_path, f'{model_name}_{uuid.uuid4()}.pt'))

    l_n = [6, 3, 2, 1, 5]
    l_s = ["6", "3", "2", "1", "5"]
    new_l_n = []
    new_l_s = []

    for i, n in enumerate(l_n):
        for index, nn in enumerate(new_l_n):
            if n > nn:
                new_l_n.insert(index, n)
                new_l_s.insert(index, l_s[i])
                break
        if len(new_l_n) <= i:
            new_l_n.append(n)
            new_l_s.append(l_s[i])

    print(new_l_s)
