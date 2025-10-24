#
#
#
#
# 脸部特征点识别，用于眼部和嘴部特征点的识别
#
#
#
#
import math

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

detector = dlib.get_frontal_face_detector()  # 人脸检测器
# model_path = "model/mmod_human_face_detector.dat"
# detector = dlib.cnn_face_detection_model_v1(model_path)
shape_predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_predictor_path)  # 人脸68点提取器


def shape_to_np(shape):
    """
    将dlib的shape转换为numpy数组方便处理
    :param shape:
    :return:
    """
    # 创建68*2
    coords = np.zeros((shape.num_parts, 2), dtype="int")
    # 遍历每一个关键点
    # 得到坐标
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def point_to_dlib_rectangle(x, y, w, h):
    return dlib.rectangle(x, y, x + w, y + h)


def eye_aspect_ratio(eye):
    """
    计算眼部长宽比
    :param eye:
    :return:
    """
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear * 100


def mouth_aspect_ratio(mouth):
    """
    嘴部长宽比
    :param mouth:
    :return:
    """
    A = dist.euclidean(mouth[2], mouth[9])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[7])  # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar * 100


def get_pose_estimation(img_size, shape):
    """
    获取旋转向量和平移向量
    :param img_size:
    :param shape:
    :return:
    """
    # 3D model points.
    model_points = np.array([
        (6.825897, 6.760612, 4.402142),  # 33 left brow left corner
        (1.330353, 7.122144, 6.903745),  # 29 left brow right corner
        (-1.330353, 7.122144, 6.903745),  # 34 right brow left corner
        (-6.825897, 6.760612, 4.402142),  # 38 right brow right corner
        (5.311432, 5.485328, 3.987654),  # 13 left eye left corner
        (1.789930, 5.393625, 4.413414),  # 17 left eye right corner
        (-1.789930, 5.393625, 4.413414),  # 25 right eye left corner
        (-5.311432, 5.485328, 3.987654),  # 21 right eye right corner
        (2.005628, 1.409845, 6.165652),  # 55 nose left corner
        (-2.005628, 1.409845, 6.165652),  # 49 nose right corner
        (2.774015, -2.080775, 5.048531),  # 43 mouth left corner
        (-2.774015, -2.080775, 5.048531),  # 39 mouth right corner
        (0.000000, -3.116408, 6.097667),  # 45 mouth central bottom corner
        (0.000000, -7.415691, 4.070434)  # 6 chin corner
    ])
    # Camera internals

    focal_length = img_size[1]
    center = (img_size[1] / 2, img_size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.array([7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000],
                           dtype="double")  # Assuming no lens distortion

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, np.array(shape[[17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]], dtype="double"),
                                                                  camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs


def get_euler_angle(rotation_vector):
    """
    从旋转向量转换为欧拉角
    :param rotation_vector:
    :return:
    """
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)

    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0][0] / theta
    y = math.sin(theta / 2) * rotation_vector[1][0] / theta
    z = math.sin(theta / 2) * rotation_vector[2][0] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)

    # print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    # 单位转换：将弧度转换为度
    pitch_degree = int((pitch / math.pi) * 180)
    yaw_degree = int((yaw / math.pi) * 180)
    roll_degree = int((roll / math.pi) * 180)

    return pitch, yaw, roll, pitch_degree, yaw_degree, roll_degree


def position_area(position):
    """
    计算区域面积
    :param position: dlib.rectangle
    :return: 面积
    """
    return abs((position.right() - position.left()) * (position.bottom() - position.top()))


def largest_face(det_list):
    face_areas = [abs((det.right() - det.left()) * (det.bottom() - det.top())) for det in det_list]
    return np.argmax(face_areas, axis=0)


def set_in_range(value, max_val, min_val=0):
    return min(max(value, min_val), max_val)


def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst

def get_area_face_attr(img, face_area):
    """
    返回人脸数据
    :param img:
    :param face_area:
    :return: 左眼长宽比，右眼长宽比，嘴部长宽比，左右偏转角，上下偏转角，倾斜偏转角
    """
    # face_area = list(face_area)
    # h, w = img.shape[:2]
    # for i in range(len(face_area)):
    #     if i % 2 == 0:
    #         face_area[i] = set_in_range(face_area[i], w)
    #     else:
    #         face_area[i] = set_in_range(face_area[i], h)

    face_area = dlib.rectangle(*face_area)
    temp_img = img.copy()
    # temp_img = unevenLightCompensate(temp_img,16)

    # temp_img = cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)
    # temp_img = np.uint8(255 * (temp_img / cv2.GaussianBlur(temp_img,(5,5),0)))

    # cv2.equalizeHist(temp_img)
    # temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
    # temp_img[face_area.top():face_area.bottom(), face_area.left():face_area.right()] = cv2.equalizeHist(
    #     temp_img[face_area.top():face_area.bottom(), face_area.left():face_area.right()])
    # cv2.imshow("tar", temp_img)
    shape = predictor(temp_img, face_area)
    shape = shape_to_np(shape)
    left_eye = eye_aspect_ratio(shape[36:42])
    right_eye = eye_aspect_ratio(shape[42:48])
    mouth = mouth_aspect_ratio(shape[48:67])

    img = draw_eye_area(img, shape[36:42])
    img = draw_eye_area(img, shape[42:48])
    img = draw_eye_area(img, shape[48:67])
    ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(img.shape, shape)
    pitch, yaw, roll, pitch_degree, yaw_degree, roll_degree = get_euler_angle(rotation_vector)
    return left_eye, right_eye, mouth, yaw_degree, pitch_degree, roll_degree


def draw_eye_area(img, eye_points):
    eye_area = cv2.convexHull(eye_points)
    return cv2.drawContours(img, [eye_area], -1, (0, 255, 255), 1)
