import cv2
from scipy.spatial import distance as dist
from ultralytics import YOLO

# coco身形点位定义
from face import get_area_face_attr

COCO_POINTS = {
    # 鼻子
    "nose": 0,
    # 左眼
    "l_eye": 1,
    # 右眼
    "r_eye": 2,
    # 左耳
    "l_ear": 3,
    # 右耳
    "r_ear": 4,
    # 左肩
    "l_sho": 5,
    # 右肩
    "r_sho": 6,
    # 左手肘
    "l_elb": 7,
    # 右手肘
    "r_elb": 8,
    # 左手腕
    "l_wri": 9,
    # 右手腕
    "r_wri": 10,
    # 左臀部
    "l_hip": 11,
    # 右臀部
    "r_hip": 12,
    # 左膝盖
    "l_knee": 13,
    # 右膝盖
    "r_knee": 14,
    # 左脚腕
    "l_ank": 15,
    # 右脚腕
    "r_ank": 16,
}

# 使用预训练好的模型
pose_model = YOLO("models/yolov8n-pose.pt", task="pose")  # load a pretrained model (recommended for training)
model = YOLO("models/best.pt")  # load a pretrained model (recommended for training)


def get_box_data(box, names):
    """
    获取探测结果
    :param box:
    :param names:
    :return:
    """
    return int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(100 * box[4]), names[int(box[5])]


def results_to_list(target_results, names):
    """
    将探测结果放在一个列表里
    :param target_results:
    :param names:
    :return:
    """
    result_list = []
    for res in target_results:
        for box in res.boxes.data:
            result_list.append(get_box_data(box, names))
    return result_list


def get_max_pose(target_results):
    """
    获取最大的身形
    :param target_results:
    :return:
    """
    max_area = 0
    max_pose = None
    for res in target_results:
        for index, box in enumerate(res.boxes.data):
            now_area = abs(box[2] * box[3])
            if now_area > max_area:
                max_area = now_area
                max_pose = res.keypoints.data[index]
    return max_pose


def get_pose_attr(frame, results):
    is_near = None
    face_attr = None
    min_len = -1
    min_type = ""
    # 由于手腕定位不准，身形定位的最大意义就是定位脸部区域
    # 寻找最大身形
    pose_key_points = get_max_pose(results)
    if pose_key_points is not None:
        # 判断手腕是否接近脸部区域，相机角度问题导致手腕定位不准，降低该值权重
        is_near = get_hand_is_near_face(pose_key_points, frame.shape)
        # 根据身形定位脸部区域，dlib自带定位耗时，这个方法快且对脸的宽度定位准，长度定位相对较差
        # 如果运算速度足够，后期可先利用这个方法快速定位，然后再用dlib的方法进行精细定位
        face_area = get_face_area(pose_key_points, frame.shape)
        # 手腕接近脸部
        if face_area is not None:
            face_attr = get_area_face_attr(frame, face_area)
            # 调用模型探测手机和香烟
            detect_results = model(frame)
            # 转为列表
            detect_list = results_to_list(detect_results, model.names)
            # 脸部中心点
            face_center_point = (int((face_area[0] + face_area[2]) / 2), int((face_area[1] + face_area[3]) / 2))
            for r in detect_list:
                if r[4] > 30:
                    # 检测出的香烟或手机区域中心点
                    detect_center_point = (int((r[0] + r[2]) / 2), int((r[1] + r[3]) / 2))
                    # 计算两个中心点距离
                    length = dist.euclidean(face_center_point, detect_center_point)
                    if min_len == -1 or min_len > length:
                        min_len = length
                        min_type = r[5]
                    # if show_line:
                    if length > 40:
                        cv2.line(frame, face_center_point, detect_center_point, (0, 255, 11), 1)
                        cv2.putText(frame, r[5] + ":" + str(length), detect_center_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 222, 255), 1)

    return face_attr, is_near, min_len, min_type


def draw_pose(frame, results, draw_face=False, show_hand_near_face=False):
    """
    寻找最大的身形，并标注点位
    :param show_hand_near_face:
    :param draw_face:
    :param frame:
    :param results:
    :return:
    """
    pose_key_points = get_max_pose(results)
    if pose_key_points is not None:
        for i, k in enumerate(pose_key_points):
            p_c = get_point_and_conf(k, frame.shape)
            if p_c is not None:
                cv2.circle(frame, (int(p_c[0]), int(p_c[1])), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.putText(frame, str(i), (int(p_c[0]), int(p_c[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 222, 255), 1)
        if draw_face:
            frame = draw_face_area(frame, pose_key_points)
        if show_hand_near_face:
            frame = cv2.putText(frame, str(get_hand_is_near_face(pose_key_points, frame.shape)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 222, 255), 1)

    return frame


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


def get_hand_is_near_face(pose_key_points, shape=(640, 640)):
    # 0-10为上半身所有点
    points = []
    for point in range(11):
        min_conf = 0.5
        # 排除7、8两个点（手肘）
        if point != 7 and point != 8:
            if point == 9 and point == 10:
                min_conf = 0.3
            points.append(get_point_and_conf(pose_key_points[point], shape, min_conf))

    for hand in points[-2:]:
        for face_point in points[:-2]:
            if hand is not None and face_point is not None:
                length = dist.euclidean((int(hand[0]), int(hand[1])), (int(face_point[0]), int(face_point[1])))
                if length < 170:
                    return True
    return False


def get_face_area(pose_key_points, shape=(640, 640)):
    """
    获取人脸区域
    :param pose_key_points:
    :param shape:
    :return:
    """
    three_point_x = []
    three_point_y = []
    for point in [COCO_POINTS["nose"], COCO_POINTS["l_eye"], COCO_POINTS["r_eye"]]:
        p_c = get_point_and_conf(pose_key_points[point], shape)
        if p_c is not None:
            three_point_x.append(p_c[0])
            three_point_y.append(p_c[1])
        else:
            return None

    # 眼睛和鼻子高度为人脸的三分之一
    max_y = int(max(three_point_y))
    min_y = int(min(three_point_y))
    height = max_y - min_y
    # 两眼距离为人脸二分之一
    max_x = int(max(three_point_x))
    min_x = int(min(three_point_x))
    width = int((max_x - min_x) / 2)
    # 人脸近似一个圆形，如果高比值较小，则使用宽
    height = min(max(width, height), 2 * width)
    # 左上右下
    return min_x - width, min_y - height, max_x + width, max_y + 3 * height


def draw_face_area(frame, pose_key_points):
    """
    框出人脸区域
    :param frame:
    :param pose_key_points:
    :return:
    """
    face_area = get_face_area(pose_key_points, frame.shape)
    if face_area is not None:
        return cv2.rectangle(frame, (face_area[0], face_area[1]), (face_area[2], face_area[3]), (0, 121, 34), 1)
    else:
        return frame
