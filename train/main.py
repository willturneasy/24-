import json
import time

import paho.mqtt.client as mqtt

from pose import *
from video import VideoReader


def read_conf():
    conf = {}
    lines = open("base.conf", mode="r", encoding="utf-8").readlines()
    for line in lines:
        if line[0] == "#":
            continue
        else:
            line = line.replace(" ", "").replace("\n", "").replace("'", "").replace('"', '')
            line = line.split("=")
            if len(line) == 2:
                key = line[0]
                value = line[1]
                try:
                    value = int(value)
                except Exception as e:
                    pass
                conf[key] = value
    return conf


ALL_CONF = read_conf()


def get_output(attrs, pre_attr):
    face_attr, is_near, min_len, min_type = attrs
    result_list = ["", "", "", ""]
    type_map = {
        "phone": "usePhone",
        "smoke": "smoking"
    }

    if face_attr is not None:
        left_eye, right_eye, mouth, yaw_degree, pitch_degree, roll_degree = face_attr
        # 闭眼检测
        if int((left_eye + right_eye) / 2) < ALL_CONF["head_eye_close"]:
            result_list[0] = "eyeClose"
        elif left_eye == 255 and right_eye == 255 and pre_attr is not None:
            result_list[0] = pre_attr[0]

        # 长时间转头
        if 255 > abs(yaw_degree) >= ALL_CONF["head_yaw_degree"] or 255 > abs(pitch_degree) >= ALL_CONF["head_pitch_degree"]:
            result_list[1] = "poorAttention"
        elif yaw_degree == 255 and pitch_degree == 255 and pre_attr is not None:
            result_list[1] = pre_attr[1]

        # 打哈欠
        if mouth >= ALL_CONF["head_yawn"]:
            result_list[3] = "yawn"
        elif mouth == 255 and pre_attr is not None:
            result_list[3] = pre_attr[3]
    # 打电话抽烟
    if min_len != -1 and min_len < ALL_CONF["head_min_len"]:
        result_list[2] = type_map[min_type]

    if result_list[2] == "" and is_near:
        result_list[2] = "smoking"

    return result_list


def run():
    video = VideoReader('0')
    # 连续的相对精准的，如果有结果为None,就使用上一个结果
    pre_output = None
    output_list = []
    model_list = ["eyeClose", "yawn", "poorAttention", "smoking", "usePhone"]
    client = mqtt.Client()
    client.connect(ALL_CONF["mqtt_ip"], ALL_CONF["mqtt_port"], ALL_CONF["mqtt_keep_live"])  # 600为keepalive的时间间隔
    model_count = 5
    start_time = time.time()
    has_warned = False

    for frame in video:
        results = pose_model(frame)
        attrs = get_pose_attr(frame, results)
        d_output = get_output(attrs, pre_output)
        output_list.extend(d_output)
        pre_output = d_output
        print(d_output)
        for m in model_list:
            if output_list.count(m) > ALL_CONF["output_max_count"] and output_list[-ALL_CONF["output_last_fps"] * model_count:].count(m) >= ALL_CONF["output_last_fps"] - 1:
                # print("{}{}".format(m, str(time.time())))

                if time.time() - start_time > 1:
                    client.publish(ALL_CONF["mqtt_topic"], payload=json.dumps({
                        "timestamp": str(time.time()),
                        "dataType": m,
                    }), qos=0)
                    start_time = time.time()
                has_warned = True
                break
            else:
                if has_warned:
                    output_list = []
                    has_warned = False

        if len(output_list) > ALL_CONF["output_max_len"]:
            output_list = output_list[-ALL_CONF["output_max_len"]:]
        frame = cv2.addWeighted(frame, 0.5, results[0].plot(), 0.5, 1)
        cv2.imshow("tar", cv2.resize(frame, (1920, 1080)))
        cv2.waitKey(1)


if __name__ == '__main__':
    run()
