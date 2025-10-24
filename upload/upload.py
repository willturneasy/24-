import json
import time

import cv2
import requests

from log.log import Log
from .conf import HEADER, BASE_PATH, EVENT_INFO, CAR_ID, CHANNEL, EVENT_VIDEO
from .utils import img_to_base64, get_md5


def upload_event_info(data_type,lng, lat, angle, speed, target_img,event_id,timestamp):
	upload_log = Log("logs", "upload.out")
	# 根据信息构建上传数据
	post_data = {
		"angle": angle,
		"cameraChannel": CHANNEL,
		"carId": CAR_ID,
		"dataType": data_type,
		"image": img_to_base64(target_img),
		"lng": lng,
		"lat": lat,
		"speed": speed,
		"timestamp": timestamp,
		"token": get_md5(timestamp),
		"eventId": event_id,
	}
	post_json = json.dumps(post_data)
	upload_log.log(f'开始上传事件信息：{event_id}->angle: {angle},cameraChannel: {CHANNEL},carId: {CAR_ID},dataType: {data_type},lng: {lng},lat: {lat},speed: {speed}')
	event_res = requests.post(BASE_PATH + EVENT_INFO, data=post_json, headers=HEADER)
	upload_log.log(f'上传事件信息返回值：{event_res.text}')
	event_res = eval(event_res.text.replace("null", "None").replace("true", "True"))
	# 事件上传成功后开始处理视频
	if event_res["success"]:
		return True
	else:
		return False


def upload_event_video(video_name,event_id):
	upload_log = Log("logs", "upload.out")
	upload_log.log(f'开始上传事件视频：{event_id}->{video_name}')
	timestamp = int(time.time() * 1000)
	# 上传视频
	get_data = {
		"eventId": event_id,
		"timestamp": timestamp,
		"token": get_md5(timestamp),
	}
	res = requests.post(BASE_PATH + EVENT_VIDEO, data=get_data, files={'file': ("file.mp4", open(video_name, 'rb'))})
	upload_log.log(f'上传事件视频返回值：{res.text}')
