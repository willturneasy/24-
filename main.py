import datetime
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
from queue import Queue

import cv2
import numpy as np
from ultralytics import YOLO

import can.receiveAndSend
import model.pose
import mqtt.mqtt
import rtk.receive
import rtk.rtkSend
import udp.send
import upload.conf
import video.reader_test
import video.writer
from log.log import Log
from mqtt.conf import LOCAL_MQTT, SERVER_MQTT
from upload.upload import upload_event_info, upload_event_video
from utils import create_share_memory, out_video
from video.conf import VIDEO_HEIGHT, VIDEO_WIDTH, PATH_TO_SAVE, VIDEO_FPS

main_log = Log("logs", "main.out")


class StartPreset(object):

	def __init__(self, path_to_save=PATH_TO_SAVE):
		self.path_to_save = path_to_save
		self.delete_workers = 3
		self.path_can_save = False
		self.delete_thread_pool = ThreadPoolExecutor(max_workers=self.delete_workers)
		# 文件夹
		self.run_save_path = None
		self.run_save_path_temp = None
		self.run_save_path_input = None
		self.run_save_path_output = None
		self.run_save_path_logs = None
		# 清除过期文件
		self.clear_out_time()
		if self.path_can_save:
			# 新的保存路径
			self.create_new_dir()

	def clear_out_time(self):
		# 设置了存储路径
		if self.path_to_save is not None:
			# 存储路径不存在即填写了错误的路径，不保存，（！！不要新建错误路径，视频存储太大，如果存储在系统空间，会导致系统无法开机）
			if not os.path.exists(self.path_to_save):
				self.path_can_save = False
				pass
			else:
				# 路径存在，则开始存储清理
				self.path_can_save = True
				# 列出文件
				dir_list = os.listdir(self.path_to_save)
				# 找出文件夹
				dirs = [f for f in dir_list if os.path.isdir(os.path.join(self.path_to_save, f))]
				# 遍历文件节
				for d in dirs:
					try:
						# 如果文件节为%Y-%m-%d标准命名，则进行日期比较
						d_data = datetime.datetime.strptime(d, "%Y-%m-%d")
						rm_f = os.path.join(self.path_to_save, d)
						# 日期大于31天的全部删除
						if (datetime.datetime.now() - d_data).days > 31:
							shutil.rmtree(rm_f)
						else:
							rm_f_list = os.listdir(rm_f)
							for rm_dir in rm_f_list:
								self.delete_thread_pool.submit(shutil.rmtree, os.path.join(rm_f, rm_dir, "temp"))
								self.delete_thread_pool.submit(shutil.rmtree, os.path.join(rm_f, rm_dir, "input"))
								self.delete_thread_pool.submit(shutil.rmtree, os.path.join(rm_f, rm_dir, "output"))
					except:
						pass

	def create_new_dir(self):
		try:
			while time.time() < 1720676559:
				pass
			date = datetime.datetime.now().strftime("%Y-%m-%d")
			# 今日保存路径
			now_to_save = os.path.join(self.path_to_save, date)
			# 如果今日没还没有文件夹则创建
			if not os.path.exists(now_to_save):
				os.mkdir(now_to_save)
			# 计算该目录下文件夹数量，确定第几次启动
			run_count = len(os.listdir(now_to_save))
			# 本次运行保存路径
			self.run_save_path = os.path.join(now_to_save, str(run_count))
			os.mkdir(self.run_save_path)
			# temp文件
			self.run_save_path_temp = os.path.join(self.run_save_path, "temp")
			os.mkdir(self.run_save_path_temp)
			# 图片存放位置
			self.run_save_path_input = os.path.join(self.run_save_path, "input")
			os.mkdir(self.run_save_path_input)
			# 短视频存放位置
			self.run_save_path_output = os.path.join(self.run_save_path, "output")
			os.mkdir(self.run_save_path_output)
			# 短视频存放位置
			self.run_save_path_logs = os.path.join(self.run_save_path, "logs")
			os.mkdir(self.run_save_path_logs)
		except:
			pass

	def can_save(self):
		return self.path_can_save

	def get_save_path(self):
		if self.path_can_save:
			return self.run_save_path
		else:
			return None

	def get_temp_path(self):
		if self.path_can_save:
			return self.run_save_path_temp
		else:
			return None

	def get_output_path(self):
		if self.path_can_save:
			return self.run_save_path_output
		else:
			return None

	def get_input_path(self):
		if self.path_can_save:
			return self.run_save_path_input
		else:
			return None

	def get_logs_path(self):
		if self.path_can_save:
			return self.run_save_path_logs
		else:
			return None


class DMS(object):
	def __init__(self):
		self.pose_model = None
		self.face_model = None
		self.hand_model = None
		self.eye_model = None
		# self.video_upload_thread_pool = None
		self.video_read_thread_pool = None
		self.type_list = ["eyeClose", "yawn", "poorAttention", "smoking", "usePhone"]

		self.start_preset = StartPreset()
		self.process_data_manager = Manager()
		self.process_task_list = []
		self.process_pool = None
		self.share_memory_face = None
		self.gps_dict = None
		self.udp_dict = None
		self.control_dict = None
		self.danger_result_list = []
		self.danger_index = 0
		self.upload_img_queue = None
		self.local_message_queue = None
		self.now_face_img = None
		self.pre_time = time.time() - 20
		self.wait_time = 20
		self.result1=None
		self.Warning_Messages=None

	def init_gps_dict(self):
		"""
		初始化gps字典，后续将字典传递给rtk线程
		:return:
		"""
		self.gps_dict = self.process_data_manager.dict()
		self.gps_dict["lng"] = 118.826245
		self.gps_dict["lat"] = 31.87095
		self.gps_dict["angle"] = 32
		self.gps_dict["speed"] = 0

	def init_udp_dict(self):
		"""
		初始化udp字典，后续将字典传递给udp线程
		:return:
		"""
		self.udp_dict = self.process_data_manager.dict()
		self.udp_dict["ip"] = None
		self.udp_dict["port"] = 60021
		self.udp_dict["start_flag"] = False
		self.udp_dict["channel1"] = False
		self.udp_dict["channel2"] = False
		self.udp_dict["channel3"] = False
		self.udp_dict["channel4"] = False
		self.udp_dict["channel5"] = False
		self.udp_dict["timestamp"] = None


	def init_control_dict(self):
		"""
		初始化udp字典，后续将字典传递给udp线程
		:return:
		"""
		self.control_dict = self.process_data_manager.dict()
		self.control_dict["need_control"] = False
		self.control_dict["content"] = "123"
		self.control_dict["timestamp"] = None


	def init_upload(self, _fps=VIDEO_FPS, _save_len=10):
		"""
		上传事件
		:param _fps:
		:param _save_len:
		:return:
		"""
		# 上传事件的线程池，一个线程，顺序
		# self.video_upload_thread_pool = ThreadPoolExecutor(max_workers=1)
		# 上传事件图片保存队列
		self.upload_img_queue = Queue(maxsize=_save_len * _fps + 10)

	def init_model(self):
		"""
		初始化模型
		:return:
		"""
		self.pose_model = YOLO("models/pose.pt", task="pose")
		self.face_model = YOLO("models/face.pt", task="pose")
		self.hand_model = YOLO("models/hand.pt")
		self.eye_model = YOLO("models/eyes.pt", task="classify")
		self.video_read_thread_pool = ThreadPoolExecutor(max_workers=1)

	def init_danger_result_list(self, _fps_count=VIDEO_FPS * 2):
		"""
		初始化危险计数
		:param _fps_count:
		:return:
		"""
		self.danger_index = 0
		self.danger_result_list = []
		for _i in range(len(self.type_list)):
			self.danger_result_list.append([""] * _fps_count)

	def set_danger_result_list_value(self, _result, _fps_count=VIDEO_FPS * 2):
		"""
		在列表内循环设值
		:param _result:
		:param _fps_count:
		:return:
		"""
		for _i in range(len(self.type_list)):
			self.danger_result_list[_i][self.danger_index] = _result[_i]
		self.danger_index = (self.danger_index + 1) % _fps_count

	def is_need_wait(self):
		"""
		危险事件上传间隔
		:return:
		"""
		return time.time() - self.pre_time < self.wait_time

	def set_pre_time(self):
		"""
		上一次上报时间
		:return:
		"""
		self.pre_time = time.time()

	def put_img_to_upload_queue(self):
		"""
		实时将图片塞入上传的保存队列
		:return:
		"""
		try:
			_img = self.now_face_img.copy()
			if self.upload_img_queue.full():
				self.upload_img_queue.get_nowait()
			self.upload_img_queue.put_nowait(self.draw_danger_pre(_img))
		except BaseException:
			pass

	def get_is_danger(self, _draw_flag=True):
		"""
		判断是否危险
		:param _draw_flag:
		:return:
		"""
		warning_messages = {
			"eyeClose": "您已疲劳驾驶，请注意行车安全",
			"yawn": "您已疲劳驾驶，请注意行车安全",
			"poorAttention": "您已分心驾驶，请注意行车安全",
			"smoking": "您已危险驾驶，请注意行车安全",
			"usePhone": "您已危险驾驶，请注意行车安全"
		}

		_danger_pre_list = self.get_danger_per_list()
		_max_danger_pre = max(_danger_pre_list)
		if _max_danger_pre > 80:
			#获取危险值>80的事件作为mqtt传输的context
			self.result1=self.type_list[_danger_pre_list.index(_max_danger_pre)]
			self.Warning_Messages=warning_messages.get(self.result1)
			return True, self.type_list[_danger_pre_list.index(_max_danger_pre)]
		else:
			return False, ""
	def get_danger_per_list(self):
		"""
		计算每个场景的危险数值
		:return:
		"""
		_danger_pre_list = []
		for _i, _item in enumerate(self.type_list):
			_danger_pre_list.append(int(100 - (self.danger_result_list[_i].count("") * 100 / len(self.danger_result_list[_i]))))

		return _danger_pre_list

	def draw_danger_pre(self, _img):
		"""
		在画面上写入危险数值
		:param _img:
		:return:
		"""
		for _i, _pre in enumerate(self.get_danger_per_list()):
			_item = self.type_list[_i]
			_img = cv2.putText(_img, f"{_item}:{_pre}%", (10, 30 * (_i + 1)), cv2.FONT_HERSHEY_PLAIN, 2,
							   (255, 255, 255) if _pre < 50 else ((0, 0, int(255 * (_pre / 80))) if _pre < 70 else (0, 0, 255)), 1)
		return _img

	def get_gps_data(self):
		"""
		获取gps数据
		:return:
		"""
		return self.gps_dict.get("lng"), self.gps_dict.get("lat"), self.gps_dict.get("angle"), self.gps_dict.get("speed")

	def get_face_img(self, width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
		"""
		实时从共享内存中获取图像
		:param width:
		:param height:
		:return:
		"""
		self.now_face_img = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=self.share_memory_face.buf).copy()
		self.put_img_to_upload_queue()

	def add_camera_task(self, need_save=True, need_send=True, width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
		"""
		添加相机任务
		:param need_save:
		:param need_send:
		:param width:
		:param height:
		:return:
		"""
		# 相机进程
		for _port in [
			video.conf.video_port100,
			video.conf.video_port102,
			video.conf.video_port104,
			video.conf.video_port106,
			video.conf.video_port108,
		]:
			# 保存队列
			save_img_queue = self.process_data_manager.Queue()
			# 实时图像共享内存
			share_img_memory = create_share_memory(f"video{_port}-{width}X{height}", width * height * 3)

			if _port == video.conf.video_port100:
				self.share_memory_face = share_img_memory

			# 读取进程
			self.process_task_list.append([
				video.reader_test.run, (_port, share_img_memory.name), f"摄像头{_port}"
			])
			# 保存及发送进程
			if self.start_preset.can_save():
				if need_save:
					self.process_task_list.append([
						# video.save.run, (_port, save_img_queue, self.start_preset.get_save_path(), self.start_preset.get_input_path()), f"摄像头{_port}视频保存"
						video.writer.run, (_port, share_img_memory.name, self.start_preset.get_save_path(), "save", False), f"摄像头{_port}视频保存"
					])
				if need_send:
					self.process_task_list.append([
						# video.save.run, (_port, save_img_queue, self.start_preset.get_save_path(), self.start_preset.get_input_path()), f"摄像头{_port}视频保存"
						video.writer.run, (_port, share_img_memory.name, "", "send"), f"摄像头{_port}视频保存"

					])

	def add_mqtt_task(self, _mqtt_desc, _mqtt_config, _log_file_name, _log_path="logs", show_data=None):
		"""
		添加mqtt任务
		:param show_data:
		:param _mqtt_desc:
		:param _mqtt_config:
		:param _log_file_name:
		:param _log_path:
		:return:
		"""
		publish_message_queue = self.process_data_manager.Queue()
		self.process_task_list.append([
			mqtt.mqtt.run, (_mqtt_config, _log_path, _log_file_name, publish_message_queue, show_data), _mqtt_desc
		])
		return publish_message_queue

	def add_rtk_task(self, mqtt_message_queue):
		"""
		添加rtk任务
		:param mqtt_message_queue:
		:return:
		"""
		self.process_task_list.append([rtk.rtkSend.run, (), "rtk发送"])
		self.process_task_list.append([rtk.receive.run, (mqtt_message_queue, self.gps_dict, self.start_preset.get_logs_path()), "rtk接收"])

	def add_udp_task(self):
		self.process_task_list.append([udp.send.run, (self.udp_dict,), "udp发送"])

	def add_can_task(self,queue=None):
		self.process_task_list.append([can.receiveAndSend.run,(queue,self.start_preset.get_logs_path()),"can数据解析及发送"])

	def start_process(self):
		"""
		开始进程
		:return:
		"""
		# 进程池数量一定要大于要运行的进程数量，保证error_callback能够重新调用进程，这一步是进程自动重启的保证
		_process_num = len(self.process_task_list) + 2
		self.process_pool = multiprocessing.Pool(processes=_process_num)
		for _p in self.process_task_list:
			self.process_pool.apply_async(_p[0], args=_p[1], error_callback=get_error_callback(self.process_pool, _p))

	def handle_img(self):
		"""
		调用模型处理图片
		:return:
		"""
		_lng, _lat, _angle, _speed = self.get_gps_data()
		try:
			_result = model.pose.handle_img(self.now_face_img, self.pose_model, self.hand_model, self.face_model, self.eye_model, _speed)
			self.set_danger_result_list_value(_result)
		except BaseException:
			pass

	def upload_handle(self, _danger_type, fps=VIDEO_FPS, debug=False):
		"""
		上传处理
		:param _danger_type:
		:param fps:
		:param debug:
		:return:
		"""
		_timestamp = int(time.time() * 1000)
		_lng, _lat, _angle, _speed = self.get_gps_data()
		_event_id = f'{upload.conf.CAR_ID}_{upload.conf.CHANNEL}_{_timestamp}'
		try:
			if not debug:
				_upload_flag = upload_event_info(_danger_type, _lng, _lat, _angle, _speed, self.now_face_img, _event_id, _timestamp)
			else:
				_upload_flag = True
				cv2.imwrite(os.path.join(self.start_preset.get_output_path(), f"{_event_id}.png"), self.now_face_img)
			if _upload_flag:
				_upload_img_list = []

				_save_count = 0
				while _save_count < 2 * fps * 10:
					_upload_img_list.append(self.upload_img_queue.get())
					_save_count += 1

				u_video_path = out_video(_upload_img_list, self.start_preset.get_output_path())
				if not debug:
					upload_event_video(u_video_path, _event_id)
					os.remove(u_video_path)

		except BaseException:
			pass

	def show_face_img(self):
		"""
		展示图片
		:return:
		"""
		#self.handle_img()
		cv2.imshow("face", self.draw_danger_pre(self.now_face_img))
		cv2.waitKey(1)

	def keep_get_img(self):
		"""
		根据时间间隔实时更新图片
		:return:
		"""
		while True:
			self.get_face_img()
			time.sleep(1 / VIDEO_FPS)
			self.show_face_img()

	def run(self):
		"""
		运行
		:return:
		"""
		self.video_read_thread_pool.submit(self.keep_get_img)
		time.sleep(1)
		while True:
			try:
				if self.local_message_queue is not None:
					if self.control_dict["need_control"]:
						self.local_message_queue.put({
							"data": {
								"timestamp": int(time.time() * 1000),
								"context": self.control_dict["content"],
								"flag":1
							},
							"topic": "auto/drive/scene/control"
						})
						self.control_dict["need_control"] = False
						self.control_dict["content"]=""
					if self.Warning_Messages:	
						self.local_message_queue.put({
							"data": {
								"timestamp": int(time.time() * 1000),
								"context":self.Warning_Messages,
								"flag":2
			 				},
							"topic": "auto/drive/scene/control"
						})
						self.control_dict["need_control"] = False
						self.Warning_Messages = ""
			except:
				pass
			if not self.is_need_wait():
				self.handle_img()
				_danger_flag, _type = self.get_is_danger()
				if _danger_flag:
					self.set_pre_time()
					#self.upload_handle(_type, debug=False)
					self.init_danger_result_list()

	def start(self):
		"""
		启动
		:return:
		"""
		# 初始化gps字典
		self.init_gps_dict()
		# 初始化udp字典
		self.init_udp_dict()
		# 初始化control字典
		self.init_control_dict()
		# 初始化模型
		self.init_model()
		# 初始化记录列表
		self.init_danger_result_list()
		# 初始化上传
		self.init_upload()
		# 添加相机进程
		#self.add_camera_task(need_save=False, need_send=False)
		self.add_camera_task()
		# 添加mqtt进程
		server_message_queue = self.add_mqtt_task("平台对接mqtt", SERVER_MQTT, "mqtt_server.out",show_data=self.control_dict)
		self.local_message_queue = self.add_mqtt_task("本地mqtt", LOCAL_MQTT, "mqtt_local.out", show_data=self.udp_dict)
		# 添加rtk进程
		self.add_rtk_task(server_message_queue)
		self.add_rtk_task(self.local_message_queue)
		self.add_udp_task()
		self.add_can_task(server_message_queue)
		# 启动进程
		self.start_process()
		self.run()
		#while True:
		# 	if self.local_message_queue is not None:
		# 		self.local_message_queue.put({
		# 				"data": {
		# 					"timestamp": int(time.time() * 1000),
		# 					"context":self.result1
		# 				},
		# 				"topic": "auto/drive/scene/control"
		# 			})
		# 		self.control_dict["need_control"] = False
		# 		self.result1 = ""
		# 	pass
		#print(self.udp_dict)


def get_error_callback(_pool, pro):
	def restart(_e):
		main_log.log(f'{pro[2]} error:{_e}')
		_pool.apply_async(pro[0], args=pro[1], error_callback=get_error_callback(_pool, pro))

	return restart


if __name__ == '__main__':
	dms = DMS()
	dms.start()
