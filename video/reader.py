import datetime
import os
import shutil
import threading
import time
import uuid
from multiprocessing import shared_memory

import cv2

from .conf import *


class VideoReader(object):
	def __init__(self, port, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, path_to_save=PATH_TO_SAVE, save_len=1, fps=VIDEO_FPS,
				 memory_name=None,manager_list=None):
		"""
		初始化
		:param port: 摄像机对应的端口：ls /dev 下video的名称
		:param width: 视频宽
		:param height: 视频高
		:param path_to_save: 保存的位置
		:param save_len: 保存的时长（不建议过长，现版本为硬盘读写,内存读写快一点，但是容易爆）
		:param fps: 视频帧数
		:param memory_name: 共享内存名称（现版本采用的是多进程，进程之间通过共享内存方式传输数据，保证了摄像头有实时性）
		"""
		self.frames = []
		self.start_save_time = None
		self.port = port
		self.path_to_save = None
		self.now_to_save = None
		self.now_to_save_temp = None
		self.save_len = save_len
		self.width = width
		self.height = height
		self.fps = int(fps)
		self.memory_name = None
		self.path_can_save = False
		self.run_count = -1
		if memory_name is not None:
			self.memory_name = memory_name
		if path_to_save is not None:
			self.path_to_save = path_to_save
			self.clear_out_time()
		try:
			self.port = int(self.port)
		except ValueError:
			pass
		self.cap = cv2.VideoCapture(self.port, apiPreference=cv2.CAP_V4L2)
		# self.cap = cv2.VideoCapture(self.port,cv2.CAP_DSHOW)
		if self.cap.isOpened():
			pass
			self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
			self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
			self.cap.set(cv2.CAP_PROP_FPS, self.fps)
			self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
		else:
			raise IOError('Video {} cannot be opened'.format(self.port))

	def clear_out_time(self):
		if self.path_to_save is not None:
			if not os.path.exists(self.path_to_save):
				self.path_can_save = False
				# os.makedirs(self.path_to_save)
				pass
			else:
				self.path_can_save = True

				dir_list = os.listdir(self.path_to_save)
				dirs = [f for f in dir_list if os.path.isdir(os.path.join(self.path_to_save, f))]
				for d in dirs:
					try:
						d_data = datetime.datetime.strptime(d, "%Y-%m-%d")
						if (datetime.datetime.now() - d_data).days > 31:
							rm_f = os.path.join(self.path_to_save, d)
							shutil.rmtree(rm_f)
					except:
						pass

				try:
					date = datetime.datetime.now().strftime("%Y-%m-%d")
					now_to_save = os.path.join(self.path_to_save, date)
					if not os.path.exists(now_to_save):
						os.mkdir(now_to_save)

					if self.run_count == -1:
						self.run_count = len(os.listdir(now_to_save))

					now_to_save = os.path.join(now_to_save, str(self.run_count))
					now_to_save_temp = os.path.join(now_to_save, "temp")

					if not os.path.exists(now_to_save):
						os.mkdir(now_to_save)

					if now_to_save != self.now_to_save:
						self.now_to_save = now_to_save
						self.now_to_save_temp = now_to_save_temp
						if not os.path.exists(now_to_save_temp):
							os.mkdir(now_to_save_temp)

				except:
					pass

	def save(self, img):
		if self.path_can_save:
			if self.now_to_save_temp is not None:
				name = os.path.join(self.now_to_save_temp, f"{uuid.uuid4()}.jpg")
				cv2.imwrite(name, img)
				self.frames.append(name)
				if self.start_save_time is None:
					self.start_save_time = time.time()
				elif time.time() - self.start_save_time > 60 * self.save_len:
					threading.Thread(target=self.save_video, args=(self.frames, self.start_save_time)).start()
					self.frames = []
					self.start_save_time = time.time()
					self.clear_out_time()

	def save_video(self, frames, save_time):
		frame_len = len(frames)
		out_frames = []
		if frame_len > 0:
			need_frames = self.fps * self.save_len * 60
			if need_frames > frame_len:
				add_m = need_frames / frame_len
				for index, f in enumerate(frames):
					for i in range(int(add_m * index), int(add_m * (index + 1))):
						out_frames.append(f)
			else:
				out_frames = frames

			name = os.path.join(self.now_to_save, f'video{self.port}-{time.strftime("%H-%M-%S", time.localtime(save_time))}.temp')
			with open(f'{name}', 'w') as out_file:
				out_file.writelines("\n@".join(out_frames).split("@"))

	def read_img(self):
		share_img = None
		# check_time = None
		if self.memory_name is not None:
			share_img = shared_memory.SharedMemory(name=self.memory_name, create=False)
		# check_time = shared_memory.ShareableList(name="check")
		error_count = 0
		while True:
			ret, frame = self.cap.read()
			if ret:
				error_count = 0
				if share_img is not None:
					share_img.buf[:] = frame.tobytes()
				# if check_time is not None:
				#     if self.port == 100:
				#         check_time[0] = time.time()
				#     elif self.port == 102:
				#         check_time[1] = time.time()
				self.save(frame)
			else:
				error_count += 1
				if error_count > 100:
					self.cap.release()
					break
		raise IOError('Video {} stop'.format(self.port))


def run(port, memory_name=None):
	video = VideoReader(port, memory_name=memory_name)
	video.read_img()
