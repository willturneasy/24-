import subprocess
import sys
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

from .conf import *


class VideoReaderTest(object):
	def __init__(self, port, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=VIDEO_FPS, memory_name=None, ffmpeg_read=False):
		self.port = port
		self.width = width
		self.height = height
		self.fps = float(fps)
		self.memory_name = None
		self.share_img = None
		if memory_name is not None:
			self.memory_name = memory_name

		self.int_port = False

		try:
			self.port = int(self.port)
			self.int_port = True
		except ValueError:
			pass
		self.cap = None
		self.ffmpeg_read = ffmpeg_read

		self.pre_time = time.time()
		self.img_count = 0

	def get_cap(self):
		if not self.ffmpeg_read:
			if self.int_port:
				if "win" in sys.platform:
					self.cap = cv2.VideoCapture(self.port, cv2.CAP_DSHOW)

				elif "linux" in sys.platform:
					self.cap = cv2.VideoCapture(self.port, apiPreference=cv2.CAP_V4L2)
			else:
				self.cap = cv2.VideoCapture(self.port)

			if self.cap and self.cap.isOpened():
				pass
				if self.int_port:
					self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
					self.cap.set(cv2.CAP_PROP_FPS, self.fps)
					self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
					self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
					self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
			else:
				raise IOError('Video {} cannot be opened'.format(self.port))
		else:
			command = ['ffmpeg',
					   '-f', 'v4l2',
					   '-input_format', 'mjpeg',
					   '-framerate', '30',
					   '-video_size', '1280x720',
					   '-i', f'/dev/video{self.port}',
					   '-f', 'image2pipe',
					   '-pix_fmt', 'bgr24',
					   '-vcodec', 'rawvideo', '-']
			self.cap = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=self.width * self.height * 3 + 100).stdout

	def set_share_img(self):
		if self.memory_name is not None:
			self.share_img = shared_memory.SharedMemory(name=self.memory_name, create=False)

	def set_share_img_value(self, frame):
		if self.share_img is not None:
			self.share_img.buf[:self.width * self.height * 3] = frame.tobytes()

	def get_img(self):
		if not self.ffmpeg_read:
			ret, frame = self.cap.read()
		else:
			ret = self.cap.readable()
			frame = np.ndarray(shape=(self.height, self.width, 3), dtype=np.uint8, buffer=self.cap.read(self.width * self.height * 3))
		return ret, frame

	def read_count(self):
		self.img_count += 1
		if time.time() - self.pre_time > 1:
			print(self.img_count)
			self.img_count = 0
			self.pre_time = time.time()

	def read_img(self, need_count=True):
		self.get_cap()
		self.set_share_img()
		while True:
			ret, frame = self.get_img()
			if need_count:
				self.read_count()
			if ret:
				self.set_share_img_value(frame)
			else:
				break
		raise IOError('Video {} stop'.format(self.port))


def run(port, memory_name=None):
	# 睡眠一秒，防止上次出错后相机资源未释放
	time.sleep(1)
	video = VideoReaderTest(port, memory_name=memory_name)
	video.read_img()
