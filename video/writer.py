import datetime
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import cv2
import numpy as np

from utils import get_share_memory
from video.conf import VIDEO_WIDTH, VIDEO_HEIGHT, RTMP_BASE_PATH, VIDEO_FPS


class VideoWriter(object):

	def __init__(self, port, memory_name, out_path, save_or_send="save", width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=VIDEO_FPS, opencv_writer=False):
		self.port = port
		self.out_command = None
		self.video_read_thread_pool = ThreadPoolExecutor(max_workers=1)

		self.img_memory = get_share_memory(memory_name, width * height * 3)
		self.now_img = None
		self.height = height
		self.width = width
		self.fps = fps
		self.opencv_writer = opencv_writer
		self.out = None
		self.out_path = out_path
		self.save_or_send = save_or_send
		if self.save_or_send == "save":
			self.set_save_command()
		else:
			self.set_rtmp_command()
		self.img_queue = Queue()
		self.video_name = None

	def set_rtmp_command(self):
		self.out_command = [
			"ffmpeg",
			"-threads",
			"6",
			"-y",
			'-f', 'rawvideo',
			'-vcodec', 'rawvideo',
			'-pix_fmt', 'bgr24',
			"-r", str(self.fps),
			"-s", f"{self.width}x{self.height}",
			"-i", "-",
			"-c:v", "h264_nvmpi",
			"-pix_fmt", "yuv420p",
			'-preset', 'ultrafast',
			'-f', 'flv',
			f"{RTMP_BASE_PATH}{self.port}"
		]

	def set_save_command(self):
		video_name = f'video{self.port}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.mp4'
		video_name = os.path.join(self.out_path, video_name)
		self.video_name = video_name
		self.out_command = [
			"ffmpeg",
			"-threads",
			"6",
			"-y",
			'-f', 'rawvideo',
			'-vcodec', 'rawvideo',
			'-pix_fmt', 'bgr24',
			"-r", str(self.fps),
			"-s", f"{self.width}x{self.height}",
			"-i", "-",
			"-c:v", "h264_nvmpi",
			'-preset', 'ultrafast',
			"-pix_fmt", "yuv420p",
			# '-vcodec', 'libx264',
			video_name
		]

	def get_img(self):
		return np.ndarray(shape=(self.height, self.width, 3), dtype=np.uint8, buffer=self.img_memory.buf)

	def keep_get_img(self):
		while True:
			try:
				self.img_queue.put(self.get_img())
				time.sleep(1 / self.fps)
			except BaseException as e:
				if self.save_or_send == "save":
					print(f"通道{self.port} save queue error{e}")
				else:
					print(f"通道{self.port} send queue  error{e}")

	def set_command(self):
		if self.save_or_send == "save":
			self.set_save_command()
		else:
			self.set_rtmp_command()

	def init_output(self):
		self.set_command()
		cmd = subprocess.Popen(self.out_command, shell=False, stdin=subprocess.PIPE)
		self.out = cmd.stdin
		if self.save_or_send == "save" and self.opencv_writer:
			save_fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')  # 其中*'MP4V'和 'M', 'P', '4', 'V'等效
			self.out = cv2.VideoWriter(self.video_name.replace(".mp4",".avi"),
									   save_fourcc,
									   self.fps,
									   (int(self.width), int(self.height)))

	def run(self):
		self.video_read_thread_pool.submit(self.keep_get_img)
		time.sleep(10)
		self.init_output()
		while True:
			try:
				self.now_img = self.img_queue.get()
				if self.now_img is not None and self.now_img.shape[0] > 0:
					if self.save_or_send == "save":
						if self.opencv_writer:
							self.out.write(self.now_img)
						else:
							self.out.write(self.now_img.tobytes())
					else:
						self.out.write(self.now_img.tobytes())

			except BaseException as e:
				if self.save_or_send == "save":
					print(f"通道{self.port} save error{e}")
				else:
					print(f"通道{self.port} send error{e}")
				self.init_output()


def run(port, memory_name, out_path, save_or_send,opencv_writer=False):
	writer = VideoWriter(port, memory_name, out_path, save_or_send,opencv_writer=opencv_writer)
	writer.run()
