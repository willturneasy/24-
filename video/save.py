import datetime
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Empty

import cv2
import numpy as np

from .conf import *


class VideoSave(object):
	def __init__(self, port, manager_queue,run_save_path,run_save_path_input):
		self.run_save_path_input = run_save_path_input
		self.run_save_path = run_save_path
		self.port = port
		self.save_count = 0
		self.rtsp_count = 0
		self.img_count = 0
		self.manager_queue = manager_queue
		self.img_workers = 6
		self.video_workers = 1
		self.img_thread_pool = ThreadPoolExecutor(max_workers=self.img_workers)
		self.video_thread_pool = ThreadPoolExecutor(max_workers=self.video_workers)
		self.rtsp_thread_pool = ThreadPoolExecutor(max_workers=self.video_workers)
		self.out = None
		self.rtsp = None
		self.out_command = None
		self.set_out_command()

		self.rtsp_command = [
			"ffmpeg",
			"-threads",
			"6",
			"-y",
			'-f', 'rawvideo',
			'-vcodec', 'rawvideo',
			'-pix_fmt', 'bgr24',
			# "-r", str(VIDEO_FPS),
			"-r", str(15),
			"-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
			"-i", "-",
			"-c:v", "h264_nvmpi",
			"-pix_fmt", "yuv420p",
			'-preset', 'ultrafast',
			# '-g',str(VIDEO_FPS * 2),
			'-f', 'flv',
			# '-max_delay', '5000', '-bufsize', '500000', '-rtbufsize', '500000',
			f"{RTMP_BASE_PATH}{self.port}"
		]

	def set_out_command(self):
		video_name = f'video{self.port}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.mp4'
		video_name = os.path.join(self.run_save_path, video_name)
		self.out_command = [
			"ffmpeg",
			"-threads",
			"6",
			"-y",
			'-f', 'rawvideo',
			'-vcodec', 'rawvideo',
			'-pix_fmt', 'bgr24',
			# "-r", str(VIDEO_FPS),
			"-r", str(15),
			"-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
			"-i", "-",
			"-c:v", "h264_nvmpi",
			'-preset', 'ultrafast',
			# "-pix_fmt", "yuv420p",
			# '-vcodec', 'libx264',
			# '-f','flv',
			video_name
		]

	def save_video(self, frame):
		if frame is not None and frame.shape[0] > 0:
			self.save_count += 1
			if self.img_count - self.save_count > 100:
				print(f"通道{self.port}缓存中图片数量过多！{self.img_count - self.save_count}")
			try:
				self.out.write(frame.tobytes())
			except BaseException as e:
				print(f"通道{self.port}save error{e}")
				self.set_out_command()
				cmd = subprocess.Popen(self.out_command, shell=False, stdin=subprocess.PIPE)
				self.out = cmd.stdin

	def send_video(self, frame):
		if frame is not None and frame.shape[0] > 0:
			self.rtsp_count += 1
			if self.img_count - self.rtsp_count > 100:
				print(f"通道{self.port}缓存中rtsp图片数量过多！{self.img_count - self.rtsp_count}")
			try:
				self.rtsp.write(frame.tobytes())
			except BaseException as e:
				print(f"通道{self.port}rtsp error{e}")
				cmd2 = subprocess.Popen(self.rtsp_command, shell=False, stdin=subprocess.PIPE)
				self.rtsp = cmd2.stdin

	def save(self):
		cmd = subprocess.Popen(self.out_command, shell=False, stdin=subprocess.PIPE)
		self.out = cmd.stdin
		cmd2 = subprocess.Popen(self.rtsp_command, shell=False, stdin=subprocess.PIPE)
		self.rtsp = cmd2.stdin
		pre_frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), np.uint8)
		while True:
			try:
				# 从进程管理的队列里取图片
				frame = self.manager_queue.get(timeout=1)
				pre_frame = frame
			except Empty as e:
				# print(f"通道{self.port}保存队列为空 {e}")
				pass

				frame = pre_frame
			# 存放在图片的指定位置
			# name = os.path.join(self.run_save_path_input, f"video{self.port}-{format_number(self.img_count)}.jpg")
			# 使用线程池去保存图片
			# self.img_thread_pool.submit(cv2.imwrite, name, frame)
			# 保存图片名称
			# frames.append(name)
			self.img_count += 1
			self.video_thread_pool.submit(self.save_video, frame)
			self.rtsp_thread_pool.submit(self.send_video, frame)






def run(port, manager_queue,run_save_path,run_save_path_input):
	saver = VideoSave(port=port, manager_queue=manager_queue,run_save_path=run_save_path,run_save_path_input=run_save_path_input)
	saver.save()
