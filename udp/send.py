import socket
import time
import logging
import cv2
import numpy as np

import video
from utils import get_share_memory
from video.conf import VIDEO_WIDTH, VIDEO_HEIGHT


class UDPImg(object):
	def __init__(self, config_dict, width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
		self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.send_img = np.full((256, 192, 3), (255, 255, 255), np.uint8)
		self.frame_index = 1
		self.config_dict = config_dict
		self.height = height
		self.width = width
		self.img_memory0 = get_share_memory(f"video{video.conf.video_port100}-{width}X{height}", width * height * 3)
		self.img_memory2 = get_share_memory(f"video{video.conf.video_port102}-{width}X{height}", width * height * 3)
		self.img_memory4 = get_share_memory(f"video{video.conf.video_port104}-{width}X{height}", width * height * 3)
		self.img_memory6 = get_share_memory(f"video{video.conf.video_port106}-{width}X{height}", width * height * 3)
		self.img_memory8 = get_share_memory(f"video{video.conf.video_port108}-{width}X{height}", width * height * 3)

		logging.basicConfig(filename='your_log_file.log', level=logging.INFO,
		 				format='%(asctime)s - %(levelname)s - %(message)s')
		self.logger = logging.getLogger()

	def img_to_udp_data(self, send_img, byteorder="little"):
		# send_img = cv2.resize(send_img, (256 * 1, 192 * 1))
		# send_img = cv2.resize(send_img, (1024, 768))
		data_list = send_img.tobytes()
		height, width = send_img.shape[:2]
		pack_nums = int(len(data_list) / 60000) + 1

		for i in range(pack_nums):
			data = data_list[i * 60000:(i + 1) * 60000 if (i + 1) * 60000< len(data_list) else len(data_list)]
			useful_num = len(data)
			if len(data) < 60000:
				data = data + int(0).to_bytes(60000- len(data), byteorder)
			b_list = [
				int(518).to_bytes(2, byteorder),  # unsigned short infoID 		2	//消息ID，固定值0x0206
				int(self.frame_index).to_bytes(4, byteorder),  # int frameID		4	//图像帧号(从0开始)
				int(256).to_bytes(256, byteorder),  # 256	//文件名
				int(width).to_bytes(4, byteorder),  # int width		4	//图像宽度
				int(height).to_bytes(4, byteorder),  # int height		4	//图像高度
				int(24).to_bytes(4, byteorder),  # int nBits		4	//1-二值图像; 8-灰度图像; 24-彩色图像（默认）)
				int(pack_nums).to_bytes(4, byteorder),  # int packNums		4	//分包总数
				int(i + 1).to_bytes(4, byteorder),  # int packNums		4	//分包总数
				int(useful_num).to_bytes(4, byteorder),  # int valuedBytes		4	//数组中有效字节个数
				data,  # unsigned char pData[60000];		60000	//图像数据
				int(60291).to_bytes(4, byteorder),  # int nSize		4	//该结构体的大小
				int(1).to_bytes(1, byteorder)  # unsigned char checksum		1	//检查和:以上数据之和
			]
			yield b''.join(b_list)

	def next_img(self):
		img_list = [None, None, None, None, None]
		# img = cv2.putText(np.full((256, 192, 3), (255, 255, 255), np.uint8), str(self.frame_index), (96, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
		if self.config_dict["channel1"]:
			img_list[0] = np.ndarray(shape=(self.height, self.width, 3), dtype=np.uint8, buffer=self.img_memory0.buf)
		if self.config_dict["channel2"]:
			img_list[1] = np.ndarray(shape=(self.height, self.width, 3), dtype=np.uint8, buffer=self.img_memory2.buf)

		if self.config_dict["channel3"]:
			img_list[2] = np.ndarray(shape=(self.height, self.width, 3), dtype=np.uint8, buffer=self.img_memory4.buf)

		if self.config_dict["channel4"]:
			img_list[3] = np.ndarray(shape=(self.height, self.width, 3), dtype=np.uint8, buffer=self.img_memory6.buf)

		if self.config_dict["channel5"]:
			img_list[4] = np.ndarray(shape=(self.height, self.width, 3), dtype=np.uint8, buffer=self.img_memory8.buf)


		self.send_img = img_list
		self.frame_index += 1

	def send(self):
		if self.config_dict["start_flag"] and self.config_dict["ip"] is not None:
			# self.logger.info("开始发送图片")
			for image_index, image in enumerate(self.send_img):
				if image is not None:
					start_time = time.time()  # 记录开始时间
					image = cv2.resize(image, (512 * 1, 384* 1))
					image = cv2.putText(image, f"channel {image_index + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
					#记录每张图片的处理信息
					self.logger.info(f"处理图片索引: {image_index + 1}")
					for s in self.img_to_udp_data(image):
						packet_size = len(s)
						self.udp_socket.sendto(s, (self.config_dict["ip"], self.config_dict["port"] + image_index))
						end_time = time.time()  # 记录结束时间
						duration = end_time - start_time
						self.logger.info(f"图片索引 {image_index + 1} 发送完成，发送数据包大小: {packet_size} 字节，用时 {duration:.4f} 秒")
					time.sleep(0.03)
			self.next_img()
def run(config):
	img_send = UDPImg(config)
	while True:
		img_send.send()
		time.sleep(1 / 30)
