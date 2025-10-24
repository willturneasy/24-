import os
import subprocess
import time
import uuid
from multiprocessing import shared_memory

import cv2

from video.conf import VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS


def create_share_memory(_name, _size):
	try:
		memory_obj = shared_memory.SharedMemory(create=True, size=_size, name=_name)
	except FileExistsError:
		memory_obj = shared_memory.SharedMemory(create=False, size=_size, name=_name)
	return memory_obj


def get_share_memory(_name, _size):
	return shared_memory.SharedMemory(create=False, size=_size, name=_name)


def out_video(img_list, out_path):
	"""
	-y表示覆盖输出文件，如果该文件已经存在；
	-pattern_type glob表示使用glob模式读取文件，这样可以顺序读取input中所有后缀为.jpg的图片文件；
	-i 'input/*.jpg'指定输入文件的格式和路径；
	-r 30表示设置输出视频的帧率为每秒30帧；
	-s 1280x720表示设置输出视频的分辨率为1280x720；
	-c:v libx264表示使用 H.264 视频编码器进行编码；
	-pix_fmt yuv420p表示使用 yuv420p 彩色空间格式；
	output/output.mp4 表示写出到的文件名。
	"""
	width = VIDEO_WIDTH
	height = VIDEO_HEIGHT
	if len(img_list) > 0:
		height = img_list[0].shape[0]
		width = img_list[0].shape[1]
	random_name = uuid.uuid4()
	avi_name = f'{out_path}/{random_name}.avi'
	mp4_name = f'{out_path}/{random_name}.mp4'
	fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')  # 其中*'MP4V'和 'M', 'P', '4', 'V'等效
	out = cv2.VideoWriter(avi_name, fourcc, VIDEO_FPS, (int(width), int(height)))
	#
	# out_path = f"{out_path}/{random_name}.mp4"
	# out_command = [
	# 	"ffmpeg",
	# 	"-threads",
	# 	"6",
	# 	"-y",
	# 	'-f', 'rawvideo',
	# 	'-vcodec', 'rawvideo',
	# 	'-pix_fmt', 'bgr24',
	# 	"-r", str(VIDEO_FPS),
	# 	# "-r", str(15),
	# 	"-s", f"{width}x{height}",
	# 	"-i", "-",
	# 	"-c:v", "libx264",
	# 	# '-preset', 'ultrafast',
	# 	# "-pix_fmt", "yuv420p",
	# 	'-f', 'flv',
	# 	out_path
	# ]

	# cmd = subprocess.Popen(out_command, shell=False, stdin=subprocess.PIPE)
	# out = cmd.stdin
	for o_img in img_list:
		out.write(o_img)

	out.release()
	cmd = subprocess.Popen(f"ffmpeg -y -i {avi_name} -c:v libx264 -x264-params crf=23 {mp4_name}",
						   shell=True)
	cmd.wait()

	time.sleep(4)
	if os.path.exists(mp4_name):
		return mp4_name
	else:
		return None


def find_most_frequent_element(lst):
	# 使用max函数找到列表中出现次数最多的元素
	# 使用len函数计算每个元素出现的次数
	# 使用一个lambda函数将元素映射到其出现的次数上
	lst = [x for x in lst if x != ""]
	return max(set(lst), key=lst.count)


def format_number(num, length=13):
	return "0" * (length - len(str(num))) + str(num)
