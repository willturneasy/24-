import base64
import hashlib

import cv2


def get_md5(text):
	md = hashlib.md5(str(text).encode())  # 创建md5对象
	return md.hexdigest()


def img_to_base64(img_array):
	encode_image = cv2.imencode(".jpg", img_array)[1]  # 用cv2压缩/编码，转为一维数组
	byte_data = encode_image.tobytes()  # 转换为二进制
	base64_str = base64.b64encode(byte_data).decode("ascii")  # 转换为base64
	return base64_str
