import math
import socket
import struct
import time

from log.log import Log
from .conf import *
from upload.conf import CAR_ID
type_str_map = {
	"Uchar": "B",
	"Char": "c",
	"Uint": "I",
	"Ulong": "L",
	"Ushort": "H",
	"Double": "d",
	"Float": "f",
}
inspvaxb_lengths = [1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 4, 4, 2, 2, 4, 4, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4]
inspvaxb_types = ["Uchar", "Uchar", "Uchar", "Uchar", "Ushort", "Char", "Uchar", "Ushort", "Ushort", "Uchar", "Uchar", "Ushort", "Ulong", "Ulong", "Ushort", "Ushort",
				  "Ulong", "Ulong", "Double", "Double", "Double", "Float", "Double", "Double", "Double", "Double", "Double", "Double", "Float", "Float", "Float", "Float",
				  "Float", "Float", "Float", "Float", "Float", "Ulong", "Ushort", "Uint"]


def get_len(target_list, start=None, end=None):
	result = 0
	for item in target_list[start:end]:
		result += item
	return result


def crc32_value(num):
	crc = num
	for i in range(8):
		if crc & 1:
			crc = (crc >> 1) ^ 0xEDB88320
		else:
			crc >>= 1
	return crc


def crc32(target_list):
	crc = 0
	for i in range(len(target_list) - 4):
		temp = (crc >> 8) & 0x00FFFFFF
		temp2 = crc32_value(int(crc ^ target_list[i]) & 0xFF)
		crc = temp ^ temp2
	return crc


gps_map = {
	0: "定位无效",
	53: "单点解",
	54: "差分定位",
	56: "固定解",
	55: "浮点解",
	52: "递推",
}
def out_of_china(lng, lat):
	"""
	判断是否在国内，不在国内不做偏移
	:param lng:
	:param lat:
	:return:
	"""
	return not (135.05 > lng > 73.66 and 53.55 > lat > 3.86)

def transform_lng(lng, lat):
	ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
		  0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
	ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 *
			math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
	ret += (20.0 * math.sin(lng * math.pi) + 40.0 *
			math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
	ret += (150.0 * math.sin(lng / 12.0 * math.pi) + 300.0 *
			math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
	return ret


def transform_lat(lng, lat):
	ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
		  0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
	ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 *
			math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
	ret += (20.0 * math.sin(lat * math.pi) + 40.0 *
			math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
	ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 *
			math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
	return ret

def wgs84_to_gcj02(lng, lat):
	"""
	WGS84转GCJ02(火星坐标系)
	:param lng:WGS84坐标系的经度
	:param lat:WGS84坐标系的纬度
	:return:
	"""
	a = 6378245.0  # 长半轴
	ee = 0.00669342162296594323  # 扁率
	if out_of_china(lng, lat):  # 判断是否在国内
		return lng, lat
	dlat = transform_lat(lng - 105.0, lat - 35.0)
	dlng = transform_lng(lng - 105.0, lat - 35.0)
	radlat = lat / 180.0 * math.pi
	magic = math.sin(radlat)
	magic = 1 - ee * magic * magic
	sqrtmagic = math.sqrt(magic)
	dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
	dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
	mglat = lat + dlat
	mglng = lng + dlng
	return [mglng, mglat]

class RTKRec(object):
	def __init__(self, ip=IP, port=PORT, mqtt_publish_queue=None,gps_dict=None,log_path="logs"):
		# 本机的ip和端口
		self.ip, self.port = ip, port
		self.socket = None
		self.data_list = None
		# self.data_type = ""
		self.change_time = time.time()
		self.lng = None
		self.lat = None
		self.timestamp = None
		self.angle = None
		self.speed = None
		self.stop_thread = False
		self.gps_change_time = time.time()
		self.gps_type = None
		self.log = Log(log_path, "RTKRec.out")
		self.mqtt_publish_queue = mqtt_publish_queue
		self.gps_dict = gps_dict

	def parse_inspvaxb(self, data_index):
		if self.data_list is not None:
			data_len = inspvaxb_lengths[data_index - 1]
			data_start_index = get_len(inspvaxb_lengths, end=data_index - 1)
			return struct.unpack("<{}".format(type_str_map[inspvaxb_types[data_index - 1]]), bytes(self.data_list[data_start_index:data_start_index + data_len]))[0]

	# def change_data_type(self, data_type):
	#     self.data_type = data_type
	#     self.change_time = time.time()
	#
	# def get_data_type(self):
	#     data_type = self.data_type
	#     # if self.data_type != "" and time.time() - self.change_time > 0.1:
	#     #     self.change_data_type("")
	#     self.change_data_type("")
	#     return data_type

	def set_gps(self, lng, lat, sec, angle, speed, gps_type):
		self.lng = lng
		self.lat = lat
		self.timestamp = sec
		self.angle = angle
		self.speed = speed
		self.gps_change_time = time.time()
		self.gps_type = gps_type

	def get_gps(self):
		lng, lat, timestamp, angle, speed, gps_type = self.lng, self.lat, self.timestamp, self.angle, self.speed, self.gps_type
		if time.time() - self.gps_change_time > 1:
			self.lng = None
			self.lat = None
			self.timestamp = None
			self.angle = None
			self.speed = None
			self.gps_change_time = time.time()
			self.gps_type = None
		return [lng, lat, timestamp, angle, speed, gps_type]

	def rec_handle(self):
		# 使用IPV4协议，使用UDP协议传输数据
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.socket.bind((self.ip, self.port))
		pre_time = time.time() - 1
		upload_fps = 1
		while not self.stop_thread:
			try:
				self.data_list, addr = self.socket.recvfrom(1024)
				# crc32检查
				if self.parse_inspvaxb(40) == crc32(self.data_list):
					# self.log.log(f'rtk:{self.data_list}')
					weeks = self.parse_inspvaxb(12)
					secs = self.parse_inspvaxb(13) / 1000
					sec = weeks * 7 * 24 * 3600 + secs
					lng = self.parse_inspvaxb(20)
					lat = self.parse_inspvaxb(19)
					speed = round(3.6 * math.sqrt(self.parse_inspvaxb(23) ** 2 + self.parse_inspvaxb(24) ** 2), 2)
					angle = self.parse_inspvaxb(28)
					gps_type = self.parse_inspvaxb(18)
					self.set_gps(lng, lat, sec, angle, speed, gps_type)
					if self.mqtt_publish_queue is not None:
						if time.time() - pre_time >= (1 / upload_fps):
							t_ll = wgs84_to_gcj02(lng, lat)
							data = {
								"carID": CAR_ID,
								"lng": t_ll[0],
								"lat": t_ll[1],
								"angle": angle,
								"speed": speed,
								"timestamp": int(time.time() * 1000),
							}
							self.mqtt_publish_queue.put({
								"data": data,
								"topic": "auto/drive/behavior/danger"
							})
							self.log.log(data)
							pre_time = time.time()

					if self.gps_dict is not None:
						self.gps_dict["lng"] = lng
						self.gps_dict["lat"] = lat
						self.gps_dict["angle"] = angle
						self.gps_dict["speed"] = speed
			except socket.error as e:
				# self.log.log(f'Error:{e}')
				pass

def terminate(self):
	self.stop_thread = True
	self.log.terminate()


def run(mqtt_publish_queue,gps_dict,log_path="logs"):
	#rtk_r = RTKRec(ALL_CONF["ip"], ALL_CONF["port"])
	rtk_r = RTKRec(mqtt_publish_queue=mqtt_publish_queue,gps_dict=gps_dict,log_path=log_path)
	rtk_r.rec_handle()
	
	
	
	
