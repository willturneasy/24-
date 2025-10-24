import sys
import time
from concurrent.futures import ThreadPoolExecutor
from ctypes import *
from typing import Any

from log.log import Log
from upload.conf import CAR_ID


class VciInitConfig(Structure):
	"""
	配置类
	"""
	_fields_ = [("AccCode", c_uint),  # 验收码。SJA1000的帧过滤验收码。对经过屏蔽码过滤为“有关位”进行匹配，全部匹配成功后，此帧可以被接收。否则不接收。
				("AccMask", c_uint),  # 屏蔽码。SJA1000的帧过滤屏蔽码。对接收的CAN帧ID进行过滤，对应位为0的是“有关位”，对应位为1的是“无关位”。屏蔽码推荐设置为0xFFFFFFFF，即全部接收。
				("Reserved", c_uint),  # 保留。
				("Filter", c_ubyte),  # 滤波方式，允许设置为1-3，
				("Timing0", c_ubyte),  # 波特率定时器 0
				("Timing1", c_ubyte),  # 波特率定时器 1
				("Mode", c_ubyte)  # 模式。=0表示正常模式（相当于正常节点），=1表示只听模式（只接收，不影响总线），=2表示自发自收模式（环回模式）
				]


class CanObj(Structure):
	"""
	消息类
	"""
	_fields_ = [("ID", c_uint),  # 报文帧ID
				("TimeStamp", c_uint),  # 接收到信息帧时的时间标识，从CAN控制器初始化开始计时，单位微秒
				("TimeFlag", c_ubyte),  # 是否使用时间标识，为1时TimeStamp有效，TimeFlag和TimeStamp只在此帧为接收帧时有意义。
				("SendType", c_ubyte),
				# 发送帧类型。=0时为正常发送，=1时为单次发送（不自动重发），=2时为自发自收（用于测试CAN卡是否损坏），=3时为单次自发自收（只发送一次，用于自测试），只在此帧为发送帧时有意义
				("RemoteFlag", c_ubyte),  # 是否是远程帧。=0时为数据帧，=1时为远程帧
				("ExternFlag", c_ubyte),  # 是否是扩展帧。=0时为标准帧（11位帧ID），=1时为扩展帧（29位帧ID）
				("DataLen", c_ubyte),  # 数据长度DLC(<=8)，即Data的长度
				("Data", c_ubyte * 8),  # CAN报文的数据。空间受DataLen的约束
				("Reserved", c_ubyte * 3)]  # 系统保留。


class CanMessage(object):
	def __init__(self):
		pass
		self._id = 0x1
		self.send_type = 0
		self.remote = 0
		self.extend = 0

		ubyte_array = c_ubyte * 8
		a = ubyte_array(1, 2, 3, 4, 5, 6, 7, 8)

		ubyte_3array = c_ubyte * 3
		self.reserved = ubyte_3array(0, 0, 0)
		self._data_list = []

	# vci_can_obj = VCI_CAN_OBJ(0x1, 0, 0, 1, 0, 0, 8, a, b)  # 单次发送

	def set_id(self, _id):
		self._id = _id

	def set_send_type(self, send_type):
		self.send_type = send_type

	def set_remote_flag(self):
		self.remote = 1

	def set_extend_flag(self):
		self.extend = 1

	def set_normal_flag(self):
		self.extend = 0

	def set_data_flag(self):
		self.remote = 0

	def set_data(self, data):
		while len(data) > 8:
			ubyte_array = (c_ubyte * 8)(*data[:8])
			self._data_list.append(ubyte_array)
			data = data[8:]
		ubyte_array = (c_ubyte * len(data))(*data)
		self._data_list.append(ubyte_array)

	def get_message(self):
		for _data in self._data_list:
			yield CanObj(self._id, 0, 0, self.send_type, self.remote, self.extend, len(_data), get_format_data(_data), self.reserved)
		self._data_list = []


def get_format_data(data, length=8):
	length_diff = length - len(data)
	if length_diff != 0:
		# 创建一个新的、更大的数组
		new_arr = (c_ubyte * length)()

		# 复制原数组到新数组
		for i in range(len(data)):
			new_arr[i] = data[i]

		# 如果需要，填充新数组剩余的部分
		for i in range(len(data), length):
			new_arr[i] = 0  # 或者其他默认值
		return new_arr
	else:
		return data


class BoardInfo(Structure):
	"""
	板卡信息
	"""
	_fields_ = [("hw_Version", c_ushort),  # 硬件版本号，用16进制表示
				("fw_Version", c_ushort),  # 固件版本号，用16进制表示
				("dr_Version", c_ushort),  # 驱动程序版本号，用16进制表示
				("in_Version", c_ushort),  # 接口库版本号，用16进制表示
				("irq_Num", c_ushort),  # 板卡所使用的中断号
				("can_Num", c_byte),  # 表示有几路CAN通道
				("str_Serial_Num", c_byte * 20),  # 此板卡的序列号，用ASC码表示
				("str_hw_Type", c_byte * 40),  # 硬件类型，用ASC码表示
				("Reserved", c_byte * 4)]  # 系统保留


class CanObjArray(Structure):
	"""
	信息数组
	"""
	_fields_ = [('SIZE', c_uint16), ('STRUCT_ARRAY', POINTER(CanObj))]

	def __init__(self, num_of_structs, *args: Any, **kw: Any):
		# 这个括号不能少
		super().__init__(*args, **kw)
		self.STRUCT_ARRAY = cast((CanObj * num_of_structs)(), POINTER(CanObj))  # 结构体数组
		self.SIZE = num_of_structs  # 结构体长度
		self.address = self.STRUCT_ARRAY[0]  # 结构体数组地址  byref()转c地址


CAN_TIMING = {
	"1M": (0, 0x14),
	"800k": (0, 0x16),
	"666k": (0x80, 0xb6),
	"500k": (0, 0x1c),
	"400k": (0x80, 0xfa),
	"250k": (0x01, 0x1c),
	"200k": (0x81, 0xfa),
	"125k": (0x03, 0x1c),
	"100k": (0x04, 0x1c),
	"80k": (0x83, 0xff),
	"50k": (0x09, 0x1c),
}

CAN_MODE = {
	"normal": 0,  # 正常模式（相当于正常节点）
	"listen": 1,  # 只听模式（只接收，不影响总线）
	"loopBack": 2,  # 自发自收模式（环回模式）。
}
CAN_FILTER = {
	"default": 0,  #
	"all": 1,  # 接收所有类型 滤波器同时对标准帧与扩展帧过滤！
	"standard": 2,  # 只接收标准帧 滤波器只对标准帧过滤，扩展帧将直接被滤除。
	"extend": 3  # 只接收扩展帧 滤波器只对扩展帧过滤，标准帧将直接被滤除。
}
RESERVED = 0


class Can(object):
	def __init__(self, device_index=0, timing="250k", mode="listen", can_filter="default",log_path="logs"):
		self.electric_cost = None
		self.head_light = None
		self.epb = None
		self.right_turn = None
		self.left_turn = None
		self.double_flash = None
		self.speed = None
		self.head_stock_angle = None
		self.gears = None
		self.reverse_disk_angle = None
		self.can_dll = None
		self.device_index = device_index
		self.device_type = 4
		self.message_array_len = 2500
		self.receive_message_array = CanObjArray(self.message_array_len)  # 结构体数组
		self.receive_thread_pool = ThreadPoolExecutor(max_workers=1)
		self._message_func = self.default_message_func
		self.timing = CAN_TIMING.get(timing) if CAN_TIMING.get(timing) is not None else CAN_TIMING.get("250k")
		self.mode = CAN_MODE.get(mode) if CAN_MODE.get(mode) is not None else CAN_MODE.get("listen")
		self.can_filter = CAN_FILTER.get(can_filter) if CAN_FILTER.get(can_filter) is not None else CAN_FILTER.get("default")
		self.log = Log(log_path, "Can.out")

	def prepare(self):
		"""
		前置
		:return:
		"""
		self.init_lib()
		# if self.close_device():
		return self.open_device()
		# return False

	def init_lib(self):
		"""
		根据平台选择依赖
		:return:
		"""
		if "linux" in sys.platform:
			can_dll_name = './libcontrolcan.so'  # 把SO放到对应的目录下,LINUX
			self.can_dll = cdll.LoadLibrary(can_dll_name)
		else:
			can_dll_name = './ControlCAN.dll'  # 把DLL放到对应的目录下
			self.can_dll = windll.LoadLibrary(can_dll_name)

	def open_device(self):
		"""
		打开设备
		:return:
		"""
		_ret = self.can_dll.VCI_OpenDevice(self.device_type, self.device_index, 0)
		return self.handle_ret(_ret, "open_device")

	def close_device(self):
		"""
		关闭设备
		:return:
		"""
		_ret = self.can_dll.VCI_CloseDevice(self.device_type, self.device_index)
		return self.handle_ret(_ret, "close_device")

	def _init_can(self, can_index=0, all_flag=False, target_id=0x80000008, more_id=0x0):
		"""
		初始化指定的CAN通道。有多个CAN通道时，需要多次调用。
		:param can_index:
		:param all_flag:
		:param target_id:
		:param more_id:
		:return:
		"""
		_code, _mask = get_acc(target_id, more_id, all_flag=all_flag)
		_ret = self.can_dll.VCI_InitCAN(self.device_type, self.device_index, can_index,
										byref(VciInitConfig(_code, _mask, RESERVED, self.can_filter, self.timing[0], self.timing[1], self.mode)))
		return self.handle_ret(_ret, "init_can")

	def _start_can(self, can_index=0):
		"""
		以启动CAN卡的某一个CAN通道。有多个CAN通道时，需要多次调用。
		:param can_index:
		:return:
		"""
		_ret = self.can_dll.VCI_StartCAN(self.device_type, self.device_index, can_index)
		return self.handle_ret(_ret, "start_can")

	def open_can(self, can_index=0, all_flag=True, target_id=0x80000000, more_id=0x0):
		"""
		打开can通道
		:param more_id:
		:param target_id:
		:param all_flag:
		:param can_index:
		:return:
		"""
		if self._init_can(can_index, all_flag, target_id, more_id):
			return self._start_can(can_index)
		return False

	def reset_can(self, can_index=0):
		"""
		数用以复位CAN。主要用与 VCI_StartCAN配合使用，无需再初始化，即可恢复CAN卡的正常状态。比如当CAN卡进入总线关闭状态时，可以调用这个函数
		:param can_index:
		:return:
		"""
		_ret = self.can_dll.VCI_ResetCAN(self.device_type, self.device_index, can_index)
		return self.handle_ret(_ret, "start_can")

	def send(self, can_index=0, message=CanObj()):
		"""
		发送消息
		:param can_index:
		:param message:
		:return:
		"""
		_ret = self.can_dll.VCI_Transmit(self.device_type, self.device_index, can_index, byref(message), c_uint16(1))
		return self.handle_message_ret(_ret, can_index)

	def receive(self, can_index=0):
		"""
		接收消息
		:param can_index:
		:return:
		"""
		return self.can_dll.VCI_Receive(self.device_type, self.device_index, can_index, byref(self.receive_message_array.address), self.message_array_len, 0)

	def clear_buffer(self, can_index=0):
		"""
		清除缓存
		:param can_index:
		:return:
		"""
		_ret = self.can_dll.VCI_ClearBuffer(self.device_type, self.device_index, can_index)
		return self.handle_ret(_ret, "clear_buffer")

	def set_message_func(self, message_func):
		"""
		设置消息的处理函数
		:param message_func:
		:return:
		"""
		self._message_func = message_func

	def default_message_func(self, message, can_index=0):
		"""
		默认处理函数
		:param message:
		:param can_index:
		:return:
		"""
		# print(f"设备序号：{self.device_index},can序号：{can_index},接收数据：")
		# print('ID：', end="")
		# print(hex(message.ID), end=" ")
		# print('DataLen：', end="")
		# print(hex(message.DataLen), end=" ")
		# print('Data：', end="")
		# print(bytes(message.Data), end=" ")
		# print(message.Data, end=" ")
		if message.ID == 0x1804A0B0:
			self.gears = ndr_upload_map.get(parse(message.Data, 2, 4))
			self.epb = hand_brake_upload_map.get(parse(message.Data, 0, 2))
			print("档位", ndr_map.get(parse(message.Data, 2, 4)), end=",")
			print("电子手刹状态", hand_brake_map.get(parse(message.Data, 0, 2)))
		if message.ID == 0x1806A0B0:
			self.left_turn = lamp_upload_map.get(parse(message.Data, 4, 1))
			self.right_turn = lamp_upload_map.get(parse(message.Data, 5, 1))
			self.double_flash = lamp_upload_map.get(parse(message.Data, 6, 1))
			self.head_light = big_lamp_upload_map.get(parse(message.Data, 8, 2))
			self.speed = parse(message.Data, 16, 8, offset=-50)
			self.electric_cost = parse(message.Data, 24, 8, 0.5)

			print("左转向灯状态", lamp_map.get(parse(message.Data, 4, 1)), end=",")
			print("右转向灯状态", lamp_map.get(parse(message.Data, 5, 1)), end=",")
			print("双闪灯状态", lamp_map.get(parse(message.Data, 6, 1)), end=",")
			print("大灯状态", big_lamp_map.get(parse(message.Data, 8, 2)), end=",")
			print("车速", parse(message.Data, 16, 8, offset=-50))
			print("电量", parse(message.Data, 24, 8, 0.5))

		if message.ID == 0x1802A0B0:
			self.reverse_disk_angle = parse(message.Data, 24, 16, 0.1, -1080)
			print("方向盘角度", parse(message.Data, 24, 16, 0.1, -1080))
		# if message.ID == 0x1810A0B0:
		# 	print("动力电池电压", parse(message.Data, 0, 16, 0.2), end=",")
		# 	print("动力电池电流", parse(message.Data, 16, 16, 0.02, -500), end=",")
		# 	print("累计充电电量", parse(message.Data, 32, 16), end=",")
		# 	print("累计输出电量", parse(message.Data, 48, 16))

	def receive_loop(self, can_index=0, queue=None):
		"""
		持续接收消息
		:param queue:
		:param can_index:
		:return:
		"""
		pre_time = time.time()
		receive_count = 0
		while True:
			receive_len = self.receive(can_index)
			receive_count += 1
			try:
				# for _i in range(receive_len):
				for _i in range(0, receive_len):
					self._message_func(self.receive_message_array.STRUCT_ARRAY[_i], can_index=can_index)
				if queue is not None:
					if time.time() - pre_time >= 1:
						data = {
								"car_id": CAR_ID,
								"push_timestamp": int(time.time() * 1000),
								"reverse_disk_angle": self.reverse_disk_angle,
								"gears": self.gears,
								"head_stock_angle": 0,
								"speed": self.speed,
								"double_flash": self.double_flash,
								"left_turn": self.left_turn,
								"right_turn": self.right_turn,
								"epb": self.epb,
								"head_light": self.head_light,
								"electric_cost": self.electric_cost,
							}
						queue.put({
							"data": data,
							"topic": "auto/vehicle/can/vehicleStatus"
						})
						self.log.log(data)

						pre_time = time.time()

			except BaseException as e:
				print(e)

			if receive_count > 100:
				self.clear_buffer(can_index)
				receive_count = 0

	def start_receive_loop(self, can_index=0):
		"""
		启动接收消息线程
		:param can_index:
		:return:
		"""
		self.receive_thread_pool.submit(self.receive_loop, can_index)

	def read_board_info(self):
		"""
		读取板卡信息
		:return:
		"""
		_board_info = BoardInfo()
		_ret = self.can_dll.VCI_ReadBoardInfo(self.device_type, self.device_index, byref(_board_info))
		if self.handle_ret(_ret, "read_board_info"):
			return _board_info

	def handle_ret(self, _ret, handle=""):
		"""
		返回标志处理
		:param _ret:
		:param handle:
		:return:
		"""
		if _ret == 1:
			return True
		else:
			if _ret == 0:
				print(f"设备序号：{self.device_index},{handle}操作失败！")
			elif _ret == -1:
				print("USB-CAN设备不存在或USB掉线")
			return False

	def handle_message_ret(self, _ret, can_index):
		"""
		消息返回标志处理
		:param _ret:
		:param can_index:
		:return:
		"""
		if _ret >= 0:
			print(f"设备序号：{self.device_index},can序号：{can_index},发送了{_ret}帧数据")
			return True
		else:
			print("USB-CAN设备不存在或USB掉线")
			return False


def get_acc(target_id, more_ids, _id_len=3, all_flag=False):
	"""
	返回accCode和accMask,用来过滤id
	:param target_id: 起始id
	:param more_ids: 结束id，或者id列表
	:param _id_len: 标准21，扩展3
	:param all_flag: 不过滤
	:return:
	"""
	if not all_flag:
		# 取32位二进制的前部有效位，后续不需要再左移
		_s = ["0"] * (32 - _id_len)
		target_ids = []
		# 如果给了开始id和结束id就使用range生成范围
		if type(more_ids) == int:
			target_ids = range(min(target_id, more_ids), max(target_id, more_ids) + 1)
		elif type(more_ids) == list:
			# 如果给了目标ID和其他ID，就直接组合
			target_ids.extend(more_ids)
			target_ids.append(target_id)
		# 去重，排序
		target_ids = list(set(target_ids))
		target_ids.sort()

		# 对每个ID进行操作
		for j in target_ids:
			# 跟目标ID每一位进行比较，如果不一致就需要屏蔽
			for i in range(32):
				# 使用1左移可以保证，32二进制中只有一个1，
				n = 1 << i
				# 是否一致
				if (n & target_id) != (n & j):
					_s[-(i + 1)] = "1"
		_s = "".join(_s)
		# 这边填充1或者0都可以
		_s = _s + "0" * _id_len
		_s = int(_s, 2)
		return (target_id << _id_len), _s
	else:
		return (target_id << _id_len), 0xFFFFFFFF


def hex_bin(hex_data):
	byte_li = []
	for hex_str in range(0, len(hex_data)):
		data = hex_data[hex_str]
		binary_str = bin(data)[2:]
		if len(binary_str) < 8:
			binary_str = '0' * (8 - len(binary_str)) + binary_str
		byte_li.append(binary_str[::-1])
	return byte_li


def start_byte(bin_data, start_bit, length):
	data = "".join(bin_data)
	return int(data[start_bit:start_bit + length][::-1], 2)


def get_real_value(value, scale=1.0, offset=0.0):
	return value * scale + offset


def parse(hex_str, start, length, scale=1.0, offset=0.0):
	return get_real_value(start_byte(hex_bin(hex_str), start, length), scale, offset)


ndr_map = {
	0.0: "N档",
	1.0: "D档",
	2.0: "R档",
	3.0: "其他",
}
ndr_upload_map = {
	0.0: 2,
	1.0: 3,
	2.0: 1,
	3.0: 0,
}
lamp_map = {
	0.0: "关闭",
	1.0: "闪烁"
}
lamp_upload_map = {
	0.0: 1,
	1.0: 2
}
big_lamp_map = {
	0.0: "保留",
	1.0: "近光开启",
	2.0: "远光开启",
	3.0: "关闭",
}
big_lamp_upload_map = {
	0.0: 0,
	1.0: 1,
	2.0: 2,
	3.0: 0,
}
hand_brake_map = {
	0.0: "保留",
	1.0: "拉起手刹",
	2.0: "释放手刹",
	3.0: "行进间制动"
}
hand_brake_upload_map = {
	1.0: 2,
	2.0: 1,
}


def run(queue,log_path="logs"):
	canHandle = Can(can_filter="extend",log_path=log_path)
	print("can init")
	canHandle.prepare()
	print("can prepare")
	# canHandle.open_can(0, all_flag=False, target_id=0x1804A0B0, more_id=[0x1806A0B0, 0x1802A0B0, 0x1810A0B0])
	canHandle.open_can(0, all_flag=False, target_id=0x1804A0B0, more_id=[0x1806A0B0, 0x1802A0B0])
	print("can open_can 0")
	# canHandle.open_can(1, all_flag=False, target_id=0x123, more_id=0x120)
	# canHandle.start_receive_loop(0)
	print("can open_can 0")
	canHandle.receive_loop(0, queue)


