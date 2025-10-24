def hex_bin(hex_data):
	byte_li = []
	for hex_str in range(0, len(hex_data), 2):
		data = hex_data[hex_str:hex_str + 2]
		binary_str = bin(int(data, 16))[2:]
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
lamp_map = {
	0.0: "关闭",
	1.0: "闪烁"
}
big_lamp_map = {
	0.0: "保留",
	1.0: "近光开启",
	2.0: "远光开启",
	3.0: "关闭",
}
hand_brake_map = {
	0.0: "保留",
	1.0: "拉起手刹",
	2.0: "释放手刹",
	3.0: "行进间制动"
}

if __name__ == '__main__':
	with open("target.csv", encoding="gbk", mode="r") as file:
		pre_time = ""
		for line in file:
			line_list = line.split(",")
			if line_list[1] != pre_time:
				pre_time = line_list[1]
				print()
			if line_list[5] == '0x1804A0B0':
				s = "".join(line_list[-1].replace("\n", "").split(" ")[1:-1])
				print("档位", ndr_map.get(parse(s, 2, 4)), end=",")
				print("电子手刹状态", hand_brake_map.get(parse(s, 0, 2)), end=",")
			if line_list[5] == '0x1806A0B0':
				s = "".join(line_list[-1].replace("\n", "").split(" ")[1:-1])
				print("左转向灯状态", lamp_map.get(parse(s, 4, 1)), end=",")
				print("右转向灯状态", lamp_map.get(parse(s, 5, 1)), end=",")
				print("双闪灯状态", lamp_map.get(parse(s, 6, 1)), end=",")
				print("大灯状态", big_lamp_map.get(parse(s, 8, 2)), end=",")
				print("车速", parse(s, 16, 8, offset=-50), end=",")

			if line_list[5] == '0x1815A0B0':
				s = "".join(line_list[-1].replace("\n", "").split(" ")[1:-1])
				print("方向盘角速度", parse(s, 0, 9), end=",")
