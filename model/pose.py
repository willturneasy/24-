import math

import cv2
import numpy as np
from scipy.spatial import distance as dist

from .conf import head_eye_close, head_yaw_offset, head_yaw_degree, head_pitch_degree, head_pitch_offset, head_yawn


def get_box_data(box):
	"""
	获取探测结果
	:param box:
	:param names:
	:return:
	"""
	return int(box[0]), int(box[1]), int(box[2]), int(box[3])


def get_point_and_conf(key_point, shape=(640, 640), min_conf=0.5):
	"""
	获取点位和可信度
	:param min_conf:
	:param key_point:
	:param shape:
	:return:
	"""
	x_coord, y_coord = key_point[0], key_point[1]
	if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
		if len(key_point) == 3:
			conf = key_point[2]
			if conf < min_conf:
				return None
			return float(x_coord), float(y_coord), float(conf)
	return None


def eye_aspect_ratio(eye):
	"""
	计算眼部长宽比
	:param eye:
	:return:
	"""
	# 垂直眼标志（X，Y）坐标
	A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
	B = dist.euclidean(eye[2], eye[4])
	# 计算水平之间的欧几里得距离
	# 水平眼标志（X，Y）坐标
	C = dist.euclidean(eye[0], eye[3])
	if C != 0:
		# 眼睛长宽比的计算
		ear = (A + B) / (2.0 * C)
		# 返回眼睛的长宽比
		return int(ear * 100)
	else:
		return 255


def mouth_aspect_ratio(mouth):
	"""
	嘴部长宽比
	:param mouth:
	:return:
	"""
	# 外圈
	# A = dist.euclidean(mouth[2], mouth[9])  # 51, 59
	# B = dist.euclidean(mouth[4], mouth[7])  # 53, 57
	# C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
	# 内圈
	A = dist.euclidean(mouth[64 - 49], mouth[66 - 49])  # 51, 59
	B = dist.euclidean(mouth[62 - 49], mouth[68 - 49])  # 53, 57
	C = dist.euclidean(mouth[61 - 49], mouth[65 - 49])  # 49, 55
	if C != 0:
		mar = (A + B) / (2.0 * C)
		return int(mar * 100)
	else:
		return 255


def get_pose_estimation(img_size, shape):
	"""
	获取旋转向量和平移向量
	:param img_size:
	:param shape:
	:return:
	"""
	# 3D model points.
	model_points = np.array([
		(6.825897, 6.760612, 4.402142),  # 33 left brow left corner
		(1.330353, 7.122144, 6.903745),  # 29 left brow right corner
		(-1.330353, 7.122144, 6.903745),  # 34 right brow left corner
		(-6.825897, 6.760612, 4.402142),  # 38 right brow right corner
		(5.311432, 5.485328, 3.987654),  # 13 left eye left corner
		(1.789930, 5.393625, 4.413414),  # 17 left eye right corner
		(-1.789930, 5.393625, 4.413414),  # 25 right eye left corner
		(-5.311432, 5.485328, 3.987654),  # 21 right eye right corner
		(2.005628, 1.409845, 6.165652),  # 55 nose left corner
		(-2.005628, 1.409845, 6.165652),  # 49 nose right corner
		(2.774015, -2.080775, 5.048531),  # 43 mouth left corner
		(-2.774015, -2.080775, 5.048531),  # 39 mouth right corner
		(0.000000, -3.116408, 6.097667),  # 45 mouth central bottom corner
		(0.000000, -7.415691, 4.070434)  # 6 chin corner
	])
	# Camera internals

	focal_length = img_size[1]
	center = (img_size[1] / 2, img_size[0] / 2)
	camera_matrix = np.array(
		[[focal_length, 0, center[0]],
		 [0, focal_length, center[1]],
		 [0, 0, 1]], dtype="double"
	)

	dist_coeffs = np.array([7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000],
						   dtype="double")  # Assuming no lens distortion

	(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, np.array(shape[[17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]], dtype="double"),
																  camera_matrix,
																  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
	return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs


def get_euler_angle(rotation_vector):
	"""
	从旋转向量转换为欧拉角
	:param rotation_vector:
	:return:
	"""
	# calculate rotation angles
	theta = cv2.norm(rotation_vector, cv2.NORM_L2)

	# transformed to quaterniond
	w = math.cos(theta / 2)
	x = math.sin(theta / 2) * rotation_vector[0][0] / theta
	y = math.sin(theta / 2) * rotation_vector[1][0] / theta
	z = math.sin(theta / 2) * rotation_vector[2][0] / theta

	ysqr = y * y
	# pitch (x-axis rotation)
	t0 = 2.0 * (w * x + y * z)
	t1 = 1.0 - 2.0 * (x * x + ysqr)

	# print('t0:{}, t1:{}'.format(t0, t1))
	pitch = math.atan2(t0, t1)

	# yaw (y-axis rotation)
	t2 = 2.0 * (w * y - z * x)
	if t2 > 1.0:
		t2 = 1.0
	if t2 < -1.0:
		t2 = -1.0
	yaw = math.asin(t2)

	# roll (z-axis rotation)
	t3 = 2.0 * (w * z + x * y)
	t4 = 1.0 - 2.0 * (ysqr + z * z)
	roll = math.atan2(t3, t4)

	# 单位转换：将弧度转换为度
	pitch_degree = int((pitch / math.pi) * 180)
	yaw_degree = int((yaw / math.pi) * 180)
	roll_degree = int((roll / math.pi) * 180)

	return pitch, yaw, roll, pitch_degree, yaw_degree, roll_degree


def draw_result(target_img, attrs, face_points):
	font = cv2.FONT_HERSHEY_PLAIN
	font_scale = 4
	font_thickness = 3
	green_color = (0, 255, 0)
	red_color = (0, 0, 255)

	left_eye, right_eye, mouth, pitch_degree, yaw_degree, roll_degree = attrs
	w, h = cv2.getTextSize(str(left_eye), font, font_scale, font_thickness)[0]
	if (left_eye + right_eye) / 2 < 25:
		text_color = red_color
	else:
		text_color = green_color

	cv2.putText(target_img, str(left_eye), (face_points[37][0] - int(w / 2), face_points[37][1] - h), font, font_scale, text_color, font_thickness)
	w, h = cv2.getTextSize(str(right_eye), font, font_scale, font_thickness)[0]
	# if right_eye < 20:
	#     text_color = red_color
	# else:
	#     text_color = green_color
	cv2.putText(target_img, str(right_eye), (face_points[44][0] - int(w / 2), face_points[44][1] - h), font, font_scale, text_color, font_thickness)
	w, h = cv2.getTextSize(str(mouth), font, font_scale, font_thickness)[0]
	if mouth > 60:
		text_color = red_color
	else:
		text_color = green_color
	cv2.putText(target_img, str(mouth), (face_points[57][0] - int(w / 2), face_points[57][1] + h), font, font_scale, text_color, font_thickness)
	if abs(pitch_degree) > 15:
		text_color = red_color
	else:
		text_color = green_color

	if pitch_degree < 0:
		text = str(f"up {abs(pitch_degree)}")
		w, h = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
		cv2.putText(target_img, text, (face_points[8][0] - int(w / 2), face_points[8][1] + 16), font, font_scale, text_color, font_thickness)
	if pitch_degree > 0:
		text = str(f"down {abs(pitch_degree)}")
		w, h = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
		cv2.putText(target_img, text, (face_points[8][0] - int(w / 2), face_points[8][1] + 16), font, font_scale, text_color, font_thickness)

	if abs(yaw_degree) > 45:
		text_color = red_color
	else:
		text_color = green_color
	if yaw_degree < 0:
		text = str(f"left {abs(yaw_degree)}")
		w, h = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
		cv2.putText(target_img, text, (face_points[0][0] - w, face_points[0][1]), font, font_scale, text_color, font_thickness)
	if yaw_degree > 0:
		cv2.putText(target_img, str(f"right {abs(yaw_degree)}"), (face_points[16][0], face_points[16][1]), font, font_scale, text_color, font_thickness)

	if abs(roll_degree) > 30:
		text_color = red_color
	else:
		text_color = green_color
	if roll_degree < 0:
		text = str(f"roll right {abs(roll_degree)}")
		w, h = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
		cv2.putText(target_img, text, (face_points[4][0] - w, face_points[4][1]), font, font_scale, text_color, font_thickness)
	if roll_degree > 0:
		cv2.putText(target_img, str(f"roll left {abs(roll_degree)}"), (face_points[12][0], face_points[12][1]), font, font_scale, text_color, font_thickness)

	avg_left = np.average(face_points[36:42], axis=0)
	avg_right = np.average(face_points[42:48], axis=0)
	eye_length = int(math.sqrt((avg_left[0] - avg_right[0]) ** 2 + (avg_left[1] - avg_right[1]) ** 2) * 3)
	return target_img


def get_length_sqrt(pts1, pts2):
	return math.sqrt((pts1[0] - pts2[0]) ** 2 + (pts1[1] - pts2[1]) ** 2)


def get_eye_attr(eye_center, eyes_distance, image, eye_model):
	eye_top, eye_left, eye_bottom, eye_right = int(eye_center[1] - eyes_distance / 3), \
											   int(eye_center[0] - eyes_distance / 2), \
											   int(eye_center[1] + eyes_distance / 3), \
											   int(eye_center[0] + eyes_distance / 2)
	# cv2.imshow("tt",image[eye_top:eye_bottom, eye_left:eye_right])

	_eye_img = image[eye_top:eye_bottom, eye_left:eye_right]
	_eye_img = adjust_gamma(_eye_img,_eye_img)
	results = eye_model.predict(_eye_img, verbose=False)
	probs = None
	# Process results list
	for result in results:
		probs = result.probs  # Probs object for classification outputs
	return eye_model.names[probs.top1]


def handle_img(target_img, pose_model, hand_model, face_model, eye_model, speed):
	eye_close = ""
	smoking = ""
	use_phone = ""
	poor_attention = ""
	yawn = ""

	# 第一步，姿态，不管有无遮挡，脸部都能准确识别
	results = pose_model.predict(target_img, verbose=False)
	attrs = []
	# 对每一个结果都进行处理
	for res in results:
		# 多个探测框
		for index, box in enumerate(res.boxes.data):
			attr = []
			rec = get_box_data(box)
			# 每个探测框，对关键点进行属性运算
			key_points = res.keypoints.data[index]
			# 先计算面积，如果只保留一个则选用最大面积,舍弃！改用脸部和肩部综合判断
			area = abs((rec[0] - rec[2]) * (rec[1] - rec[3]))
			# 保存面积
			attr.append(rec)
			attr.append(area)

			# 肩部2个关键点
			sho_points_len = None
			sho_points = np.zeros((2, 2), dtype="int")
			for i, point in enumerate(key_points[5:7]):
				sho_points[i] = ([int(point[0]), int(point[1])])
			if np.all(sho_points[0] == 0) and np.all(sho_points[1] == 0):
				pass
			else:
				if np.all(sho_points[0] == 0):
					sho_points[0] = rec[:2]
				if np.all(sho_points[1] == 0):
					sho_points[1] = rec[2:4]
				sho_points_len = int(get_length_sqrt(sho_points[0], sho_points[1]))
			attr.append(sho_points_len)

			# 脸部5个关键点
			points = np.zeros((5, 2), dtype="int")
			for i, point in enumerate(key_points[:5]):
				points[i] = ([int(point[0]), int(point[1])])
			attr.append(points)

			# 用非零的关键点计算脸部中心
			no_zero_points = points[[not np.all(points[i] == 0) for i in range(points.shape[0])], :]
			face_center = None
			if len(no_zero_points) > 0:
				face_center = np.int_(np.average(no_zero_points, axis=0))
			# 保存脸部中心
			attr.append(face_center)
			# 脸部范围
			face_radius = None
			# 鼻子，左右眼睛为脸部最重要属性，如果任意一个没有则认为该探测框脸部信息丢失，如果均存在则可以计算出脸部范围（扩大了！）
			if np.all(points[0] == 0) or np.all(points[1] == 0) or np.all(points[2] == 0):
				pass
			# cv2.putText(img, "no face", face_center, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
			else:
				length_list = []
				if not np.all(points[3] == 0):
					length_list.append(int(get_length_sqrt(points[3], points[1]) * 3))
				# length_list.append(int(math.sqrt((points[3][0] - points[1][0]) ** 2 + (points[3][1] - points[1][1]) ** 2) * 3))
				if not np.all(points[4] == 0):
					length_list.append(int(get_length_sqrt(points[4], points[2]) * 3))

				# length_list.append(int(math.sqrt((points[4][0] - points[2][0]) ** 2 + (points[4][1] - points[2][1]) ** 2) * 3))
				length_list.append(int(get_length_sqrt(points[1], points[2]) * 3))
				length_list.append(int(get_length_sqrt(points[0], points[2]) * 3))
				length_list.append(int(get_length_sqrt(points[0], points[1]) * 3))
				# length_list.append(int(math.sqrt((points[1][0] - points[2][0]) ** 2 + (points[1][1] - points[2][1]) ** 2) * 3))
				# length_list.append(int(math.sqrt((points[0][0] - points[2][0]) ** 2 + (points[0][1] - points[2][1]) ** 2) * 3))
				# length_list.append(int(math.sqrt((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2) * 3))
				face_radius = np.int_(np.average(length_list))
			# cv2.circle(img, face_center, int(face_radius * 4 / 3), (0, 255, 255), 1)

			attr.append(face_radius)
			attrs.append(attr)

	sho_len_index = 2
	face_radius_index = -1
	face_center_index = -2
	face_points_index = -3
	box_rec_index = 0
	if len(attrs) > 0:
		# 选取最大的区域
		max_radius = 0
		max_index = -1
		for i, attr in enumerate(attrs):
			face_radius = attr[face_radius_index] if attr[face_radius_index] is not None else -1
			sho_len = attr[sho_len_index] if attr[sho_len_index] is not None else -1
			# 脸部和肩部长度综合比较，得出最大的人脸
			max_len = max(int(face_radius * 8 / 3), sho_len)
			if max_len > max_radius:
				max_radius = max_len
				max_index = i
		# 如果没得到最大区域，则代表有人身，却无肩部和人脸，返回no face
		if max_index > -1 and attrs[max_index][face_center_index] is not None and attrs[max_index][face_radius_index] is not None:
			try:
				# 判断眼睛是否闭合
				left_eye = attrs[max_index][face_points_index][1]
				right_eye = attrs[max_index][face_points_index][2]
				if not np.all(left_eye == 0) and not np.all(right_eye == 0):
					e_length = math.sqrt((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2)
					half_len = int(e_length / 2)
					l_1 = get_eye_attr(left_eye, half_len, target_img, eye_model)
					r_1 = get_eye_attr(right_eye, half_len, target_img, eye_model)
					if l_1 == "closeEye" or r_1 == "closeEye":
						eye_close = "eyeClose"
				# return "eyeClose"

				else:
					poor_attention = "poorAttention"
			# return "poorAttention"

			except BaseException as e:
				print(e)
				poor_attention = "poorAttention"
			# return "poorAttention"

			max_radius = attrs[max_index][face_radius_index]
			# 截取区域
			people_area_rec = attrs[max_index][box_rec_index]
			people_face_center = (attrs[max_index][face_center_index][0] - people_area_rec[0], attrs[max_index][face_center_index][1] - people_area_rec[1])
			people_area_img = target_img[people_area_rec[1]:people_area_rec[3], people_area_rec[0]:people_area_rec[2]]

			# 对手部进行探测
			hand_result = hand_model.predict(people_area_img, verbose=False)
			for res in hand_result:
				for index, box in enumerate(res.boxes.data):
					rec = get_box_data(box)
					box_center = (int(rec[0] / 2.0 + rec[2] / 2.0), int(rec[1] / 2.0 + rec[3] / 2.0))
					# 计算手部和脸部的距离
					hand_face_distance = get_length_sqrt(people_face_center, box_center)
					# 如果小于脸部半径，即在脸部范围内，则认为是手部动作
					if hand_face_distance < max_radius:
						# cv2.circle(img, np.array(people_area_rec[:2]) + np.array(box_center), 3, (0, 0, 255), -1)
						hand_box_center = np.array(people_area_rec[:2]) + np.array(box_center)
						left_down_point = np.array(attrs[max_index][face_center_index]) - np.array([max_radius, -max_radius])
						right_down_point = np.array(attrs[max_index][face_center_index]) + np.array([max_radius, max_radius])
						mouth_points = np.array([attrs[max_index][face_center_index], left_down_point, right_down_point])
						# cv2.polylines(img,[mouth_points],True,(0,0,255),1)
						if cv2.pointPolygonTest(mouth_points, (int(hand_box_center[0]), int(hand_box_center[1])), False) >= 0:
							smoking = "smoking"
						# return "smoking"
						else:
							use_phone = "usePhone"
			# return "usePhone"
			try:
				face_img = get_adjust_face(people_area_img,people_face_center,max_radius)

				if face_img.shape[0] > 0 and face_img.shape[1] > 0:
					face_result = face_model.predict(face_img, verbose=False)
					has_face_result = False
					for res in face_result:
						# face_img = cv2.addWeighted(face_img,1,res.plot(),0.5,1)
						for index, box in enumerate(res.boxes.data):
							has_face_result = True

							key_points = res.keypoints.data[index]
							points = np.zeros((68, 2), dtype="int")
							for i, point in enumerate(key_points):
								points[i] = ([int(point[0]), int(point[1])])
							left_eye = eye_aspect_ratio(points[36:42])
							right_eye = eye_aspect_ratio(points[42:48])
							mouth = mouth_aspect_ratio(points[48:68])
							_, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(face_img.shape, points)
							pitch, yaw, roll, pitch_degree, yaw_degree, roll_degree = get_euler_angle(rotation_vector)
							pitch_degree = pitch_degree - head_pitch_offset
							yaw_degree = yaw_degree - head_yaw_offset
							if (left_eye + right_eye) / 2.0 < head_eye_close:
								eye_close = "eyeClose"
							# return "eyeClose"
							if mouth > head_yawn:
								yawn = "yawn"
							# return "yawn"
							if abs(pitch_degree) > head_pitch_degree or abs(yaw_degree) > head_yaw_degree or abs(roll_degree) > 30:
								poor_attention = "poorAttention"
					# return "poorAttention"
					# 人脸探测不到，说明姿态已经超出正常范围！
					if not has_face_result:
						poor_attention = "poorAttention"
			except BaseException as e:
				print(e)

		else:
			poor_attention = "poorAttention"
	# return "poorAttention"
	# return "no face"
	else:
		if speed is not None and speed > 10:
			poor_attention = "poorAttention"
	# return "poorAttention"
	# return "no people"
	# return "normal"
	return eye_close, yawn, poor_attention, smoking, use_phone


def get_adjust_face(_people_img, _face_center, _face_radius):
	_face_rec = [max((_face_center[0] - _face_radius), 0), max((_face_center[1] - _face_radius), 0), (_face_center[0] + _face_radius),
				 (_face_center[1] + _face_radius)]
	_light_rec = [max(int(_face_center[0] - _face_radius / 3), 0), max(int(_face_center[1] - _face_radius / 3), 0), int(_face_center[0] + _face_radius / 3),
				  int(_face_center[1] + _face_radius / 3)]

	_face_img = _people_img[
				_face_rec[1]:_face_rec[3],
				_face_rec[0]:_face_rec[2]
				]
	_light_img = _people_img[
				 _light_rec[1]:_light_rec[3],
				 _light_rec[0]:_light_rec[2]
				 ]
	return adjust_gamma(_face_img, _light_img)


def get_img_light(target_img):
	b, g, r = cv2.split(target_img)
	return np.mean(0.299 * r + 0.587 * g + 0.114 * b)


def adjust_gamma(_image, _light_img):
	_gamma = (256 - get_img_light(_light_img)) / 128

	_inv_gamma = 1.0 / _gamma
	table = np.array(
		[((i / 255.0) ** _inv_gamma) * 255 for i in np.arange(0, 256)]
	).astype("uint8")
	return cv2.LUT(_image, table)

# def handle_img(target_img, pose_model, hand_model, face_model,eye_model, speed):
# 	# 第一步，姿态，不管有无遮挡，脸部都能准确识别
# 	results = pose_model.predict(target_img, verbose=False)
# 	attrs = []
# 	# 对每一个结果都进行处理
# 	for res in results:
# 		# 多个探测框
# 		for index, box in enumerate(res.boxes.data):
# 			attr = []
# 			rec = get_box_data(box)
# 			# 每个探测框，对关键点进行属性运算
# 			key_points = res.keypoints.data[index]
# 			# 先计算面积，如果只保留一个则选用最大面积,舍弃！改用脸部和肩部综合判断
# 			area = abs((rec[0] - rec[2]) * (rec[1] - rec[3]))
# 			# 保存面积
# 			attr.append(rec)
# 			attr.append(area)
#
# 			# 肩部2个关键点
# 			sho_points_len = None
# 			sho_points = np.zeros((2, 2), dtype="int")
# 			for i, point in enumerate(key_points[5:7]):
# 				sho_points[i] = ([int(point[0]), int(point[1])])
# 			if np.all(sho_points[0] == 0) and np.all(sho_points[1] == 0):
# 				pass
# 			else:
# 				if np.all(sho_points[0] == 0):
# 					sho_points[0] = rec[:2]
# 				if np.all(sho_points[1] == 0):
# 					sho_points[1] = rec[2:4]
# 				sho_points_len = int(get_length_sqrt(sho_points[0], sho_points[1]))
# 			attr.append(sho_points_len)
#
# 			# 脸部5个关键点
# 			points = np.zeros((5, 2), dtype="int")
# 			for i, point in enumerate(key_points[:5]):
# 				points[i] = ([int(point[0]), int(point[1])])
# 			attr.append(points)
#
# 			# 用非零的关键点计算脸部中心
# 			no_zero_points = points[[not np.all(points[i] == 0) for i in range(points.shape[0])], :]
# 			face_center = None
# 			if len(no_zero_points) > 0:
# 				face_center = np.int_(np.average(no_zero_points, axis=0))
# 			# 保存脸部中心
# 			attr.append(face_center)
# 			# 脸部范围
# 			face_radius = None
# 			# 鼻子，左右眼睛为脸部最重要属性，如果任意一个没有则认为该探测框脸部信息丢失，如果均存在则可以计算出脸部范围（扩大了！）
# 			if np.all(points[0] == 0) or np.all(points[1] == 0) or np.all(points[2] == 0):
# 				pass
# 			# cv2.putText(img, "no face", face_center, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
# 			else:
# 				length_list = []
# 				if not np.all(points[3] == 0):
# 					length_list.append(int(get_length_sqrt(points[3], points[1]) * 3))
# 				# length_list.append(int(math.sqrt((points[3][0] - points[1][0]) ** 2 + (points[3][1] - points[1][1]) ** 2) * 3))
# 				if not np.all(points[4] == 0):
# 					length_list.append(int(get_length_sqrt(points[4], points[2]) * 3))
#
# 				# length_list.append(int(math.sqrt((points[4][0] - points[2][0]) ** 2 + (points[4][1] - points[2][1]) ** 2) * 3))
# 				length_list.append(int(get_length_sqrt(points[1], points[2]) * 3))
# 				length_list.append(int(get_length_sqrt(points[0], points[2]) * 3))
# 				length_list.append(int(get_length_sqrt(points[0], points[1]) * 3))
# 				# length_list.append(int(math.sqrt((points[1][0] - points[2][0]) ** 2 + (points[1][1] - points[2][1]) ** 2) * 3))
# 				# length_list.append(int(math.sqrt((points[0][0] - points[2][0]) ** 2 + (points[0][1] - points[2][1]) ** 2) * 3))
# 				# length_list.append(int(math.sqrt((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2) * 3))
# 				face_radius = np.int_(np.average(length_list))
# 			# cv2.circle(img, face_center, int(face_radius * 4 / 3), (0, 255, 255), 1)
#
# 			attr.append(face_radius)
# 			attrs.append(attr)
#
# 	sho_len_index = 2
# 	face_radius_index = -1
# 	face_center_index = -2
# 	face_points_index = -3
# 	box_rec_index = 0
# 	if len(attrs) > 0:
# 		# 选取最大的区域
# 		max_radius = 0
# 		max_index = -1
# 		for i, attr in enumerate(attrs):
# 			face_radius = attr[face_radius_index] if attr[face_radius_index] is not None else -1
# 			sho_len = attr[sho_len_index] if attr[sho_len_index] is not None else -1
# 			# 脸部和肩部长度综合比较，得出最大的人脸
# 			max_len = max(int(face_radius * 8 / 3), sho_len)
# 			if max_len > max_radius:
# 				max_radius = max_len
# 				max_index = i
# 		# 如果没得到最大区域，则代表有人身，却无肩部和人脸，返回no face
# 		if max_index > -1 and attrs[max_index][face_center_index] is not None and attrs[max_index][face_radius_index] is not None:
# 			# try:
# 			# 	# 判断眼睛是否闭合
# 			# 	left_eye = attrs[max_index][face_points_index][1]
# 			# 	right_eye = attrs[max_index][face_points_index][2]
# 			# 	if not np.all(left_eye == 0) and not np.all(right_eye == 0):
# 			# 		e_length = math.sqrt((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2)
# 			# 		half_len = int(e_length / 2)
# 			# 		l_1 = get_eye_attr(left_eye, half_len, target_img,eye_model)
# 			# 		r_1 = get_eye_attr(right_eye, half_len, target_img,eye_model)
# 			# 		if l_1 == "closeEye" or r_1 == "closeEye":
# 			# 			return "eyeClose"
# 			# 	else:
# 			# 		return "poorAttention"
# 			# except BaseException as e:
# 			# 	print(e)
# 			# 	return "poorAttention"
#
#
# 			max_radius = attrs[max_index][face_radius_index]
# 			# 截取区域
# 			people_area_rec = attrs[max_index][box_rec_index]
# 			people_face_center = (attrs[max_index][face_center_index][0] - people_area_rec[0], attrs[max_index][face_center_index][1] - people_area_rec[1])
# 			people_area_img = target_img[people_area_rec[1]:people_area_rec[3], people_area_rec[0]:people_area_rec[2]]
#
# 			# 对手部进行探测
# 			hand_result = hand_model.predict(people_area_img, verbose=False)
# 			for res in hand_result:
# 				for index, box in enumerate(res.boxes.data):
# 					rec = get_box_data(box)
# 					box_center = (int(rec[0] / 2.0 + rec[2] / 2.0), int(rec[1] / 2.0 + rec[3] / 2.0))
# 					# 计算手部和脸部的距离
# 					hand_face_distance = get_length_sqrt(people_face_center, box_center)
# 					# 如果小于脸部半径，即在脸部范围内，则认为是手部动作
# 					if hand_face_distance < max_radius:
# 						# cv2.circle(img, np.array(people_area_rec[:2]) + np.array(box_center), 3, (0, 0, 255), -1)
# 						hand_box_center = np.array(people_area_rec[:2]) + np.array(box_center)
# 						left_down_point = np.array(attrs[max_index][face_center_index]) - np.array([max_radius, -max_radius])
# 						right_down_point = np.array(attrs[max_index][face_center_index]) + np.array([max_radius, max_radius])
# 						mouth_points = np.array([attrs[max_index][face_center_index], left_down_point, right_down_point])
# 						# cv2.polylines(img,[mouth_points],True,(0,0,255),1)
# 						if cv2.pointPolygonTest(mouth_points, (int(hand_box_center[0]), int(hand_box_center[1])), False) >= 0:
# 							return "smoking"
# 						else:
# 							return "usePhone"
#
# 			# face_img = target_img[attrs[max_index][1] - max_radius:attrs[max_index][1] + max_radius, attrs[max_index][0] - max_radius:attrs[max_index][0] + max_radius]
# 			face_rec = [max((people_face_center[0] - max_radius), 0), max((people_face_center[1] - max_radius), 0), (people_face_center[0] + max_radius),
# 						(people_face_center[1] + max_radius)]
#
# 			face_img = people_area_img[
# 					   face_rec[1]:face_rec[3],
# 					   face_rec[0]:face_rec[2]
# 					   ]
# 			if face_img.shape[0] > 0 and face_img.shape[1] > 0:
# 				face_result = face_model.predict(face_img, verbose=False)
# 				for res in face_result:
# 					# face_img = cv2.addWeighted(face_img,1,res.plot(),0.5,1)
# 					for index, box in enumerate(res.boxes.data):
# 						key_points = res.keypoints.data[index]
# 						points = np.zeros((68, 2), dtype="int")
# 						for i, point in enumerate(key_points):
# 							points[i] = ([int(point[0]), int(point[1])])
# 						left_eye = eye_aspect_ratio(points[36:42])
# 						right_eye = eye_aspect_ratio(points[42:48])
# 						mouth = mouth_aspect_ratio(points[48:68])
# 						_, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(face_img.shape, points)
# 						pitch, yaw, roll, pitch_degree, yaw_degree, roll_degree = get_euler_angle(rotation_vector)
# 						pitch_degree = pitch_degree - head_pitch_offset
# 						yaw_degree = yaw_degree - head_yaw_offset
# 						if (left_eye + right_eye) / 2.0 < head_eye_close:
# 							return "eyeClose"
# 						if mouth > head_yawn:
# 							return "yawn"
# 						if abs(pitch_degree) > head_pitch_degree or abs(yaw_degree) > head_yaw_degree or abs(roll_degree) > 30:
# 							return "poorAttention"
# 		else:
# 			return "poorAttention"
# 	# return "no face"
# 	else:
# 		if speed is not None and speed > 10:
# 			return "poorAttention"
# 	# return "no people"
# 	# return "normal"
# 	return ""


# def handle_img(target_img,pose_model,hand_model,face_model,speed):
#     # 第一步，姿态，不管有无遮挡，脸部都能准确识别
#     results = pose_model.predict(target_img,verbose=False)
#     attrs = []
#     # 对每一个结果都进行处理
#     for res in results:
#         # 多个探测框
#         for index, box in enumerate(res.boxes.data):
#             attr = []
#             rec = get_box_data(box)
#             # 每个探测框，对关键点进行属性运算
#             key_points = res.keypoints.data[index]
#             # 先计算面积，如果只保留一个则选用最大面积,舍弃！改用脸部和肩部综合判断
#             area = abs((rec[0] - rec[2]) * (rec[1] - rec[3]))
#             # 保存面积
#             attr.append(rec)
#             attr.append(area)
#
#             # 肩部2个关键点
#             sho_points_len = None
#             sho_points = np.zeros((2, 2), dtype="int")
#             for i, point in enumerate(key_points[5:7]):
#                 sho_points[i] = ([int(point[0]), int(point[1])])
#             if np.all(sho_points[0] == 0) and np.all(sho_points[1] == 0):
#                 pass
#             else:
#                 if np.all(sho_points[0] == 0):
#                     sho_points[0] = rec[:2]
#                 if np.all(sho_points[1] == 0):
#                     sho_points[1] = rec[2:4]
#                 sho_points_len = int(get_length_sqrt(sho_points[0], sho_points[1]))
#             attr.append(sho_points_len)
#
#             # 脸部5个关键点
#             points = np.zeros((5, 2), dtype="int")
#             for i, point in enumerate(key_points[:5]):
#                 points[i] = ([int(point[0]), int(point[1])])
#
#             # 用非零的关键点计算脸部中心
#             no_zero_points = points[[not np.all(points[i] == 0) for i in range(points.shape[0])], :]
#             face_center = None
#             if len(no_zero_points) > 0:
#                 face_center = np.int_(np.average(no_zero_points, axis=0))
#             # 保存脸部中心
#             attr.append(face_center)
#             # 脸部范围
#             face_radius = None
#             # 鼻子，左右眼睛为脸部最重要属性，如果任意一个没有则认为该探测框脸部信息丢失，如果均存在则可以计算出脸部范围（扩大了！）
#             if np.all(points[0] == 0) or np.all(points[1] == 0) or np.all(points[2] == 0):
#                 pass
#                 # cv2.putText(img, "no face", face_center, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
#             else:
#                 length_list = []
#                 if not np.all(points[3] == 0):
#                     length_list.append(int(get_length_sqrt(points[3], points[1]) * 3))
#                     # length_list.append(int(math.sqrt((points[3][0] - points[1][0]) ** 2 + (points[3][1] - points[1][1]) ** 2) * 3))
#                 if not np.all(points[4] == 0):
#                     length_list.append(int(get_length_sqrt(points[4], points[2]) * 3))
#
#                     # length_list.append(int(math.sqrt((points[4][0] - points[2][0]) ** 2 + (points[4][1] - points[2][1]) ** 2) * 3))
#                 length_list.append(int(get_length_sqrt(points[1], points[2]) * 3))
#                 length_list.append(int(get_length_sqrt(points[0], points[2]) * 3))
#                 length_list.append(int(get_length_sqrt(points[0], points[1]) * 3))
#                 # length_list.append(int(math.sqrt((points[1][0] - points[2][0]) ** 2 + (points[1][1] - points[2][1]) ** 2) * 3))
#                 # length_list.append(int(math.sqrt((points[0][0] - points[2][0]) ** 2 + (points[0][1] - points[2][1]) ** 2) * 3))
#                 # length_list.append(int(math.sqrt((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2) * 3))
#                 face_radius = np.int_(np.average(length_list))
#                 # cv2.circle(img, face_center, int(face_radius * 4 / 3), (0, 255, 255), 1)
#
#             attr.append(face_radius)
#             attrs.append(attr)
#
#     sho_len_index = 2
#     face_radius_index = -1
#     face_center_index = -2
#     box_rec_index = 0
#     if len(attrs) > 0:
#         # 选取最大的区域
#         max_radius = 0
#         max_index = -1
#         for i, attr in enumerate(attrs):
#             face_radius = attr[face_radius_index] if attr[face_radius_index] is not None else -1
#             sho_len = attr[sho_len_index] if attr[sho_len_index] is not None else -1
#             # 脸部和肩部长度综合比较，得出最大的人脸
#             max_len = max(int(face_radius * 8 / 3), sho_len)
#             if max_len > max_radius:
#                 max_radius = max_len
#                 max_index = i
#         # 如果没得到最大区域，则代表有人身，却无肩部和人脸，返回no face
#         if max_index > -1 and attrs[max_index][face_center_index] is not None and attrs[max_index][face_radius_index] is not None:
#             max_radius = attrs[max_index][face_radius_index]
#             # 截取区域
#             people_area_rec = attrs[max_index][box_rec_index]
#             people_face_center = (attrs[max_index][face_center_index][0] - people_area_rec[0], attrs[max_index][face_center_index][1] - people_area_rec[1])
#             people_area_img = target_img[people_area_rec[1]:people_area_rec[3], people_area_rec[0]:people_area_rec[2]]
#
#             # 对手部进行探测
#             hand_result = hand_model.predict(people_area_img,verbose=False)
#             for res in hand_result:
#                 for index, box in enumerate(res.boxes.data):
#                     rec = get_box_data(box)
#                     box_center = (int(rec[0] / 2.0 + rec[2] / 2.0), int(rec[1] / 2.0 + rec[3] / 2.0))
#                     # 计算手部和脸部的距离
#                     hand_face_distance = get_length_sqrt(people_face_center, box_center)
#                     # 如果小于脸部半径，即在脸部范围内，则认为是手部动作
#                     if hand_face_distance < max_radius:
#                         # cv2.circle(img, np.array(people_area_rec[:2]) + np.array(box_center), 3, (0, 0, 255), -1)
#                         hand_box_center = np.array(people_area_rec[:2]) + np.array(box_center)
#                         left_down_point = np.array(attrs[max_index][face_center_index]) - np.array([max_radius,-max_radius])
#                         right_down_point = np.array(attrs[max_index][face_center_index]) + np.array([max_radius,max_radius])
#                         mouth_points = np.array([attrs[max_index][face_center_index],left_down_point,right_down_point])
#                         # cv2.polylines(img,[mouth_points],True,(0,0,255),1)
#                         if cv2.pointPolygonTest(mouth_points, (int(hand_box_center[0]),int(hand_box_center[1])), False) >= 0:
#                             return "smoking"
#                         else:
#                             return "usePhone"
#
#             # face_img = target_img[attrs[max_index][1] - max_radius:attrs[max_index][1] + max_radius, attrs[max_index][0] - max_radius:attrs[max_index][0] + max_radius]
#             face_rec = [max((people_face_center[0] - max_radius), 0),max((people_face_center[1] - max_radius), 0),(people_face_center[0] + max_radius),(people_face_center[1] + max_radius)]
#
#             face_img = people_area_img[
#                        face_rec[1]:face_rec[3],
#                        face_rec[0]:face_rec[2]
#                        ]
#             if face_img.shape[0] > 0 and face_img.shape[1] > 0:
#                 face_result = face_model.predict(face_img,verbose=False)
#                 for res in face_result:
#                     # face_img = cv2.addWeighted(face_img,1,res.plot(),0.5,1)
#                     for index, box in enumerate(res.boxes.data):
#                         key_points = res.keypoints.data[index]
#                         points = np.zeros((68, 2), dtype="int")
#                         for i, point in enumerate(key_points):
#                             points[i] = ([int(point[0]), int(point[1])])
#                         left_eye = eye_aspect_ratio(points[36:42])
#                         right_eye = eye_aspect_ratio(points[42:48])
#                         mouth = mouth_aspect_ratio(points[48:68])
#                         _, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(face_img.shape, points)
#                         pitch, yaw, roll, pitch_degree, yaw_degree, roll_degree = get_euler_angle(rotation_vector)
#                         pitch_degree = pitch_degree - head_pitch_offset
#                         yaw_degree = yaw_degree - head_yaw_offset
#                         if (left_eye + right_eye) / 2.0 < head_eye_close:
#                             return "eyeClose"
#                         if mouth > head_yawn:
#                             return "yawn"
#                         if abs(pitch_degree) > head_pitch_degree or abs(yaw_degree) > head_yaw_degree or abs(roll_degree) > 30:
#                             return "poorAttention"
#         else:
#             return "poorAttention"
#             # return "no face"
#     else:
#         if speed is not None and speed > 10:
#             return "poorAttention"
#         # return "no people"
#     # return "normal"
#     return ""
