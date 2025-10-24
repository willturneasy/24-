#
#
#
# # # import time
# # # import multiprocessing
# # #
# # # import jetson_utils
# # # import numpy as np
# # # from concurrent.futures import ThreadPoolExecutor
# # #
# # # def get_img(s,p):
# # # 	s = jetson_utils.videoSource(s)
# # # 	start_time = time.time()
# # # 	count = 0
# # # 	while True:
# # # 		if time.time() - start_time > 1:
# # # 			print(p,count)
# # # 			count = 0
# # # 			start_time = time.time()
# # # 		s.Capture()
# # # 		count += 1
# # #
# # # #
# # # # pool = multiprocessing.Pool(processes=5)
# # # #
# # # # pool.apply_async(get_img, args=("/dev/video0",0))
# # # # pool.apply_async(get_img, args=("/dev/video2",2))
# # # # pool.apply_async(get_img, args=("/dev/video4",4))
# # # # pool.apply_async(get_img, args=("/dev/video6",6))
# # #
# # # # img_thread_pool.submit(get_img,source,0)
# # # # img_thread_pool.submit(get_img,source2,2)
# # # # img_thread_pool.submit(get_img,source4,4)
# # # # img_thread_pool.submit(get_img,source6,6)
# # # s = jetson_utils.videoSource("/dev/video0")
# # # start_time = time.time()
# # # count = 0
# # # while True:
# # # 	if time.time() - start_time > 1:
# # # 		print(count)
# # # 		count = 0
# # # 		start_time = time.time()
# # # 	s.Capture()
# # # 	count += 1
# # #
# # import subprocess as sp
# # import time
# #
# # import cv2
# # import multiprocessing
# # import numpy as np
# # def get_img(port):
# # 	command = ['ffmpeg',
# # 			   '-f', 'v4l2',
# # 			   '-input_format', 'mjpeg',
# # 			   '-framerate', '30',
# # 			   '-video_size', '1280x720',
# # 			   '-i', f'/dev/video{port}',
# # 			   '-f', 'image2pipe',
# # 			   '-pix_fmt', 'bgr24',
# # 			   '-vcodec', 'rawvideo', '-']
# # 	pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=1280 * 720 * 3 + 100)
# #
# # 	rtsp_command = [
# # 		"ffmpeg",
# # 		"-threads",
# # 		"6",
# # 		"-y",
# # 		'-f', 'rawvideo',
# # 		'-vcodec', 'rawvideo',
# # 		'-pix_fmt', 'bgr24',
# # 		"-r", str(30),
# # 		"-s", f"{1280}x{720}",
# # 		"-i", "-",
# # 		"-c:v", "h264_nvmpi",
# # 		"-pix_fmt", "yuv420p",
# # 		'-preset', 'ultrafast',
# # 		# '-g',str(VIDEO_FPS * 2),
# # 		'-f', 'flv',
# # 		# '-max_delay', '5000', '-bufsize', '500000', '-rtbufsize', '500000',
# # 		f"rtmp://106.14.10.193:1935/car/11{port}"
# # 	]
# # 	cmd2 = sp.Popen(rtsp_command, shell=False, stdin=sp.PIPE)
# # 	rtsp = cmd2.stdin
# # 	while True:
# # 		# 读取420 * 360 * 3字节（= 1帧）
# # 		raw_image = pipe.stdout.read(1280 * 720 * 3)
# # 		# img = np.ndarray(shape=(720, 1280, 3), dtype=np.uint8, buffer=raw_image)
# # 		# cv2.putText(img,str(time.time() * 1000),(200,200),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),1)
# # 		rtsp.write(raw_image)
# # 		# # 丢弃管道缓冲区中的数据。
# # 		pipe.stdout.flush()
# #
# #
# # pool = multiprocessing.Pool(processes=5)
# # for port in [0,2,4,6]:
# # 	pool.apply_async(get_img, args=(port,),error_callback=print)
# #
# # while True:
# # 	pass
#
# 上方USB接口：1-4.2:1.0
# 下方USB接口：1-4.1:1.0
# type-c以标志为上,上-下：1-4
# 1号USB接口：1-2.2:1.0
# 2号USB接口：1-2.1:1.0
# 3号USB接口：1-2.4:1.0
# 4号USB接口：1-2.3:1.0
# 另一端
# 上方USB接口：1-4.4:1.0
# 下方USB接口：1-4.3:1.0
# type-c以标志为上,上-下：1-4
# 1号USB接口：1-1.2:1.0
# 2号USB接口：1-1.1:1.0
# 3号USB接口：1-1.4:1.0
# 4号USB接口：1-1.3:1.0

# port_list = ["1-4.2:1.0","1-4.1:1.0","1-2.2:1.0",
# 			 "1-2.1:1.0","1-2.4:1.0","1-2.3:1.0",
# 			 "1-4.4:1.0","1-4.3:1.0","1-1.2:1.0",
# 			 "1-1.1:1.0","1-1.4:1.0","1-1.3:1.0","3-1:1.0"]
# e_list = []
# o_list = []
# for i in range(len(port_list) * 2):
# 	if i % 2 == 0:
# 		e_list.append(i)
# 	else:
# 		o_list.append(i)
# rules_list = []
#
# for index,port in enumerate(port_list):
# 	print(f'ACTION=="add", KERNEL=="video*",ATTR{{index}}=="0", KERNELS=="{port}",SUBSYSTEMS=="usb",MODE:="0777",SYMLINK+="video1{"0" * (2 - len(str(index * 2))) + str(index * 2)}"')
	# print(f'ACTION=="add", KERNEL=="video{o_list}", KERNELS=="{port}",SUBSYSTEMS=="usb",MODE:="0777",SYMLINK+="video1{"0" * (2 - len(str(index * 2 + 1))) + str(index * 2 + 1)}"')
# import time
#
# import cv2
#
# cap1 = cv2.VideoCapture(0,cv2.CAP_V4L2)
# cap2 = cv2.VideoCapture(2,cv2.CAP_V4L2)
# cap3 = cv2.VideoCapture(4,cv2.CAP_V4L2)
# cap4 = cv2.VideoCapture(6,cv2.CAP_V4L2)
# cap5 = cv2.VideoCapture(8,cv2.CAP_V4L2)
# cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
# cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
# cap3.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
# cap4.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
# cap5.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
#
# ret1,img1 = cap1.read()
# ret2,img2 = cap2.read()
# ret3,img3 = cap3.read()
# ret4,img4 = cap4.read()
# ret5,img5 = cap5.read()
#
# start_time = time.time()
# count = 1
# while ret1:
# 	if time.time() - start_time > 1:
# 		print(count)
# 		count = 0
# 		start_time = time.time()
#
# 	cv2.imshow("tar1",img1)
# 	cv2.imshow("tar2",img2)
# 	cv2.imshow("tar3",img3)
# 	cv2.imshow("tar4",img4)
# 	cv2.imshow("tar5",img5)
# 	cv2.waitKey(1)
# 	ret1, img1 = cap1.read()
# 	ret2, img2 = cap2.read()
# 	ret3, img3 = cap3.read()
# 	ret4, img4 = cap4.read()
# 	ret5, img5 = cap5.read()
# 	count += 1
# from udp.handle import run
#
# if __name__ == '__main__':
#     run()