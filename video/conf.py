from upload.conf import CAR_ID
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
SAVE_MIN = 20
# RTMP_BASE_PATH = f"rtmp://106.14.10.193:1935/car/{CAR_ID}?sign=41db35390ddad33f83944f44b8b75ded"
RTMP_BASE_PATH = f"rtmp://106.14.10.193:1935/car/{CAR_ID}"
# 绝对路径！！！！
# PATH_TO_SAVE = r"D:\WorkSpace\dmsV3\files"
# PATH_TO_SAVE = r"D:\WorkSpace\192.168.10.107\files"
PATH_TO_SAVE = r"/media/njoak/Video2T"
# PATH_TO_SAVE = r"E:\video"
# video_port = r"D:\WorkSpace\OpenCVTest\target.mp4"
# 显示器接口一端
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
# 底部m2接口：3-1:1.0


video_port100 = "100"
video_port102 = "102"
video_port104 = "104"
video_port106 = "106"
video_port108 = "108"
# ffmpeg -y -f v4l2 -input_format mjpeg -framerate 30 -video_size 1280x720 -i /dev/video0 -c:v h264_nvmpi output1.mp4
# ffmpeg -y -f v4l2 -input_format mjpeg  -framerate 30 -video_size 1280x720 -i /dev/video2 -c:v h264_nvmpi output2.mp4
# ffmpeg -y -f v4l2 -input_format mjpeg  -framerate 30 -video_size 1280x720 -i /dev/video4 -c:v h264_nvmpi output3.mp4
# ffmpeg -y -f v4l2 -input_format mjpeg  -framerate 30 -video_size 1280x720 -i /dev/video6 -c:v h264_nvmpi output4.mp4
