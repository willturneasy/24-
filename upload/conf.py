# 当前车辆ID
CAR_ID = "苏A482JC"

# 云平台相关
BASE_PATH = "http://123.249.115.189:8123"
EVENT_INFO = "/event/uploadEventInfo"
EVENT_VIDEO = "/event/uploadEventVideo"

# 请求的头部
HEADER = {
	# "cookie": f"token={get_md5(timestamp)};",
	"Accept": "*/*",
	"Accept-Encoding": "gzip, deflate, br",
	"Accept-Language": "zh-CN,zh;q=0.9",
	"Connection": "keep-alive",
	"Content-Type": "application/json",
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
				  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
}

CHANNEL = 100