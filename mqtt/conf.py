# 云平台MQTT
import server.handle
import udp.handle
from upload.conf import CAR_ID

SERVER_MQTT = {
	"ip": "icad3ab4.ala.cn-hangzhou.emqxsl.cn",
	"port": 8883,
	"keepalive": 60,
	"ssl": True,
	"username": "zxwl",
	"password": "123@abcd",
	"on_message": {
		"top": print,
		f"auto/drive/intervene/text/{CAR_ID}":server.handle.handle_intervene_message
	}
}

# 本机MQTT
LOCAL_MQTT = {
	"ip": "127.0.0.1",
	#"ip": "223.112.18.146",
	"port": 1883,
	"keepalive": 60,
	"ssl": False,
	"username": None,
	"password": None,
	"on_message": {
		"auto/drive/param/req/monitor": udp.handle.on_message
	}
}


