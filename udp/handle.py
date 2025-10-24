import json
import os


def on_message(client, userdata, msg):
	data = json.loads(msg.payload)
	if data.get("timestamp") is not None:
		try:
			if userdata["timestamp"] is None:
				userdata["timestamp"] = int(data["timestamp"])
			else:
				timestamp = int(data["timestamp"])
				if timestamp > userdata["timestamp"]:
					userdata["timestamp"] = timestamp
				else:
					return
		except ValueError:
			pass
	if data.get("ip") is not None:
		userdata["ip"] = data["ip"]

	if data.get("type") is not None:
		if data["type"] == "start":
			userdata["start_flag"] = True
		elif data["type"] == "stop":
			userdata["start_flag"] = False
	if data.get("channel1") is not None:
		try:
			if int(data["channel1"]) == 0:
				userdata["channel1"] = False
			elif int(data["channel1"]) == 1:
				userdata["channel1"] = True
		except ValueError:
			pass
	if data.get("channel2") is not None:
		try:
			if int(data["channel2"]) == 0:
				userdata["channel2"] = False
			elif int(data["channel2"]) == 1:
				userdata["channel2"] = True
		except ValueError:
			pass
	if data.get("channel3") is not None:
		try:
			if int(data["channel3"]) == 0:
				userdata["channel3"] = False
			elif int(data["channel3"]) == 1:
				userdata["channel3"] = True
		except ValueError:
			pass
	if data.get("channel4") is not None:
		try:
			if int(data["channel4"]) == 0:
				userdata["channel4"] = False
			elif int(data["channel4"]) == 1:
				userdata["channel4"] = True
		except ValueError:
			pass
	if data.get("channel5") is not None:
		try:
			if int(data["channel5"]) == 0:
				userdata["channel5"] = False
			elif int(data["channel5"]) == 1:
				userdata["channel5"] = True
		except ValueError:
			pass

