import json


def handle_intervene_message(client, userdata, msg):
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
	if data.get("context") is not None:
		userdata["content"] = data["context"]
		userdata["need_control"] = True
	print(data)
