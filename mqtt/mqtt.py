import json
import random
import threading
import time

from paho.mqtt import client as mqtt_client

from log.log import Log


class MqttClient(object):

	def __init__(self, config, log_path, log_name, queue=None, show_data=None):
		self.client_id = f'python-mqtt-subscribe-{random.randint(0, 1000)}-{time.strftime("%Y%m%d%H%M%S")}'
		self.client = mqtt_client.Client(self.client_id, userdata=show_data)
		if config is not None and type(config) == dict:
			if config.get("ssl") is not None:
				if config.get("ssl"):
					self.client.tls_set_context()
					if config.get("username") is not None and config.get("password") is not None:
						self.client.username_pw_set(config.get("username"), config.get("password"))

		self.config = config
		self.log_path = log_path
		self.log_name = log_name
		self.client.on_disconnect = self.on_disconnect
		self.client.on_message = self.on_message
		self.log_func = Log(self.log_path, self.log_name)
		self.queue = queue

	def connect(self):
		try:
			self.log_func.log(f"=================开始连接：{self.config.get('ip')}:{self.config.get('port')}====================")
			self.client.connect(self.config.get("ip"), self.config.get("port"), self.config.get("keepalive"))
			for topic in self.config["on_message"].keys():
				self.client.subscribe(topic)
				self.log_func.log(f"=================已订阅话题：{topic}！====================")
		except Exception as e:
			self.log_func.log(e)
		if self.queue is not None:
			threading.Thread(target=self.publish_listener).start()

		self.client.loop_forever()

	def on_disconnect(self, c, userdata, result):
		self.log_func.log(f"================={self.config.get('ip')}失去连接，重新链接！====================")
		self.connect()

	def on_message(self, client, userdata, message):
		if self.config["on_message"].get(message.topic) is not None:
			try:
				self.config["on_message"].get(message.topic)(client, userdata, message)
			except BaseException as e:
				self.log_func.log(f"{self.config.get('ip')} on_message error:{e}")

	def publish_listener(self):
		self.log_func.log(f'publish_listener start!')
		while True:
			try:
				if not self.queue.empty():
					publish_data = self.queue.get()
					if publish_data is not None:
						if type(publish_data) == dict:
							if publish_data.get("topic") is not None and publish_data.get("data") is not None:
								self.publish(publish_data.get("topic"), publish_data.get("data"))
			except Exception as e:
				self.log_func.log(f'publish_listener error:{e}')

	def publish(self, topic, data, qos=0):
		if type(data) == dict:
			data = json.dumps(data)
		self.client.publish(topic, payload=data, qos=qos)


def run(config, log_path, log_name, queue=None, show_data=None):
	mqtt_ins = MqttClient(config, log_path, log_name, queue, show_data)
	mqtt_ins.connect()
