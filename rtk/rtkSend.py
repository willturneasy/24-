import base64
import math
import socket
import time

import serial  # 导入模块
from log.log import Log



class RTKSend(object):
    def __init__(self):
        # 串口相关
        # 超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
        self.timex = 0
        # 波特率，标准值之一：50,75,110,134,150,200,300,600,1200,1800,2400,4800,9600,19200,38400,57600,115200
        self.bps = 115200
        # 端口，GNU / Linux上的/ dev / ttyUSB0 等 或 Windows上的 COM3 等
        self.portx = "/dev/rtk"
        # self.portx="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A400fueI-if00-port0"
        self.ser = None

        # 差分服务相关
        self.conn = socket.socket()
        # IP地址与端口号
        # self.ip, self.port = '140.143.212.42', 8002
        self.ip, self.port = '120.253.239.161', 8002
        # 设置ntrip账号
        self.ntrip_account = 'abdc3499'
        # 设置ntrip密码
        self.ntrip_password = '00fx5dkd'
        # 设置挂载点
        self.mountPoint = 'RTCM33_GRCEJ'
        # 将差分账号和密码进行base64编码
        self.account_password = base64.b64encode((self.ntrip_account + ":" + self.ntrip_password).encode('utf-8')).decode('utf-8')

        self.content = "GET /" + self.mountPoint + " HTTP/1.1\r\nHost: " + self.ip + "\r\nAuthorization: Basic " + self.account_password + \
                       "\r\nUser-Agent: NTRIP NtripClient/1.0.0\r\nAccept: */*\r\nConnection: close\r\n\r\n"

        self.stop_thread = False
        self.log = Log("logs", "RTKSend.out")

    def connect_usb(self):
        # 打开串口，并得到串口对象
        self.ser = serial.Serial(self.portx, self.bps, timeout=self.timex)

    def read_tcp_write_usb(self):
        # 链接差分服务
        self.conn.connect((self.ip, self.port))
        # 发送登录账号
        self.conn.send(self.content.encode('utf-8'))
        # 链接串口
        self.connect_usb()
        # 接收差分服务返回数据
        data = self.conn.recv(1024)
        # 如果登录成功则进入
        if b'ICY 200 OK' in data:
            self.log.log("登录成功")
            need_local = True
            for i in range(3):
                gga = self.ser.readall()
                # 如果有信息则，对比格式，开头正确直接透传
                if len(gga) > 0:
                    if list(gga)[:6] == list(b'$GPGGA'):
                        self.log.log(f'gga:{gga}')
                        self.conn.send(gga)
                        need_local = False
                        break
            if need_local:
                # 没有从串口中读取到定位信息，代码生成
                lon, lat = 121.21465, 31.285769
                lat = math.floor(lat) * 100 + (lat - math.floor(lat)) * 60
                lon = math.floor(lon) * 100 + (lon - math.floor(lon)) * 60
                ggaTime = time.strftime("%H%M%S", time.gmtime(time.time()))
                gga = "GPGGA," + ggaTime + "," + str(lat) + ",N," + str(lon) + ",E,3,15,2,100,M,2,M,6,0"
                # 计算校验码
                checkNum = ord(gga[0])
                for i in range(1, len(gga)):
                    checkNum = checkNum ^ ord(gga[i])
                gga = "$" + gga + "*" + hex(checkNum)[2:].upper() + "\r\n"
                self.log.log(f'gga:{gga}')
                self.conn.send(gga.encode('utf-8'))

            # 开始接收数据
            rtcm = self.conn.recv(1024)
            self.log.log(f'rtcm:{rtcm}')
            while rtcm != b'' and not self.stop_thread:
                # 读取串口信息
                gga = self.ser.readall()
                # 如果有信息则，对比格式，开头正确直接透传
                if len(gga) > 0:
                    if list(gga)[:6] == list(b'$GPGGA'):
                        self.log.log(f'gga:{gga}')
                        self.conn.send(gga)
                # 持续接收差分数据,不管串口是否有数据返回
                rtcm = self.conn.recv(1024)
                self.log.log(f'rtcm:{rtcm}')
                # 将结果写入串口
                self.ser.write(rtcm)
        if not self.stop_thread:
            time.sleep(10)
            self.read_tcp_write_usb()

    def terminate(self):
        self.stop_thread = True
        self.log.terminate()


def run():
    rtk_s = RTKSend()
    rtk_s.read_tcp_write_usb()


