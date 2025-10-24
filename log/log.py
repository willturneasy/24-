import os
import time


class Log(object):
    def __init__(self, file_path, name):
        self.file_path = file_path
        # if not os.path.exists(self.file_path):
        #     os.makedirs(self.file_path)
        #     if os.path.exists(self.file_path):
        #         print("log文件路径成功创建")
        #     else:
        #         print("log文件路径创建失败")
        self.log_file_name = os.path.join(self.file_path, name)
        self.log_file = open(self.log_file_name, "a", encoding="utf-8")

    def log(self, text):
        self.log_file.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}:{text}\n')
        try:
            if self.log_file.tell() > 600 * 1024 * 1024:
                self.log_file.truncate(0)
        except OSError:
            self.terminate()
            os.remove(self.log_file_name)
            self.log_file = open(self.log_file_name, "a", encoding="utf-8")

    def terminate(self):
        self.log_file.close()
