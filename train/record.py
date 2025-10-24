import uuid

import cv2

from test import VideoReader

if __name__ == '__main__':
    video = VideoReader('0')
    for img in video:
        cv2.imwrite("E:/数据集/自用场景数据集/JPEGImages2/{}.jpg".format(uuid.uuid4()), img)
        cv2.imshow("tar",img)
        cv2.waitKey(1)
