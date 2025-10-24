import os

import cv2
import numpy as np
from ultralytics import YOLO


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name,cv2.CAP_DSHOW)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)  # 设置分辨率
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

def reform_concat_images(image_list, line_count=6):
    temp_img = image_list.copy()
    while len(temp_img) % line_count != 0:
        temp_img.append(np.full((768, 1024, 3), np.uint8((255, 255, 255))))
    scale = int((len(temp_img) / (line_count ** 2)) * 10)
    if scale <= 20:
        y_image = []
        for i in range(0, len(temp_img), line_count):
            y_image.append(np.concatenate(temp_img[i:i + line_count], axis=1))
        return np.concatenate(y_image, axis=0)
    else:
        return reform_concat_images(image_list, line_count + 1)

if __name__ == '__main__':
    video = VideoReader('0')
    # Load a model
    models = []
    for root, dirs, files in os.walk("runs/detect"):
        for f in files:
            if f.endswith("best.pt"):
                model = YOLO("yolov8n.yaml")  # build a new model from scratch
                model = YOLO(os.path.join(root,f).replace("\\","/"))  # load a pretrained model (recommended for training)
                models.append(model)
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # model = YOLO("runs/detect/train6/weights/best.pt")  # load a pretrained model (recommended for training)
    for img in video:
        img_list = []
        for model in models:
            results = model(img)  # predict on an image
            annotated_frame = results[0].plot()
            img_list.append(annotated_frame)
        # for r in results:
        #     for b in r.boxes.data:
        #         cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 1)
        #         cv2.putText(img, model.names[int(b[5])] + ":" + str(int(b[4] * 100)), (int(b[0]), int(b[1]) - 16),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        if len(img_list) > 0:
            cv2.imshow("tar",cv2.resize(reform_concat_images(img_list),(1920,1080)) )
            cv2.waitKey(1)
        else:
            cv2.imshow("tar", img)
            cv2.waitKey(1)

