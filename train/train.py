import os
import random
import shutil
import xml.etree.ElementTree as ET

from ultralytics import YOLO


def get_can_use_dir(all_class, is_alone=False):
    print("=========================开始寻找数据集==================================")
    path = "E:/数据集/分类后的数据集"
    dir_list = os.listdir(path)
    dirs = [f for f in dir_list if os.path.isdir(os.path.join(path, f))]
    can_use_xml = []
    for d in dirs:
        can_use = True
        if is_alone:
            if d not in all_class:
                can_use = False
        else:
            for c in d.split("-"):
                if c not in all_class:
                    can_use = False
                    break
        if can_use:
            for root, _, files in os.walk(os.path.join(path, d)):
                for f in files:
                    if f.endswith(".xml"):
                        f = os.path.join(root, f)
                        can_use_xml.append(f)
    print("{}个xml文件可用".format(str(len(can_use_xml))))
    print("=========================结束寻找数据集==================================")
    return can_use_xml


def get_val_xml():
    path = "E:/数据集/自用场景数据集/Annotations"
    val_xml = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".xml"):
                f = os.path.join(root, f)
                val_xml.append(f)
    return val_xml


def vail_class():
    print("=========================开始验证数据集==================================")
    path = "E:/PythonProject/yolov8/data/Annotations"
    class_map = {}
    for f in os.listdir(path):
        if f.endswith(".xml"):
            tree = ET.parse(os.path.join(path, f))
            root = tree.getroot()
            for name in root.findall(".//name"):
                name_txt = name.text
                if name_txt in class_map:
                    class_map[name_txt] += 1
                else:
                    class_map[name_txt] = 1
    print(class_map)
    print("=========================结束验证数据集==================================")
    return len(class_map.keys())


def make_path_exit(file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)


def clear_and_add_xml(add_xml_list):
    print("=========================开始构建数据集==================================")
    if os.path.exists("E:/PythonProject/yolov8/data/Annotations"):
        shutil.rmtree("E:/PythonProject/yolov8/data/Annotations")
    make_path_exit("E:/PythonProject/yolov8/data/Annotations")
    if os.path.exists("E:/PythonProject/yolov8/data/images"):
        shutil.rmtree("E:/PythonProject/yolov8/data/images")
    make_path_exit("E:/PythonProject/yolov8/data/images")
    for x in add_xml_list:
        print("复制:{}".format(os.path.basename(x)))
        shutil.copy(x, "E:/PythonProject/yolov8/data/Annotations/")
        shutil.copy(x.replace("Annotations", "JPEGImages").replace(".xml", ".jpg"),
                    "E:/PythonProject/yolov8/data/images/")
    print("=========================结束构建数据集==================================")


def split_date_set(val_xml_list=None):
    print("=========================开始划分数据集==================================")
    total_xml = os.listdir("E:/PythonProject/yolov8/data/Annotations")
    if val_xml_list is not None:
        for xml in val_xml_list:
            total_xml.remove(os.path.basename(xml))

    total_len = len(total_xml)
    total_range = range(total_len)

    # 训练和验证用全部90%
    train_val_percent = 0.9
    train_val_list = random.sample(total_range, int(train_val_percent * total_len))
    train_val_file = open("data/ImageSets/trainval.txt", "w")
    if val_xml_list is not None:
        train_list = train_val_list
    else:
        # 训练用训练和验证的90%
        train_percent = 0.9
        train_list = random.sample(train_val_list, int(int(train_percent * total_len) * train_percent))

    train_file = open("data/ImageSets/train.txt", "w")
    test_file = open("data/ImageSets/test.txt", "w")
    val_file = open("data/ImageSets/val.txt", "w")

    for i in total_range:
        name = total_xml[i][:-4] + '\n'
        print("划分id:{}".format(name))
        if i in train_val_list:
            train_val_file.write(name)
            if i in train_list:
                train_file.write(name)
            else:
                val_file.write(name)
        else:
            test_file.write(name)

    if val_xml_list is not None:
        for xml in val_xml_list:
            name = os.path.basename(xml)[:-4] + '\n'
            train_val_file.write(name)
            val_file.write(name)

    train_val_file.close()
    train_file.close()
    test_file.close()
    val_file.close()

    print("=========================结束划分数据集==================================")


# 进行归一化操作
def convert(size, box):  # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
    dw = 1. / size[0]  # 1/w
    dh = 1. / size[1]  # 1/h
    x = (box[0] + box[1]) / 2.0  # 物体在图中的中心点x坐标
    y = (box[2] + box[3]) / 2.0  # 物体在图中的中心点y坐标
    w = box[1] - box[0]  # 物体实际像素宽度
    h = box[3] - box[2]  # 物体实际像素高度
    x = x * dw  # 物体中心点x的坐标比(相当于 x/原图w)
    w = w * dw  # 物体宽度的宽度比(相当于 w/原图w)
    y = y * dh  # 物体中心点y的坐标比(相当于 y/原图h)
    h = h * dh  # 物体宽度的宽度比(相当于 h/原图h)
    return x, y, w, h  # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]


def xml_to_labels(all_class):
    print("=========================开始转化labels==================================")
    txt_list = ["train", "test", "val"]
    for txt in txt_list:
        image_ids = open('data/ImageSets/%s.txt' % txt).read().strip().split()
        list_file = open('data/main/%s.txt' % txt, 'w')
        for image_id in image_ids:
            out_file = open('data/labels/%s.txt' % image_id, 'w')
            tree = ET.parse("data/Annotations/{}.xml".format(image_id))
            root = tree.getroot()
            size = root.find("size")
            can_add = False
            if size is not None:
                # 获得宽
                w = int(size.find('width').text)
                # 获得高
                h = int(size.find('height').text)
                # 遍历目标obj
                for obj in root.iter('object'):
                    # 获得difficult ？？
                    difficult = obj.find('difficult')
                    if difficult is not None:
                        # 如果difficult==1则跳过
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            continue

                    # 获得类别 =string 类型
                    cls = obj.find('name')
                    if cls is not None:
                        can_add = True
                        cls = cls.text
                        # 通过类别名称找到id
                        cls_id = all_class.index(cls)
                        # 找到bndbox 对象
                        xmlbox = obj.find('bndbox')
                        # 获取对应的bndbox的数组 = ['xmin','xmax','ymin','ymax']
                        b = (
                            float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                            float(xmlbox.find('ymin').text),
                            float(xmlbox.find('ymax').text))
                        print(image_id, cls, b)
                        # 带入进行归一化操作
                        # w = 宽, h = 高， b= bndbox的数组 = ['xmin','xmax','ymin','ymax']
                        bb = convert((w, h), b)
                        # bb 对应的是归一化后的(x,y,w,h)
                        # 生成 calss x y w h 在label文件中
                        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

            if can_add:
                list_file.write('E:/PythonProject/yolov8/data/images/%s.jpg\n' % image_id)

    print("=========================结束转化labels==================================")


if __name__ == '__main__':
    # 构架数据集
    # 所包含的类别，配置文件中也要保持一致
    # names = ["smoke", "phone", "face"]
    names = ["closed_eye", "closed_mouth", "open_eye", "open_mouth"]
    # 获取到所有类别的xml文件,is_alone为True时，只使用单一的，False则使用复合的（class1-class2）
    # xml_list = get_can_use_dir(names, is_alone=True)
    xml_list = get_can_use_dir(names)
    val_list = None
    use_self_val = False
    if use_self_val:
        val_list = get_val_xml()

    # 当数据量小于5000的时候不训练
    if len(xml_list) < 2000:
        print("数据量不足")
    else:
        train_count = 0
        # 随机300次,主要为了随机测试数据集,用随机来减少标注的错误
        while train_count < 300:
            # 数据量充足，从xml中获取3000到5000的xml
            # a_xml = random.sample(xml_list, random.randint(10000, 12000))
            a_xml = xml_list
            # 如果有自用数据集，也拉进来
            if use_self_val and val_list is not None:
                a_xml.extend(val_list)
            # 清除之前的xml，将新的xml复制进来
            clear_and_add_xml(a_xml)
            # 验证下类型的数量，要与配置文件中保持一致
            if vail_class() == len(names):
                print("验证通过")
                # 划分数据集，分为train和val及其他
                # 下一步的思路需要标注当前试验场景的验证集（数量300起步），这样才能选出适用于项目的最佳数据模型
                # 使用最后300张自用场景照片作为准确率验证，其余用于训练
                # split_date_set(val_list[-300:])
                temp_val_list = None
                if val_list is not None:
                    temp_val_list = random.sample(val_list, 333)
                    temp_val_list.extend(random.sample(val_list, 333))
                    temp_val_list = list(set(temp_val_list))
                    # 使用随机333张自用场景照片作为准确率验证，其余用于训练
                split_date_set(temp_val_list)
                # 转化为labels用于训练
                xml_to_labels(names)
                # 加载模型,选择测试模式
                model = YOLO("yolov8x.pt", task="train")
                # 训练300次,训练一次至少3小时,目前最高识别率:map50:71%,清研微视目前识别率为85%，使用多数据帧的方式提升至90
                # 使用自用数据集后识别率提升至96%，对个别特定角度识别不出
                model.train(data="data/smoke.yaml", batch=6, epochs=300, imgsz=640, device=0, workers=8,
                            single_cls=False, )
            train_count += 1
