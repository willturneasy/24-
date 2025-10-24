import os
import random
import shutil
import time
import uuid

from ultralytics import YOLO

from train import make_path_exit


# # Load a model
# model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
#
# # Train the model
# results = model.train(data='data/classify', epochs=300, imgsz=640)

def get_model_pre_str(target_model_path, target_class_list, target_set, weights=None):
    # 判断是否存在新的模型
    if os.path.exists(target_model_path):
        print("=========================开始测试新的模型==================================")
        target_model = YOLO(target_model_path, task="classify")
        target_model_name = ""
        weights_all = 0
        result_weight = 0
        if weights is not None and len(weights) == len(target_class_list):
            for t_w in weights:
                weights_all += t_w
        for t_cls_index, t_cls in enumerate(target_class_list):
            target_results = target_model(os.path.join(target_set, f"{t_cls}/"), stream=True)
            target_all_count = 0
            target_error_count = 0
            for t_r in target_results:
                if target_model.names[t_r.probs.top1] != t_cls:
                    target_error_count += 1
                target_all_count += 1
            target_cls_pre = int(100 * (target_all_count - target_error_count) / target_all_count)
            if weights_all != 0:
                result_weight += target_cls_pre * (weights[t_cls_index] / weights_all)
            print(f"{t_cls}:{target_cls_pre}")
            target_model_name = f"{t_cls}_{target_cls_pre}_{target_model_name}"

        if weights is not None:
            return int(result_weight), target_model_name
        else:
            return target_model_name
    return None


if __name__ == '__main__':
    # 分类列表
    class_list = ["closeEye", "openEye"]
    # 分类的比重:比重/总和
    class_weight = [3, 1]
    # 存放数据集的位置
    path_to_data_set = "E:/数据集/自用场景数据集/classify"
    # 训练数据集位置
    path_to_train = "E:/PythonProject/yolov8/data/classify"
    # 结果所在位置
    path_to_result = "E:/PythonProject/yolov8/runs/classify"
    # 保存模型的位置
    path_to_save = "E:/数据集/训练的数据集"
    # 创建类型文件夹
    make_path_exit(os.path.join(path_to_save, "-".join(class_list)))
    # 训练多少个数据集
    model_count = 40
    # 是否需要更新以往的模型，在新数据加入时使用
    need_review = True
    # 每个类型用多少张图片训练[min,max)
    class_min_size, class_max_size = 666, 888
    # 训练比例,多少比例用来训练[0,1)
    train_val_percent = 0.9
    date = time.strftime("%Y-%m-%d-%H-%M-%S")
    path_to_save_list_date = os.path.join("-".join(class_list), date)
    make_path_exit(os.path.join(path_to_save, path_to_save_list_date))
    print("=========================开始构建数据集==================================")
    # 搜索存放数据集的位置，将所有的分类提取出来
    dir_list = os.listdir(path_to_data_set)
    dirs = [f for f in dir_list if os.path.isdir(os.path.join(path_to_data_set, f))]
    class_map = {}
    for d in dirs:
        if d in class_list:
            class_map[d] = []
            for root, _, files in os.walk(os.path.join(path_to_data_set, d)):
                for f in files:
                    if f.endswith(".jpg"):
                        f = os.path.join(root, f)
                        print(f"提取:{f}")
                        class_map[d].append(f)
    print("=========================是否满足所有分类==================================")
    # 判断是否满足所有分类
    has_all_cls = True
    for cls in class_list:
        if class_map.get(cls) is None:
            has_all_cls = False
            break
        else:
            print(f"{cls}：{len(class_map.get(cls))}")
    # 只有满足所有分类才训练
    if has_all_cls:
        print("=========================即将开始训练==================================")
        while model_count > 0:
            # 清空训练数据集的位置
            if os.path.exists(path_to_train):
                shutil.rmtree(path_to_train)
            make_path_exit(path_to_train)
            make_path_exit(os.path.join(path_to_train, "train/"))
            make_path_exit(os.path.join(path_to_train, "test/"))
            use_size = random.randint(class_min_size, class_max_size)
            # 划分数据集
            for cls in class_list:
                make_path_exit(os.path.join(path_to_train, f"train/{cls}/"))
                make_path_exit(os.path.join(path_to_train, f"test/{cls}/"))
                train_test_list = random.sample(class_map[cls],
                                                use_size if len(class_map[cls]) > use_size else len(
                                                    class_map[cls]))

                train_list = random.sample(train_test_list, int(train_val_percent * len(train_test_list)))
                for img in train_test_list:
                    if img in train_list:
                        shutil.copy(img, os.path.join(path_to_train, f"train/{cls}/"))
                    else:
                        shutil.copy(img, os.path.join(path_to_train, f"test/{cls}/"))
            # 保存原先的文件夹目录
            result_dir_list = os.listdir(path_to_result)
            result_dirs = [f for f in result_dir_list if os.path.isdir(os.path.join(path_to_result, f))]
            # 开始训练
            model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
            results = model.train(data='data/classify', epochs=random.randint(20, 300), imgsz=640)
            # 读取新的文件夹目录
            new_result_dir_list = os.listdir(path_to_result)
            # 比较两个文件目录差异
            new_result_dirs = [f for f in new_result_dir_list if
                               (os.path.isdir(os.path.join(path_to_result, f)) and f not in result_dirs)]
            # 只有新增一个时才是新的
            if len(new_result_dirs) == 1:
                model_path = os.path.join(path_to_result, f'{new_result_dirs[0]}/weights/best.pt')
                model_name = get_model_pre_str(model_path, class_list, path_to_data_set)
                if model_name is not None:
                    shutil.copy(model_path,
                                os.path.join(path_to_save, f'{path_to_save_list_date}/{model_name}_{uuid.uuid4()}.pt'))
            model_count -= 1

        # 测验以往所有的模型，
        if need_review:
            all_path = os.path.join(path_to_save, os.path.join("-".join(class_list), "all"))
            make_path_exit(all_path)
            # model_weight_list = []
            # model_name_list = []

            for root, _, files in os.walk(os.path.join(path_to_save, "-".join(class_list))):
                for f in files:
                    if f.endswith(".pt"):
                        f = os.path.join(root, f)
                        model = YOLO(f, task="classify")
                        # if model.names.keys()
                        class_check = True
                        for key in model.names:
                            if model.names[key] not in class_list:
                                class_check = False
                                break

                        if class_check:
                            class_check = len(model.names.keys()) == len(class_list)

                        if class_check:
                            w, model_name = get_model_pre_str(f, class_list, path_to_data_set, weights=class_weight)
                            # # model_name = os.path.join(all_path, f'{model_name}_{uuid.uuid4()}.pt')
                            # need_add = True
                            # for index, nn in enumerate(model_weight_list):
                            #     if w > nn:
                            #         model_weight_list.insert(index, w)
                            #         model_name_list.insert(index, f'{f}&{model_name}')
                            #         need_add = False
                            #         break
                            # if need_add:
                            #     model_weight_list.append(w)
                            #     model_name_list.append(f'{f}&{model_name}')

                            if model_name is not None:
                                shutil.move(f, os.path.join(all_path, f'{w}_{model_name}_{uuid.uuid4()}.pt'))
            # for index, name_list in enumerate(model_name_list):
            #     name_list = name_list.split("&")
            #     shutil.move(name_list[0], os.path.join(all_path, f'{index}_{name_list[1]}_{uuid.uuid4()}.pt'))
