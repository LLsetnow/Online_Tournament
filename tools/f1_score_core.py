import argparse
# 本程序用于计算单张图像的f1_score值 和 平均值

import codecs
import os
import random
import time
import sys
sys.path.append('D:\github\Online_Tournament\submission\PaddleDetection')
import json
import yaml
from functools import reduce
import multiprocessing

from PIL import Image
import cv2
import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

import xml.etree.ElementTree as ET

from deploy.python.preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride
from deploy.python.utils import argsparser, Timer, get_current_memory_mb


class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.tracker = None
        if 'tracker' in yml_conf:
            self.tracker = yml_conf['tracker']
        if 'NMS' in yml_conf:
            self.nms = yml_conf['NMS']
        if 'fpn_stride' in yml_conf:
            self.fpn_stride = yml_conf['fpn_stride']
        self.print_config()

    def print_config(self):
        print('%s: %s' % ('Model Arch', self.arch))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))


def get_test_images(dataset_folder, infer_file):
    with open(infer_file, 'r') as f:
        dirs = f.readlines()
    images = []
    for dir in dirs:
        dir = dir.split(' ')[0]
        dir = os.path.join(dataset_folder, dir)
        images.append(eval(repr(dir.replace('\n',''))).replace('\\', '/'))
    assert len(images) > 0, "no image found in {}".format(infer_file)
    return images

def load_predictor(model_dir):
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))
    # initial GPU memory(M), device ID
    config.enable_use_gpu(2000, 0)
    # optimize graph and fuse op
    config.switch_ir_optim(True)
    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor, config



def create_inputs(imgs, im_info):
    inputs = {}

    im_shape = []
    scale_factor = []
    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    origin_scale_factor = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    padding_imgs_shape = []
    padding_imgs_scale = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = np.array(img, dtype=np.float32)
        padding_imgs.append(padding_im)
        padding_imgs_shape.append(
            np.array([max_shape_h, max_shape_w]).astype('float32'))
        rescale = [float(max_shape_h) / float(im_h), float(max_shape_w) / float(im_w)]
        padding_imgs_scale.append(np.array(rescale).astype('float32'))
    inputs['image'] = np.stack(padding_imgs, axis=0)
    inputs['im_shape'] = np.stack(padding_imgs_shape, axis=0)
    inputs['scale_factor'] = origin_scale_factor
    return inputs


class Detector(object):

    def __init__(self,
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(model_dir)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.preprocess_ops = self.get_ops()

    def get_ops(self):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        return preprocess_ops


    def predict(self, inputs):
        # preprocess
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        np_boxes, np_boxes_num = [], []

        # model_prediction
        self.predictor.run()
        np_boxes.clear()
        np_boxes_num.clear()
        output_names = self.predictor.get_output_names()
        num_outs = int(len(output_names) / 2)


        for out_idx in range(num_outs):
            np_boxes.append(
                self.predictor.get_output_handle(output_names[out_idx])
                .copy_to_cpu())
            np_boxes_num.append(
                self.predictor.get_output_handle(output_names[
                    out_idx + num_outs]).copy_to_cpu())

        np_boxes, np_boxes_num = np.array(np_boxes[0]), np.array(np_boxes_num[0])
        return dict(boxes=np_boxes, boxes_num=np_boxes_num)

# 阈值字典
def set_threshold_list():
    return {
        'bomb': 0.18,
        'bridge': 0.21,
        'safety': 0.42,
        'cone': 0.37,
        'crosswalk': 0.22,
        'danger': 0.30,
        'evil': 0.35,
        'block': 0.18,
        'patient': 0.35,
        'prop': 0.49,
        'spy': 0.17,
        'thief': 0.23,
        'tumble': 0.43
    }

# 计数字典
def set_number_list():
    return {
        'bomb': 0,
        'bridge': 0,
        'safety': 0,
        'cone': 0,
        'crosswalk': 0,
        'danger': 0,
        'evil': 0,
        'block': 0,
        'patient': 0,
        'prop': 0,
        'spy': 0,
        'thief': 0,
        'tumble': 0
    }

# xml文件的读取
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.iter('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append({'label': label, 'bbox': [xmin, ymin, xmax, ymax]})
    return objects

# 交并比计算
def compute_iou(boxA, boxB):
    # 计算交集坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交集面积
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算各自的边界框面积
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算并集面积和 IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# 计f1_score
def evaluate_f1(true_objects, predict_objects, iou_threshold=0.5):
    TP = 0
    FP = 0
    matched_indices = set()  # 用于记录匹配上的真实对象索引

    # 对于每个预测对象，尝试找到最佳匹配的真实对象
    for pred in predict_objects:
        pred_bbox = pred['bbox']
        best_iou = 0
        best_match_idx = -1
        for idx, true in enumerate(true_objects):
            if idx in matched_indices:
                continue  # 如果这个真实对象已经被匹配，则跳过
            true_bbox = true['bbox']
            iou = compute_iou(pred_bbox, true_bbox)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = idx
        if best_iou >= iou_threshold:
            TP += 1
            matched_indices.add(best_match_idx)  # 标记这个真实对象为已匹配
        else:
            FP += 1

    FN = len(true_objects) - len(matched_indices)  # 未被任何预测匹配的真实对象

    # 计算精确度和召回率
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    # 计算 F1 分数
    F1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return F1

def append_results_to_json(image_id, f1_score, true_objects=None, predict_objects=None, file_path="F1_Score.json"):
    # 尝试读取现有的 JSON 文件
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}

    # 准备要添加的数据
    data = {"f1_score": f1_score}
    if f1_score < 1:
        data["true_objects"] = true_objects
        data["predict_objects"] = predict_objects

    # 将新的结果添加到字典中
    results[image_id] = data

    # 将更新后的字典转换为 JSON 字符串
    json_str = json.dumps(results, indent=4)

    # 写回文件
    with open(file_path, "w") as f:
        f.write(json_str)

# 绘制数据集物体框
def draw_dataset_box(true_objects, image):
    for true_object in true_objects:
        bbox = true_object['bbox']
        label = true_object['label']
        draw_predict_box(bbox, label, image)

    cv2.putText(image, "dataset", (0, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# 绘制预测框
def draw_downer_predict_box(location, label, image, threshold, flag):
    label_colors = {
        "bomb": "red",
        "bridge": "green",
        "safety": "yellow",
        "cone": "orange",
        "crosswalk": "purple",
        "danger": "pink",
        "evil": "black",
        "block": "white",
        "patient": "gray",
        "prop": "cyan",
        "spy": "magenta",
        "thief": "blue",
        "tumble": "brown"
    }

    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "blue": (0, 0, 255),
        "brown": (165, 42, 42)
    }
    # 使用标签选择颜色
    selected_color_name = label_colors.get(label, "black")  # 如果未找到标签，默认为黑色
    selected_color_rgb = colors[selected_color_name]

    xmin, ymin, xmax, ymax = map(int, location)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors["blue"], 1)
    if flag:
        cv2.putText(image, f"{label}!{threshold:.2f}", (xmin, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, selected_color_rgb, 1)
    else:
        cv2.putText(image, f"{label}/{threshold:.2f}", (xmin, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, selected_color_rgb, 1)


# 画预测框
def draw_predict_box(location, label, image):
    label_colors = {
        "bomb": "red",
        "bridge": "green",
        "safety": "yellow",
        "cone": "orange",
        "crosswalk": "purple",
        "danger": "pink",
        "evil": "black",
        "block": "white",
        "patient": "gray",
        "prop": "cyan",
        "spy": "magenta",
        "thief": "blue",
        "tumble": "brown"
    }

    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "blue": (0, 0, 255),
        "brown": (165, 42, 42)
    }
    # 使用标签选择颜色
    selected_color_name = label_colors.get(label, "black")  # 如果未找到标签，默认为黑色
    selected_color_rgb = colors[selected_color_name]

    xmin, ymin, xmax, ymax = map(int, location)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors["blue"], 1)
    cv2.putText(image, label, (xmin, ymin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, selected_color_rgb, 1)


class Result:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

def cone_filter(image):
    # 定义黄色范围的HSV阈值
    lower_hue = 25
    upper_hue = 70
    lower_yellow = (lower_hue, 80, 80)
    upper_yellow = (upper_hue, 255, 255)

    # 将图像从BGR转换到HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 黄色区域变为白色，其余区域变为黑色
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化结果列表
    results = []

    # 将mask转换为三通道的BGR图像，以便在上面绘制彩色边框
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 遍历所有轮廓
    for contour in contours:
        # 获取轮廓的边界矩形
        x, y, width, height = cv2.boundingRect(contour)

        # 将结果存储为Result对象
        results.append(Result(x, y, width, height))

        size = width * height
        print(f"锥桶大小为{size}")

        # 在mask图像上用黄色边框标记区域
        cv2.rectangle(mask_bgr, (x, y), (x + width, y + height), (0, 255, 255), 1)

    return mask_bgr, results


def predict_image(detector, image_list, result_path, annotations_path, det_model_path):
    f1_results = {"F1-AVERAGE-SCORE": [], "F1-SCORE": {"err": [], "ok": []}}
    f1_average = {"sum": 0, "num": 0, "average": 0}
    threshold_list = set_threshold_list()
    label_list = ['bomb', 'bridge', 'safety', 'cone', 'crosswalk', 'danger', 'evil', 'block', 'patient', 'prop', 'spy', 'thief', 'tumble']
    number_list = set_number_list()

    cone_state = 0
    for index in range(len(image_list)):
        # 检测模型图像预处理
        input_im_lst = []
        input_im_info_lst = []

        im_path = image_list[index]
        im, im_info = preprocess(im_path, detector.preprocess_ops)

        # 读取图像
        image = cv2.imread(im_path)
        mask_bgr, results = cone_filter(image)
        cv2.imshow("a", mask_bgr)
        cv2.imshow("b", image)
        cv2.waitKey(0)

        input_im_lst.append(im)
        input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)

        image_id = os.path.basename(im_path).split('.')[0]

        # 检测模型预测结果
        det_results = detector.predict(inputs)

        # 检测模型写结果
        im_bboxes_num = det_results['boxes_num'][0]

        # 预测结果可视化
        img_predict = cv2.imread(image_list[index])
        img_datast = cv2.imread(image_list[index])

        print(f"处理图像{image_id}  {index}/{len(image_list)}")


        if im_bboxes_num > 0:
            bbox_results = det_results['boxes'][0:im_bboxes_num, 2:]
            id_results = det_results['boxes'][0:im_bboxes_num, 0]
            score_results = det_results['boxes'][0:im_bboxes_num, 1]

            xml_id = f"{image_id.rsplit('.')[0]}.xml"
            xml_path = os.path.join(annotations_path, xml_id)

            true_objects = parse_xml(xml_path)          # 真实标签
            draw_dataset_box(true_objects, img_datast)  # 画框
            predict_objects = []                        # 预测标签
            cone_state = 0                              # 锥桶状态置零
            ignore_cones = False                        # 初始化标志，指示图像中是否存在evil或thief
            flag_cone = False                           # 是否有锥桶标志位

            # 首先检查是否需要忽略cone实例(evil 与 thief 忽略 cone)
            for idx in range(im_bboxes_num):
                label = label_list[int(id_results[idx])]
                if float(score_results[idx]) >= threshold_list[label]:
                    if label == "evil" or label == "thief":
                        ignore_cones = True
                        break  # 如果找到evil或thief，则跳出循环，因为不需要处理其余实例

            # 遍历每一个实例
            for idx in range(im_bboxes_num):
                label = label_list[int(id_results[idx])]
                xmin = float(bbox_results[idx][0])
                xmax = float(bbox_results[idx][2])
                ymin = float(bbox_results[idx][1])
                ymax = float(bbox_results[idx][3])
                location = [xmin, ymin, xmax, ymax]

                if ymin == 0:
                    threshold_list["cone"] = 0.40
                # 阈值设置
                elif cone_state:
                    threshold_list["cone"] = 0.04
                    # print("当前张有cone")
                elif ignore_cones:
                    threshold_list["cone"] = 0.99
                    print("发现evil/thief 忽略 cone")
                elif label == "cone" and ymin > 200 and xmin < 20 and ((ymax - ymin) / (xmax - xmin)) > 0.6:
                    threshold_list["cone"] = 0.01
                    print("触发角落锥桶降低阈值")

                else:
                    threshold_list["cone"] = 0.37

                if label == "cone":
                    print(f"cone 的位置为 y = {ymin} x = {xmin} 置信度为{score_results[idx]}\n")


                # 上一张图像有锥桶 降低锥桶置信度
                if float(score_results[idx]) >= threshold_list[label]:

                    # 记录锥桶状态
                    if label == "cone":
                        cone_state = 1

                    number_list[label] = number_list[label] + 1     # 实例数量统计

                    # 预测画框
                    draw_downer_predict_box(location, label, img_predict, float(score_results[idx]), 1)

                    predict_objects.append({'label': label, 'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)]})
                    # predict_objects.append({'label': label, 'bbox': {'xmin': int(xmin), 'ymin': int(ymin), 'xmax': int(xmax), 'ymax': int(ymax)}})
                    # c_results["result"].append({"image_id": image_id,
                    #                             "type": int(id_results[idx]) + 1,
                    #                             "x": float(bbox_results[idx][0]),
                    #                             "y": float(bbox_results[idx][1]),
                    #                             "width": xmax - xmin,
                    #                             "height": ymax - ymin,
                    #                             "segmentation": []})
                # elif label == "cone" :
                #     draw_downer_predict_box(location, label, img_predict, float(score_results[idx]), 0)

            duplicated_image = np.hstack((img_datast, img_predict))

            f1_score = evaluate_f1(true_objects, predict_objects)
            f1_average["sum"] += f1_score
            f1_average["num"] += 1
            if f1_score < 1:
                f1_results["F1-SCORE"]['err'].append({
                    "imged_id": image_id,
                     "f1_score": f1_score,
                    # "true": true_objects,
                    # "predict": predict_objects
                })

                # 预测画框图像保存
                folder_path = os.path.join(det_model_path, "analyse\\images\\err")
                img_path = os.path.join(folder_path, f"{image_id}.jpg")
                cv2.imwrite(img_path, duplicated_image)
            else:
                f1_results["F1-SCORE"]['ok'].append({
                    "imged_id": image_id,
                     "f1_score": f1_score
                })

                # 预测画框图像保存
                folder_path = os.path.join(det_model_path, "analyse\\images\\ok")
                img_path = os.path.join(folder_path, f"{image_id}.jpg")
                cv2.imwrite(img_path, duplicated_image)

            # f1_results["F1-SCORE"].append({
            #     "imged_id": image_id,
            #      "f1_score": f1_score
            #     })

    f1_score_ave = f1_average["sum"] / f1_average["num"]
    f1_average["average"] = f1_average["sum"] / f1_average["num"]
    f1_results["F1-AVERAGE-SCORE"].append(f1_average["average"])

    print(f"F1_Score = {f1_score_ave}")
    # 写文件
    with open(result_path, 'w') as ft:
        json.dump(f1_results, ft, indent=4)

    number_list_path = os.path.join(det_model_path, 'analyse\\number_list.json')
    with open(number_list_path, 'w') as ft:
        json.dump(number_list, ft, indent=4)

def main(dataset_folder, item, result_path, det_model_path):
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path)
    # predict from image
    infer_txt = os.path.join(dataset_folder, f"{item}.txt")
    img_list = get_test_images(dataset_folder, infer_txt)
    annotations_path = os.path.join(dataset_folder, 'annotations')
    predict_image(detector, img_list, result_path, annotations_path, det_model_path)

if __name__ == '__main__':
    start_time = time.time()
    # det_model_path = "D:\github\Online_Tournament\model\model97178_yolov3_mobilenet_v1_ssld_270e_voc"

    paddle.enable_static()
    # infer_txt = sys.argv[1]
    # result_path = sys.argv[2]

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some location')

    # 添加参数
    parser.add_argument('--model', dest='model', default='model', help='input your model path', type=str)
    parser.add_argument('--dataset', dest='dataset', help='input your dataset path', type=str)
    parser.add_argument('--output', dest='output', default='F1_Score.json', help='your output json_file path', type=str)

    # 解析命令行参数
    args = parser.parse_args()

    det_model_path = args.model
    dataset_folder = args.dataset
    result_path = args.output


    # dataset_folder = "D:\github\Online_Tournament\my_dataset\\v2"
    # result_path = "D:\github\Online_Tournament\model\model97178_yolov3_mobilenet_v1_ssld_270e_voc\F1_Score.json"

    main(dataset_folder, 'all_test', result_path, det_model_path)
    print(f'已将f1_ccore得分保存至{result_path}')
    print('total time:', time.time() - start_time)
