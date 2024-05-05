# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codecs
import os
import re
import time
import sys
from xml.dom import minidom

sys.path.append('PaddleDetection')
import json
import yaml
from functools import reduce
import multiprocessing
import argparse
import sys
sys.path.append('D:\github\Online_Tournament\submission\PaddleDetection')
import shutil
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


def get_test_images(infer_file):
    with open(infer_file, 'r') as f:
        dirs = f.readlines()
    images = []
    for dir in dirs:
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




def save_to_xml(c_results, result_path):
    # 创建XML根节点
    annotation = ET.Element('annotation')

    # 提取第一张图片作为例子
    if c_results["result"]:
        first_result = c_results["result"][0]
        file_name = first_result['image_id'] + ".jpg"

        # 文件名
        ET.SubElement(annotation, 'filename').text = file_name

        # 对象数量
        ET.SubElement(annotation, 'object_num').text = str(len(c_results['result']))

        # 图像尺寸（示例中使用假设值，需要根据实际情况调整）
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = '320'
        ET.SubElement(size, 'height').text = '240'

    # 每个对象的信息
    for obj in c_results["result"]:
        object_elem = ET.SubElement(annotation, 'object')
        ET.SubElement(object_elem, 'name').text = label_list[obj['type'] - 1]
        ET.SubElement(object_elem, 'difficult').text = '0'

        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(obj['x']))
        ET.SubElement(bndbox, 'ymin').text = str(int(obj['y']))
        ET.SubElement(bndbox, 'xmax').text = str(int(obj['x'] + obj['width']))
        ET.SubElement(bndbox, 'ymax').text = str(int(obj['y'] + obj['height']))

    # 使用minidom美化输出
    rough_string = ET.tostring(annotation, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml_as_string = reparsed.toprettyxml(indent="  ")

    # 保存到文件
    with open(result_path, 'w') as f:
        f.write(pretty_xml_as_string)



label_list = ['bomb', 'bridge', 'safety', 'cone', 'crosswalk', 'danger', 'evil', 'block', 'patient', 'prop', 'spy', 'thief', 'tumble']
# 97
def set_threshold_list():
    return {
        'bomb': 0.99,
        'bridge': 0.99,
        'safety': 0.99,
        'cone': 0.99,
        'crosswalk': 0.99,
        'danger': 0.99,
        'evil': 0.99,
        'block': 0.99,
        'patient': 0.99,
        'prop': 0.99,
        'spy': 0.99,
        'thief': 0.99,
        'tumble': 0.99
    }

def extract_prefix(s):
    match = re.search(r'\d', s)
    if match:
        return s[:match.start()]
    return s  # 如果没有数字，返回整个字符串

def predict_image(detector, image_list, result_path):

    for index in range(len(image_list)):
        # 检测模型图像预处理
        input_im_lst = []
        input_im_info_lst = []
        # results = []
        c_results = {"result": []}

        im_path = image_list[index]
        im, im_info = preprocess(im_path, detector.preprocess_ops)

        input_im_lst.append(im)
        input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)

        image_id = os.path.basename(im_path).split('.')[0]
        find_label = extract_prefix(image_id)

        # 检测模型预测结果
        det_results = detector.predict(inputs)

        # 检测模型写结果
        im_bboxes_num = det_results['boxes_num'][0]

        print(f"处理图像{image_id}.jpg  {index + 1}/{len(image_list)}")
        
        if im_bboxes_num > 0:
            bbox_results = det_results['boxes'][0:im_bboxes_num, 2:]
            id_results = det_results['boxes'][0:im_bboxes_num, 0]
            score_results = det_results['boxes'][0:im_bboxes_num, 1]

            for idx in range(im_bboxes_num):
                label = label_list[int(id_results[idx])]
                if label != find_label:
                    continue
                xmin = float(bbox_results[idx][0])
                xmax = float(bbox_results[idx][2])
                ymin = float(bbox_results[idx][1])
                ymax = float(bbox_results[idx][3])
                location = [xmin, ymin, xmax, ymax]

                if float(score_results[idx]) >= 0.1:

                    c_results["result"].append({"image_id": image_id,
                                                "type": int(id_results[idx]) + 1,
                                                "x": float(bbox_results[idx][0]),
                                                "y": float(bbox_results[idx][1]),
                                                "width": float(bbox_results[idx][2]) - float(bbox_results[idx][0]),
                                                "height": float(bbox_results[idx][3]) - float(bbox_results[idx][1]),
                                                "segmentation": []})

        xml_name = image_id + ".xml"
        xml_file_path = os.path.join(result_path, xml_name)
        save_to_xml(c_results, xml_file_path)

def main(infer_txt, result_path, det_model_path):
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path)

    # predict from image
    img_list = get_test_images(infer_txt)
    predict_image(detector, img_list, result_path)

def alltxt(dataset):
    base_path = f'D:\\github\\Online_Tournament\\my_dataset\\{dataset}'
    image_folder = os.path.join(base_path, 'images')

    # 检查images文件夹是否存在，如果不存在则创建
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"创建文件夹: {image_folder}")
        for filename in os.listdir(base_path):
            if filename.endswith('.jpg'):
                source_path = os.path.join(base_path, filename)
                destination_path = os.path.join(image_folder, filename)

                # 移动文件
                shutil.move(source_path, destination_path)
                print(f"移动 {filename} 到 {image_folder}")

    files = os.listdir(image_folder)
    txt_path = f"D:\github\Online_Tournament\my_dataset\\{dataset}\\all.txt"

    with open(txt_path, 'w') as f:
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(image_folder, file)
                f.write(f"{image_path}\n")
                print(f"写入图像{image_path}")
        f.close()

if __name__ == '__main__':
    start_time = time.time()
    det_model_path = "model"


    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some location')

    # 添加参数
    parser.add_argument('--model', dest='model', default='model', help='input your model path', type=str)
    parser.add_argument('--images', dest='images', help='input your dataset path', type=str)
    parser.add_argument('--output', dest='output', default='F1_Score.json', help='your output json_file path', type=str)

    # 解析命令行参数
    args = parser.parse_args()

    det_model_path = args.model
    images_txt = args.images
    result_path = args.output

    # 只要改这3个
    dataset = 'good_v4_bomb'
    model_name = 'yv982'

    # 将图像路径写入 all.txt
    alltxt(dataset)

    os.chdir(f'D:\github\Online_Tournament\my_dataset\\{dataset}')
    det_model_path = f'D:\github\Online_Tournament\model\\{model_name}'
    images_txt = f'D:\github\Online_Tournament\my_dataset\\{dataset}\\all.txt'
    result_path = f"D:\github\Online_Tournament\my_dataset\\{dataset}\\annotations"

    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    shutil.copytree(f'D:\github\Online_Tournament\my_dataset\\{dataset}\\images', result_path)
    print(f"读取模型{det_model_path}")
    print(f"读取图像路径文本{images_txt}")
    print(f"读取标注输出路径{result_path}")



    main(images_txt, result_path, det_model_path)
    print('total time:', time.time() - start_time)
