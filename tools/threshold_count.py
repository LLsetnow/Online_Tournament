
# 该程序用于枚举每一种目标，在不同置信度度时，检测得到的实例数量，用于计算分析合适的置信度

import codecs
import os
import time
import sys
sys.path.append('D:\github\Online_Tournament\PaddleDetection')
import json
import yaml
from functools import reduce
import multiprocessing

from PIL import Image
import cv2
import argparse
import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor


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

# 加载模型并进行图像的目标检测预测
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

    # 获取预处理操作
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



threshold_file = {}

def predict_image(detector, image_list, item, start_threshold=0.9, end_threshold=1.0, step=0.01):
    c_results = {"result": []}
    label_list = ['bomb', 'bridge', 'safety', 'cone', 'crosswalk', 'danger', 'evil', 'block', 'patient', 'prop', 'spy', 'thief', 'tumble']

    # 初始化一个空字典来存储结果，根据传入的阈值范围和步长
    thresholds = [start_threshold + x * step for x in range(int((end_threshold - start_threshold) / step) + 1)]
    count_results = {label: {f"{thresh:.2f}": 0 for thresh in thresholds} for label in label_list}
    threshold_file[item] = count_results

    for index, im_path in enumerate(image_list):
        # 检测模型图像预处理
        input_im_lst = []
        input_im_info_lst = []

        im, im_info = preprocess(im_path, detector.preprocess_ops)

        input_im_lst.append(im)
        input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)

        image_id = os.path.basename(im_path).split('.')[0]

        # 检测模型预测结果
        det_results = detector.predict(inputs)

        # 检测模型写结果
        im_bboxes_num = det_results['boxes_num'][0]

        print(f"处理图像{image_id}.jpg  {index + 1}/{len(image_list)}")

        if im_bboxes_num > 0:
            bbox_results = det_results['boxes'][0:im_bboxes_num, 2:]
            id_results = det_results['boxes'][0:im_bboxes_num, 0]
            score_results = det_results['boxes'][0:im_bboxes_num, 1]

            for thresh in thresholds:
                threshold_str = f"{thresh:.2f}"
                for idx in range(im_bboxes_num):
                    label = label_list[int(id_results[idx])]

                    if float(score_results[idx]) >= thresh:
                        count_results[label][threshold_str] += 1

    # 写入结果文件
    # with open(result_path, 'w') as ft:
    #     json.dump(c_results, ft, indent=4)

    # 写入统计文件
    # with open(result_file, 'w') as f:
    #     json.dump(count_results, f, indent=4)

def main(dataset_folder, item, det_model_path, start_threshold, end_threshold, step):
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path)

    infer_txt = os.path.join(dataset_folder, f"{item}.txt")

    # predict from image
    img_list = get_test_images(dataset_folder, infer_txt)

    # 阈值遍历的起始点和步长

    predict_image(detector, img_list, item, start_threshold, end_threshold, step)

if __name__ == '__main__':
    start_time = time.time()

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some location')
    # 添加参数
    parser.add_argument('--model', dest='model', default='model', help='input your model path', type=str)
    parser.add_argument('--dataset', dest='dataset', help='input your dataset path', type=str)
    parser.add_argument('--output', dest='output', default='threshold.json', help='your output json_file path', type=str)
    parser.add_argument('--start_threshold', dest='start_threshold', default=0.0, help='input your threshold start point', type=float)
    parser.add_argument('--end_threshold', dest='end_threshold', default=1.0, help='input your threshold end point', type=float)
    parser.add_argument('--step', dest='step', default=0.01, help='input your threshold step', type=float)

    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行提供的路径
    det_model_path = args.model             # 模型路径
    dataset_folder = args.dataset           # 数据集路径
    result_file = args.output               # 输出文件路径
    start_threshold = args.start_threshold  # 枚举阈值起点
    end_threshold = args.end_threshold      # 枚举阈值终点
    step = args.step                        # 枚举阈值步长

    paddle.enable_static()
    # infer_txt = sys.argv[1]     # data.txt
    # result_path = sys.argv[2]   # result.json

    main(dataset_folder, "train", det_model_path, start_threshold, end_threshold, step)
    main(dataset_folder, "val", det_model_path, start_threshold, end_threshold, step)
    main(dataset_folder, "all", det_model_path, start_threshold, end_threshold, step)

    print(f"图像阈值统计结果保存至{result_file}")

    # 写入统计文件
    with open(result_file, 'w') as f:
        json.dump(threshold_file, f, indent=4)

    print('total time:', time.time() - start_time)
