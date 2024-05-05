import os
import xml.etree.ElementTree as ET
import json
import argparse

# 用于统计数据集内实例个数

# 创建两个字典，分别用于存储验证集和训练集的标签计数
def set_label_count():
    return{
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
        'tumble': 0,
    }

val_label_count = set_label_count()
train_label_count = set_label_count()
all_label_count = set_label_count()

# 准备一个字典来存储最终结果
results = {
    "train": train_label_count,
    "val": val_label_count,
    "all":all_label_count
}

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Process some integers.')
# 添加 -i 参数
parser.add_argument('-i', '--input', help='Input dataset folder', required=True)

# 解析命令行参数
args = parser.parse_args()

# 使用命令行提供的路径
dataset_folder = args.input

# 设置数据集文件夹的路径
# dataset_folder = "D:\\github\\Online_Tournament\\my_dataset\\v2"
os.chdir(dataset_folder)

# 分别设置验证集和训练集文件的名称
file_val = "val.txt"
file_train = "train.txt"

# 定义一个函数，用于读取 XML 文件并更新标签计数
def read_xml(path, label_count):
    root = ET.parse(path).getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name in label_count:
            label_count[name] += 1
            all_label_count[name] += 1
        else:
            print(f"Warning: '{name}' not found in label dictionary.")

# 处理验证集文件
with open(file_val, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line:
            _, annotation_path = line.split()
            read_xml(annotation_path, val_label_count)

# 处理训练集文件
with open(file_train, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line:
            _, annotation_path = line.split()
            read_xml(annotation_path, train_label_count)


output_file = "count.json"
# 使用 'with open' 结构来打开（或创建）文件，并将其命名为 'label_counts.json'，设置模式为 'w' 表示写入模式
with open(output_file, 'w') as file:
    # 使用 json.dump() 方法将结果字典转换为 JSON 格式并写入文件
    # ensure_ascii=False 允许写入非 ASCII 字符，indent=4 为输出格式添加缩进，使其更易于阅读
    json.dump(results, file, ensure_ascii=False, indent=4)

print(f"已将统计数据保存至{dataset_folder}\\{output_file}")