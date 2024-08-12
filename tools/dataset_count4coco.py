import json
import os.path
from collections import Counter
import argparse

def count_categories(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    categories = {category['id']: category['name'] for category in data['categories']}
    category_counts = Counter(annotation['category_id'] for annotation in data['annotations'])
    category_name_counts = {categories[cat_id]: count for cat_id, count in category_counts.items() if cat_id in categories}
    return category_name_counts

def merge_counts(counts1, counts2):
    return Counter(counts1) + Counter(counts2)

def print_category_counts(category_counts):
    for category in category_order:
        count = category_counts.get(category, 0)
        print(f"{category}: {count}")

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Count category instances in JSON files.')
parser.add_argument('-i', '--input', help='Dataset version', required=True)
args = parser.parse_args()

# 使用传入的数据集版本
dataset = args.input

# 定义类别顺序
category_order = [
    'bomb', 'bridge', 'safety', 'cone', 'crosswalk', 'danger', 'evil', 'block',
    'patient', 'prop', 'spy', 'thief', 'tumble'
]

# 构建文件路径
train_json_path = os.path.join(dataset, 'train.json')
train_counts = count_categories(train_json_path)
print("train 数据集实例统计")
print_category_counts(train_counts)
print()

val_json_path = os.path.join(dataset, 'val.json')
val_counts = count_categories(val_json_path)
print("val 数据集实例统计")
print_category_counts(val_counts)
print()

# 计算总和
total_counts = merge_counts(train_counts, val_counts)
print("train 和 val 总和")
print_category_counts(total_counts)
