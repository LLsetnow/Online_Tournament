import os
import random
import argparse

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description="用于接收数据集所在地址")

# 必需参数
parser.add_argument("-i", "--input", type=str, help="传入数据集所在地址")

# 解析命令行参数
args = parser.parse_args()

dataset_floder = args.input

os.chdir(dataset_floder)
# 生成train.txt
xml_dir = 'annotations'
img_dir = 'images'
path_list = []

for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img)
    xml_path = os.path.join(xml_dir, img.replace('jpg', 'xml'))
    path_list.append((img_path, xml_path))

random.shuffle(path_list)  # 随机打乱数据顺序

train_f = open('train.txt', 'w')
val_f = open('val.txt', 'w')

for i, content in enumerate(path_list):
    img, xml = content
    img = img.replace(os.sep, '/')
    xml = xml.replace(os.sep, '/')
    text = img + ' ' + xml + '\n'
    if i % 5 == 0:
        val_f.write(text)
    else:
        train_f.write(text)

train_f.close()
val_f.close()

print(f'生成训练集和测试集文件地址到{dataset_floder}下的 train.txt 和 val.txt')
