import os
import xml.etree.ElementTree as ET

# 定义函数以查找所有不包含指定标签的XML文件
def find_invalid_labels(directory, valid_labels):
    invalid_files = []

    # 遍历指定目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):  # 确保处理的是XML文件
            file_path = os.path.join(directory, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            # 检查每个<object>标签下的<name>元素
            for obj in root.iter('object'):
                name = obj.find('name').text
                if name not in valid_labels:
                    invalid_files.append(filename)
                    break  # 一旦找到无效标签，就记录并停止检查这个文件

    return invalid_files

# 定义允许的标签列表
valid_labels = ['bomb', 'bridge', 'safety', 'cone', 'crosswalk', 'danger',
                'evil', 'block', 'patient', 'prop', 'spy', 'thief', 'tumble']

# 指定要检查的文件夹路径
directory = 'D:\github\Online_Tournament\my_dataset\good_v3\\annotations'

# 调用函数并打印结果
invalid_files = find_invalid_labels(directory, valid_labels)
print("非法标签文件:")
for file in invalid_files:
    print(file)


