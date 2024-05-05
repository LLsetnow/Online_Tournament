import os
import argparse

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description="images文件夹所在根目录")

# 必需参数
parser.add_argument("-i", "--input", type=str, help="传入images文件夹所在根目录")

# 解析命令行参数
args = parser.parse_args()

root_folder = args.input

# 切换到输入的根目录
os.chdir(root_folder)

# 生成train.txt
txt_dir = 'all.txt'
img_dir = 'images'

# 在开始写入前清空或创建新的txt文件
open(txt_dir, 'w').close()

# 使用追加模式写入文件，并使用os.path.join处理路径
for file in os.listdir(img_dir):
    if file.endswith(".jpg"):
        with open(txt_dir, 'a') as f:
            f.write(os.path.join(img_dir, file) + '\n')
