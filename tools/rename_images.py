import re
import os
import shutil
import argparse

def set_name_start():
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
            'thief': 204,
            'tumble': 526
        }

# 分离数字和字母
def split_letters_digits(s):
    # 定义正则表达式模式，分别匹配所有字母和数字
    pattern = re.compile('([a-zA-Z]+)|([0-9]+)')

    # 使用findall方法找到所有匹配的部分
    matches = pattern.findall(s)

    # 初始化两个列表，分别用于存储字母和数字
    letters = []
    digits = []

    # 遍历所有匹配的结果，根据结果的类型分别添加到相应的列表
    for letter, digit in matches:
        if letter:
            letters.append(letter)
        if digit:
            digits.append(digit)

    # 将列表转换为字符串
    letters = ''.join(letters)
    digits = ''.join(digits)

    return letters, digits


parser = argparse.ArgumentParser(description='data')
parser.add_argument('--input', dest='input', help='The input dir of images', type=str)
parser.add_argument('--output', dest='output', default='temp', help='The output dir of images', type=str)
args = parser.parse_args()

# image_folder = args.input
# output_folder = args.output

dataset = 'tumble_1'
image_folder = f"D:\github\Online_Tournament\photo\prop"
output_folder = f"D:\github\Online_Tournament\photo\prop\images_rename"
if(os.path.exists(output_folder) == False):
    os.mkdir(output_folder)

number_list = set_name_start()
number_list['prop'] = 605
change_label = 'prop'

for f in os.listdir(image_folder):
    if(f.endswith((".jpg"))):
        name =f.split(".")[0]
        label, number = split_letters_digits(name)
        label = change_label
        original_file = os.path.join(image_folder, f)

        new_file = os.path.join(output_folder, f"{label}{number_list[label]}.{f.split('.')[1]}")
        print(new_file)
        with open(new_file, "w") as file:
            shutil.copy(original_file, new_file)
            file.close()
        if(f.endswith(".jpg")):
            number_list[label] += 1

# os.rename(image_folder, f"D:\github\Online_Tournament\my_dataset\{dataset}\images_yuan")
# os.rename(output_folder, f"D:\github\Online_Tournament\my_dataset\{dataset}\images")
print()
print(f"重命名的文件已全部保存至{output_folder}")


