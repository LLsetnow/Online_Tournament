import os

# 只要修改以下三个
label = 'thief'
num = 801
dataset = 'thief_2'


folder_base = "D:\github\Online_Tournament\my_dataset"
folder_name = os.path.join(folder_base, dataset)
files = os.listdir(folder_name)
for file in files:
    if file.endswith('.jpg'):
        file_path = os.path.join(folder_name, file)
        img_name = f'{label}{num}.jpg'
        num += 1
        img_path = os.path.join(folder_name, img_name)
        os.rename(file_path, img_path)
        print(f"重命名文件{img_path}")
