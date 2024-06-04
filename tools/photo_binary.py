import cv2
import os
import glob

# 输入和输出文件夹路径
input_folder = 'D:\github\Online_Tournament\photo\\big_huan\\left_input'
output_folder = 'D:\github\Online_Tournament\photo\\big_huan\\left_output'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹内所有图像文件
image_files = glob.glob(os.path.join(input_folder, '*'))

for image_file in image_files:
    # 读取图像
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # 二值化处理
    _, binary_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # 获取图像文件名并生成输出路径
    file_name = os.path.splitext(os.path.basename(image_file))[0] + '.bmp'
    output_path = os.path.join(output_folder, file_name)

    # 保存二值化后的图像
    cv2.imwrite(output_path, binary_img)

print('所有图像已处理并保存至', output_folder)
