import cv2
import os
import glob

# 输入和输出文件夹路径
input_folder = 'C:\\Users\\admin\\Desktop\\BigIs\\resizeRGB_photo'
output_folder = 'C:\\Users\\admin\\Desktop\\BigIs\\output'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹内所有图像文件
image_files = glob.glob(os.path.join(input_folder, '*'))

for image_file in image_files:
    # 读取图像
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否正确加载
    if img is None:
        print(f"图像加载失败: {image_file}")
        continue

    # 二值化处理
    _, binary_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # 获取图像文件名并生成输出路径
    file_name = os.path.splitext(os.path.basename(image_file))[0] + '.bmp'
    output_path = os.path.join(output_folder, file_name)

    # 保存二值化后的图像
    success = cv2.imwrite(output_path, binary_img)
    if not success:
        print(f"二值化图像保存失败: {output_path}")

print('所有图像已处理并保存至', output_folder)
