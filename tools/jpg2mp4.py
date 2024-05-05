# 将图片集转视频，用于capture类

import cv2
import glob
import os

# 设置你的图片所在的文件夹路径
folder_path = 'D:\\github\\Online_Tournament\\my_dataset\\v1\\images'  # 替换为你的文件夹路径
# 设置输出视频的路径
output_video_path = 'D:\\github\\Online_Tournament\\photo\\videos\\output_video4.mp4'

# 获取文件夹内所有的.JPG文件
image_files = glob.glob(os.path.join(folder_path, '*.JPG'))

# 根据文件名排序（确保是从1.jpg开始）
# image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

# 读取第一张图片来获取视频的分辨率
frame = cv2.imread(image_files[0])
height, width, _ = frame.shape

# 定义视频编码器和视频输出
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_path, fourcc, 1, (width, height))

# 将每张图片添加到视频中
for image_file in image_files:
    frame = cv2.imread(image_file)
    video.write(frame)

# 释放资源
video.release()
print('视频创建完成')
