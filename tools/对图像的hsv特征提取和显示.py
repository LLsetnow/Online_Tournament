import cv2
import numpy as np

# 读取图片
image_path = 'D:\\github\\Online_Tournament\\photo\\bomb\\563.jpg'  # 修改为你的图片路径
image = cv2.imread(image_path)
height, width = image.shape[:2]

# 检查图片尺寸
assert height == 240 and width == 320, "图片尺寸必须为320x240"

# 转换为HSV色彩空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 创建3张白色画布
canvas_h = np.ones((2400, 3200, 3), dtype=np.uint8) * 255
canvas_s = np.ones((2400, 3200, 3), dtype=np.uint8) * 255
canvas_v = np.ones((2400, 3200, 3), dtype=np.uint8) * 255

# 分割方块并计算平均值
block_size = 10
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (0, 0, 0)  # 黑色
grid_color = (0, 0, 255)  # 红色

for row in range(0, height, block_size):
    for col in range(0, width, block_size):
        # 获取方块区域
        block = hsv_image[row:row + block_size, col:col + block_size]

        # 计算H, S, V的平均值
        h_mean = int(np.mean(block[:, :, 0]))
        s_mean = int(np.mean(block[:, :, 1]))
        v_mean = int(np.mean(block[:, :, 2]))

        # 放大坐标
        scaled_row, scaled_col = row * 10, col * 10

        # 在每个画布上打印平均值
        cv2.putText(canvas_h, f'{h_mean}', (scaled_col + 5, scaled_row + 195), font, font_scale, text_color,
                    font_thickness)
        cv2.putText(canvas_s, f'{s_mean}', (scaled_col + 5, scaled_row + 195), font, font_scale, text_color,
                    font_thickness)
        cv2.putText(canvas_v, f'{v_mean}', (scaled_col + 5, scaled_row + 195), font, font_scale, text_color,
                    font_thickness)

        # 绘制红色方格线
        cv2.rectangle(image, (col, row), (col + block_size, row + block_size), grid_color, 1)
        cv2.rectangle(canvas_h, (scaled_col, scaled_row), (scaled_col + block_size * 10, scaled_row + block_size * 10), grid_color, 1)
        cv2.rectangle(canvas_s, (scaled_col, scaled_row), (scaled_col + block_size * 10, scaled_row + block_size * 10), grid_color, 1)
        cv2.rectangle(canvas_v, (scaled_col, scaled_row), (scaled_col + block_size * 10, scaled_row + block_size * 10), grid_color, 1)

outputFolder = "C:\\Users\\admin\\Desktop\\"

# 保存结果
cv2.imwrite(outputFolder + 'h_values.png', canvas_h)
cv2.imwrite(outputFolder + 's_values.png', canvas_s)
cv2.imwrite(outputFolder + 'v_values.png', canvas_v)
cv2.imwrite(outputFolder + 'original.png', image)
