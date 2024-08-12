import os

import cv2
import numpy as np
from scipy.optimize import differential_evolution



images = [file for file in os.listdir("C:\\Users\\admin\Desktop\hsv") if file.endswith("'jpg")]


def filter_and_evaluate(params):
    hmin, smin, vmin, hmax, smax, vmax = params
    lower_yellow = np.array([hmin, smin, vmin])
    upper_yellow = np.array([hmax, smax, vmax])

    total_score = 0
    for image in images:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # 找到轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 评估每个轮廓的面积
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:  # 假设锥桶的面积大于500
                total_score += 1  # 增加得分，表示找到一个大的锥桶
            else:
                total_score -= 10  # 减少得分，表示找到一个小的噪点

    # 打印当前参数和对应的分数
    print(f'Params: HMin={hmin}, SMin={smin}, VMin={vmin}, HMax={hmax}, SMax={smax}, VMax={vmax}, Score={-total_score}')

    return -total_score  # 目标是最大化得分，所以返回负值进行最小化


# 定义HSV范围
bounds = [(0, 179), (0, 255), (0, 255), (0, 179), (0, 255), (0, 255)]

# 使用差分进化算法进行优化
result = differential_evolution(filter_and_evaluate, bounds, strategy='best1bin', maxiter=100, popsize=15)

# 获取最佳阈值
optimal_hsv = result.x
print(f'Optimal HSV thresholds: {optimal_hsv}')

# 使用最佳阈值生成最终的掩码图像
hmin, smin, vmin, hmax, smax, vmax = optimal_hsv
lower_yellow = np.array([hmin, smin, vmin])
upper_yellow = np.array([hmax, smax, vmax])

for idx, image in enumerate(images):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    result_path = f'D:\\github\\Online_Tournament\\photo\\result_{idx}.png'
    cv2.imwrite(result_path, mask)
    print(f'Saved mask to {result_path}')