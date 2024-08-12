import cv2
import numpy as np


def nothing(x):
    pass


h_min, s_min, v_min = 23, 120, 46
h_max, s_max, v_max = 70, 255, 200

# 创建一个窗口
cv2.namedWindow('HSV Threshold')

# 创建滑条用于调节H、S、V的最小和最大值
cv2.createTrackbar('H Min', 'HSV Threshold', 0, 179, nothing)
cv2.createTrackbar('S Min', 'HSV Threshold', 0, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Threshold', 0, 255, nothing)
cv2.createTrackbar('H Max', 'HSV Threshold', 179, 179, nothing)
cv2.createTrackbar('S Max', 'HSV Threshold', 255, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Threshold', 255, 255, nothing)

# 设置初始值
cv2.setTrackbarPos('H Min', 'HSV Threshold', h_min)
cv2.setTrackbarPos('S Min', 'HSV Threshold', s_min)
cv2.setTrackbarPos('V Min', 'HSV Threshold', v_min)
cv2.setTrackbarPos('H Max', 'HSV Threshold', h_max)
cv2.setTrackbarPos('S Max', 'HSV Threshold', s_max)
cv2.setTrackbarPos('V Max', 'HSV Threshold', v_max)

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头的帧
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧从BGR转换为HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 获取当前滑条的位置
    h_min = cv2.getTrackbarPos('H Min', 'HSV Threshold')
    s_min = cv2.getTrackbarPos('S Min', 'HSV Threshold')
    v_min = cv2.getTrackbarPos('V Min', 'HSV Threshold')
    h_max = cv2.getTrackbarPos('H Max', 'HSV Threshold')
    s_max = cv2.getTrackbarPos('S Max', 'HSV Threshold')
    v_max = cv2.getTrackbarPos('V Max', 'HSV Threshold')

    # 设置HSV的阈值范围
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    # 对HSV图像进行二值化
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 将结果显示出来
    cv2.imshow('Original', frame)
    cv2.imshow('HSV Threshold', mask)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
