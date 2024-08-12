import cv2
import numpy as np

def update(val=0):
    hmin = cv2.getTrackbarPos('HMin', 'Cone Filter')
    smin = cv2.getTrackbarPos('SMin', 'Cone Filter')
    vmin = cv2.getTrackbarPos('VMin', 'Cone Filter')
    hmax = cv2.getTrackbarPos('HMax', 'Cone Filter')
    smax = cv2.getTrackbarPos('SMax', 'Cone Filter')
    vmax = cv2.getTrackbarPos('VMax', 'Cone Filter')

    lower_yellow = np.array([hmin, smin, vmin])
    upper_yellow = np.array([hmax, smax, vmax])

    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    cv2.imshow('Cone Filter', mask)

# 读取图片
image_path = 'D:\github\Online_Tournament\my_dataset\good_v10\images\evil145.jpg'
srcMat = cv2.imread(image_path)

# 检查图片是否读取成功
if srcMat is None:
    print("Could not open or find the image")
    exit()

# 将图像从BGR转换到HSV
hsv_image = cv2.cvtColor(srcMat, cv2.COLOR_BGR2HSV)

# 创建窗口
cv2.namedWindow('Cone Filter')

# 设置默认值
hmin_default, smin_default, vmin_default = 0, 127, 25
hmax_default, smax_default, vmax_default = 10, 255, 153

# 创建HSV阈值滑条
cv2.createTrackbar('HMin', 'Cone Filter', hmin_default, 359, update)
cv2.createTrackbar('SMin', 'Cone Filter', smin_default, 255, update)
cv2.createTrackbar('VMin', 'Cone Filter', vmin_default, 255, update)
cv2.createTrackbar('HMax', 'Cone Filter', hmax_default, 359, update)
cv2.createTrackbar('SMax', 'Cone Filter', smax_default, 255, update)
cv2.createTrackbar('VMax', 'Cone Filter', vmax_default, 255, update)

# 初始化滑条位置
cv2.setTrackbarPos('HMin', 'Cone Filter', hmin_default)
cv2.setTrackbarPos('SMin', 'Cone Filter', smin_default)
cv2.setTrackbarPos('VMin', 'Cone Filter', vmin_default)
cv2.setTrackbarPos('HMax', 'Cone Filter', hmax_default)
cv2.setTrackbarPos('SMax', 'Cone Filter', smax_default)
cv2.setTrackbarPos('VMax', 'Cone Filter', vmax_default)

# 初始化显示
update()

# 保持窗口打开，直到用户按下ESC键
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # ESC键
        break

cv2.destroyAllWindows()
