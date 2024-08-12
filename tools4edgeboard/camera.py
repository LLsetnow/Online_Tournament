import cv2

COLS_IMG = 640  # Image width
ROWS_IMG = 480  # Image height

def main():
    # Open the camera
    capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not capture.isOpened():
        print("Cannot open video device")
        return

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, COLS_IMG)  # Set image width
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, ROWS_IMG)  # Set image height

    rate = capture.get(cv2.CAP_PROP_FPS)  # Get frame rate
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # Get image width
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Get image height
    # print(f"Camera Param: frame rate = {rate} width = {width} height = {height}")

    while True:
        ret, frame = capture.read()
        if not ret:
            print("No video frame")
            continue

        # Draw grid lines
        rows = ROWS_IMG // 30  # 8
        cols = COLS_IMG // 32  # 10

        for i in range(1, rows):  # Draw horizontal lines
            cv2.line(frame, (0, 30 * i), (frame.shape[1] - 1, 30 * i), (211, 211, 211), 1)
        
        for i in range(1, cols):  # Draw vertical lines
            if i == cols // 2:
                cv2.line(frame, (32 * i, 0), (32 * i, frame.shape[0] - 1), (0, 0, 255), 2)
            else:
                cv2.line(frame, (32 * i, 0), (32 * i, frame.shape[0] - 1), (211, 211, 211), 1)

        cv2.imshow("img", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()