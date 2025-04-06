import cv2  # 导入 OpenCV 库，用于图像处理和显示
import mediapipe as mp  # 导入 MediaPipe 库，用于姿势检测
import time  # 导入时间模块，用于计算 FPS

# 初始化 MediaPipe 的绘图工具和姿势检测模型
mp_drawing = mp.solutions.drawing_utils  # 用于绘制关键点和连接线
mp_pose = mp.solutions.pose  # 用于姿势检测

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头，如果有多个摄像头可以尝试 1, 2, 等

# 初始化 FPS 计算相关变量
prev_time = 0  # 用于存储上一帧的时间戳

# 使用 MediaPipe 的姿势检测模型
with mp_pose.Pose(
        min_detection_confidence=0.5,  # 检测置信度阈值，高于此值才认为检测到姿势
        min_tracking_confidence=0.5  # 跟踪置信度阈值，高于此值才继续跟踪
) as pose:
    # 进入主循环，持续读取摄像头画面
    while cap.isOpened():
        success, image = cap.read()  # 读取一帧画面
        image = cv2.flip(image, 1)
        if not success:  # 如果读取失败（如摄像头断开）
            print("无法读取摄像头画面。")
            break

        # --- 计算 FPS ---
        curr_time = time.time()  # 获取当前帧的时间戳
        dt = curr_time - prev_time  # 计算与上一帧的时间间隔（秒）
        fps = 1 / dt if dt != 0 else 0  # 计算帧率（FPS），避免除以零错误
        prev_time = curr_time  # 更新上一帧的时间戳

        # 将图像从 BGR 格式转换为 RGB 格式（MediaPipe 需要 RGB 格式）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用姿势检测模型处理图像
        results = pose.process(image_rgb)

        # 如果检测到姿势关键点，绘制关键点和连接线
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,  # 要绘制的图像
                results.pose_landmarks,  # 检测到的姿势关键点
                mp_pose.POSE_CONNECTIONS,  # 定义关键点之间的连接线
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),  # 关键点的颜色和粗细
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # 连接线的颜色和粗细
            )

        # --- 在画面上显示 FPS ---
        fps_text = "FPS: {:.0f}".format(fps)  # 格式化 FPS 值，保留两位小数
        cv2.putText(
            image,  # 要绘制的图像
            fps_text,  # 显示的文本内容
            (10, 30),  # 文本位置（左上角，距离左边 10 像素，距离顶部 30 像素）
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型
            1,  # 字体大小
            (0, 255, 0),  # 字体颜色（绿色）
            2  # 字体粗细
        )

        # 显示处理后的图像
        cv2.imshow('Real-Time Pose Detection', image)

        # 检测按键输入，如果按下 'q' 键则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Real-Time Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

# 释放摄像头资源
cap.release()

# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()