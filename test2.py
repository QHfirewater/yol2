import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO()

# 调用摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("摄像头无法打开")
    exit()





# 获取摄像头参数
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # 如果返回0，可手动设置（如30） 该项为帧率 

# 设置MP4编码器（关键步骤）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'avc1', 'h264'
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    try:

        ret, frame = cap.read()
        if not ret:
            print("无法读取画面")
            break
        
        # 执行检测并绘制结果
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # 写入视频文件
        out.write(annotated_frame)
        
        # 显示实时画面（可选）
        cv2.imshow("检测画面", annotated_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    except KeyboardInterrupt:
        break

cap.release()
out.release()  # 确保释放资源，否则视频可能损坏
cv2.destroyAllWindows()
print("视频已保存为 output.mp4")