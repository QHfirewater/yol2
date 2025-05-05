
import cv2
from ultralytics import YOLO

# 加载预训练的YOLO模型（会自动下载权重文件）
model = YOLO("yolov8n.pt")  # 使用YOLOv8n模型，可选yolov8s/m/l/x

# 调用摄像头（0表示默认摄像头，若有多个摄像头可尝试1,2等）
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 循环读取摄像头画面
while True:
    # 读取一帧画面
    ret, frame = cap.read()
    
    # 如果读取失败则退出
    if not ret:
        print("无法获取画面")
        break
    
    # 使用YOLO进行目标检测
    results = model(frame)
    
    # 在画面上绘制检测结果（边界框、类别、置信度）
    annotated_frame = results[0].plot()
    
    # 显示处理后的画面
    cv2.imshow("YOLO Camera Detection", annotated_frame)
    
    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
