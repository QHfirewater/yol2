from ultralytics import YOLO
import cv2
import os

def process_video(input_path, output_path, conf_threshold=0.5):
    # 初始化模型
    model = YOLO()  # 使用官方预训练模型
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用'XVID'等编码器
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 执行推理
        results = model(frame, conf=conf_threshold)
        
        # 绘制检测结果
        annotated_frame = results[0].plot()
        
        # 写入处理后的帧
        out.write(annotated_frame)
        
        # 实时显示（可选）
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_input = r"C:\Users\PC\Desktop\mp4\9a4b93a84cc90dfc610173dc62aa038f.mp4"
    video_output = "output_video.mp4"
    process_video(video_input, video_output, conf_threshold=0.5)