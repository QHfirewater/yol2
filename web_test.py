from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolo11n.pt")  # 加载YOLOv8模型

def generate_frames():
    cap = cv2.VideoCapture(0)  # 打开摄像头（索引0为默认摄像头）
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # YOLOv8推理
        results = model(frame)  
        annotated_frame = results[0].plot()  # 绘制检测框
        # 编码为JPEG格式
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # 调整压缩质量
        frame_bytes = buffer.tobytes()
        # 生成MJPEG流格式（注意边界符）
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)  # 启用多线程支持