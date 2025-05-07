# app.py
from flask import Flask, Response
import cv2
from ultralytics import YOLO
import threading
import time
import os
import torch

app = Flask(__name__)
model = YOLO("yolo11n.pt")  # 加载模型
model.to('cuda') if torch.cuda.is_available() else None  # GPU加速

# 全局变量控制视频保存
is_recording = False
video_writer = None
last_write_time = time.time()

def video_processing():
    global is_recording, video_writer, last_write_time
    
    cap = cv2.VideoCapture(0)  # 摄像头/RTSP/视频文件
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 视频保存参数
    fps = 20.0
    frame_size = (640, 480)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLOv8 检测
        results = model(frame,verbose=False)
        annotated_frame = results[0].plot()
        
        
        # 自动创建视频写入器（每小时分割文件）
        current_time = time.time()
        if is_recording and (current_time - last_write_time > 10 or video_writer is None):
            if video_writer is not None:
                video_writer.release()
            filename = f"recordings/output_{time.strftime('%Y%m%d_%H%M%S')}.avi"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            video_writer = cv2.VideoWriter(filename, codec, fps, frame_size)
            last_write_time = current_time
        
        # 写入视频文件
        if is_recording and video_writer is not None:
            video_writer.write(annotated_frame)
        
        # 缓存最新帧用于网页显示
        global latest_frame
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        latest_frame = buffer.tobytes()

# 启动后台处理线程
processing_thread = threading.Thread(target=video_processing)
processing_thread.daemon = True
processing_thread.start()

@app.route('/')
def index():
    return '''
   
<html>
<body>
    <h1>YOLOv8实时检测</h1>
    <img src="/video_feed">
    <button id="recordBtn" onclick="toggleRecording()" style="padding: 10px 20px; margin: 10px;">
        开始录制
    </button>
    <div id="statusMsg" style="color: green;"></div>
    <script>
        const btn = document.getElementById('recordBtn');
        const statusMsg = document.getElementById('statusMsg');

        async function toggleRecording() {
            btn.disabled = true; // 防止重复点击
            btn.innerText = '请求中...';
            
            try {
                const response = await fetch('/toggle_record');
                const text = await response.text();
                const isRecording = text.includes('True'); // 后端返回状态解析
                
                btn.innerText = isRecording ? '停止录制' : '开始录制';
                statusMsg.innerText = `状态：${text}`;
                statusMsg.style.color = isRecording ? 'red' : 'green';
                
            } catch (error) {
                statusMsg.innerText = '请求失败，请检查网络连接';
                statusMsg.style.color = 'red';
                btn.innerText = '开始录制'; // 恢复按钮状态
            } finally {
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
    '''

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if 'latest_frame' in globals():
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       latest_frame + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_record')
def toggle_record():
    global is_recording
    is_recording = not is_recording
    return f"Recording: {is_recording}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)