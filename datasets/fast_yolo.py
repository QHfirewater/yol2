
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import asyncio
import json
import base64
from typing import List
from datetime import datetime
import os

app = FastAPI()

# 加载 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 可以根据需要更换模型大小

# 创建保存视频的目录
os.makedirs("saved_videos", exist_ok=True)

# 用于管理 WebSocket 连接
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.recording = False
        self.video_writer = None
        self.recording_start_time = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    def start_recording(self, frame_width, frame_height):
        if not self.recording:
            self.recording = True
            self.recording_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"saved_videos/detection_{self.recording_start_time}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
            return video_path
        return None

    def stop_recording(self):
        if self.recording and self.video_writer is not None:
            self.recording = False
            self.video_writer.release()
            self.video_writer = None
            return True
        return False

    def write_frame(self, frame):
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)

manager = ConnectionManager()

# 提供静态文件（HTML, JS, CSS）
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/yol2", response_class=HTMLResponse)
async def get():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv8 实时检测</title>
    </head>
    <body>
        <h1>YOLOv8 实时目标检测</h1>
        <img id="videoFeed" width="640" height="480"/>
        <div>
            <button id="recordBtn">开始录制</button>
            <span id="recordingStatus">未录制</span>
        </div>
        <script>
            const videoElement = document.getElementById('videoFeed');
            const recordBtn = document.getElementById('recordBtn');
            const recordingStatus = document.getElementById('recordingStatus');
            const ws = new WebSocket('ws://' + window.location.host + '/ws');
            
            let isRecording = false;
            
            recordBtn.addEventListener('click', () => {
                isRecording = !isRecording;
                ws.send(JSON.stringify({
                    type: 'control',
                    action: isRecording ? 'start_recording' : 'stop_recording'
                }));
                recordBtn.textContent = isRecording ? '停止录制' : '开始录制';
                recordingStatus.textContent = isRecording ? '录制中...' : '未录制';
            });
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'detection') {
                    console.log(data.detections);
                } else if (data.type === 'frame') {
                    videoElement.src = 'data:image/jpeg;base64,' + data.frame;
                } else if (data.type === 'recording_status') {
                    recordingStatus.textContent = data.status;
                    if (data.video_path) {
                        alert(`视频已保存到: ${data.video_path}`);
                    }
                }
            };
        </script>
    </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # 打开摄像头
        cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 执行目标检测
            results = model(frame, verbose=False)
            
            # 绘制检测结果
            annotated_frame = results[0].plot()
            
            # 如果正在录制，写入帧
            if manager.recording:
                manager.write_frame(annotated_frame)
            
            # 转换为 JPEG 格式
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            jpeg_data = buffer.tobytes()
            
            # 转换为 base64
            frame_base64 = base64.b64encode(jpeg_data).decode('utf-8')
            
            # 获取检测结果数据
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        'class': result.names[box.cls[0].item()],
                        'confidence': float(box.conf[0].item()),
                        'bbox': box.xyxy[0].tolist()
                    })
            
            # 发送帧和检测结果
            await websocket.send_text(json.dumps({
                'type': 'frame',
                'frame': frame_base64
            }))
            
            await websocket.send_text(json.dumps({
                'type': 'detection',
                'detections': detections
            }))
            
            # 处理控制消息
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                data = json.loads(data)
                if data.get('type') == 'control':
                    if data['action'] == 'start_recording':
                        video_path = manager.start_recording(frame_width, frame_height)
                        await websocket.send_text(json.dumps({
                            'type': 'recording_status',
                            'status': '录制中...',
                            'video_path': None
                        }))
                    elif data['action'] == 'stop_recording':
                        if manager.stop_recording():
                            await websocket.send_text(json.dumps({
                                'type': 'recording_status',
                                'status': '未录制',
                                'video_path': f"saved_videos/detection_{manager.recording_start_time}.mp4"
                            }))
            except asyncio.TimeoutError:
                pass
            
            # 控制帧率
            await asyncio.sleep(0.05)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        cap.release()
        if manager.recording:
            manager.stop_recording()
    except Exception as e:
        print(f"Error: {e}")
        manager.disconnect(websocket)
        cap.release()
        if manager.recording:
            manager.stop_recording()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)