
import torch
import cv2
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
res = model.train(data = r'D:\reinforce\yolo\datasets\mycoco.yaml',epochs=100,device = [0])





