# app.py
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from threading import Thread
import requests
import torch.nn as nn
from torchvision import models
import time

app = FastAPI()

# ---------------- CONFIG ----------------
MODEL_PATH = "joint8_oversampled_best.pt"
ENDPOINT_URL = "http://localhost:8001/api/emotions"
CAPTURE_INTERVAL = 5  # seconds
CLASS_NAMES = ["anger","disgust","fear","happy","neutral","sad","surprise","contempt"]
# ----------------------------------------

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 8)  # 8 emotion classes
# now load weights
state_dict = torch.load("joint8_oversampled_best.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
# Preprocessing
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor

def predict(frame):
    with torch.no_grad():
        inputs = preprocess(frame)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

def send_to_server(data):
    try:
        r = requests.post(ENDPOINT_URL, json=data, timeout=5)
        print("Sent:", r.status_code)
    except Exception as e:
        print("Error sending:", e)

def camera_loop():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        preds = predict(frame)
        print(preds)
        send_to_server(preds)
        time.sleep(CAPTURE_INTERVAL)

@app.on_event("startup")
def startup_event():
    Thread(target=camera_loop, daemon=True).start()

@app.get("/")
def root():
    return {"status": "camera running", "sending every": f"{CAPTURE_INTERVAL}s"}
