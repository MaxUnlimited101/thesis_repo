import cv2
import torch
import torch.nn.functional as F
import requests
import time
from urllib.parse import urljoin
import uuid


# ---------------- CONFIG ----------------
MODEL_PATH = "model_efb0.pth"
ENDPOINT_URL = ""
CAPTURE_INTERVAL = 5  # seconds
# sasha's model
#CLASS_NAMES = ("anger","disgust","fear","happy","neutral","sad","surprise","contempt")
# max's model
CLASS_NAMES = ('angry', 'disgust', 'contempt', 'fear', 'happy', 'neutral', 
            'sad', 'surprise')
# ----------------------------------------


def init():
    # ask for endpoint URL
    global ENDPOINT_URL
    ENDPOINT_URL = input("Enter the Educator server URL (e.g., http://localhost:8001/): ").strip()
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")
    
    model = torch.load(MODEL_PATH, weights_only=False, map_location=device)
    model = model.to(device)
    model.eval()
    return model, device


# Preprocessing
def preprocess(frame, device='cpu'):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def predict(frame, model, device='cpu'):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = preprocess(frame, device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


def send_to_server(data):
    try:
        r = requests.post(urljoin(ENDPOINT_URL, "api/emotions"), json=data, timeout=5)
        print(f"Sent: {r.status_code}")
    except Exception as e:
        print(f"Error sending: {e}")


def main():
    print("Initializing model...")
    model, device = init()
    
    print(f"Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print(f"Camera running. Sending predictions every {CAPTURE_INTERVAL}s to {ENDPOINT_URL}")
    print("Press Ctrl+C to stop")
    
    try:
        GUID = uuid.uuid4()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Could not read frame")
                continue
            
            preds = predict(frame, model, device)
            print(f"Predictions: {preds}")
            data = {
                "id": str(GUID),
                "predictions": preds
            }
            send_to_server(data)
            
            time.sleep(CAPTURE_INTERVAL)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        print("Camera released. Goodbye!")


if __name__ == "__main__":
    main()
