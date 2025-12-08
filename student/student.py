import cv2
import torch
import torch.nn.functional as F
import requests
import time
from urllib.parse import urljoin
import uuid
import argparse


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


def list_available_cameras(max_cameras=10):
    """Find available cameras by trying to open them"""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def select_camera():
    """Let user select a camera from available options"""
    print("\nScanning for available cameras...")
    cameras = list_available_cameras()
    
    if not cameras:
        print("Error: No cameras found!")
        return None
    
    print(f"\nAvailable cameras: {cameras}")
    
    if len(cameras) == 1:
        print(f"Using camera {cameras[0]} (only one found)")
        return cameras[0]
    
    while True:
        try:
            choice = input(f"Select camera number {cameras}: ").strip()
            camera_id = int(choice)
            if camera_id in cameras:
                return camera_id
            else:
                print(f"Invalid choice. Please select from: {cameras}")
        except (ValueError, KeyboardInterrupt):
            print(f"\nInvalid input. Please enter a number from: {cameras}")
    

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


def parse_args():
    parser = argparse.ArgumentParser(description="Emotion client for Educator server.")
    parser.add_argument(
        '--show-camera',
        '-s',
        action='store_true',
        help='Display the real-time camera feed.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Initializing model...")
    model, device = init()
    
    # Select camera
    camera_id = select_camera()
    if camera_id is None:
        return
    
    print(f"\nOpening camera {camera_id}...")
    
    # Try to open camera with retries
    max_retries = 3
    cap = None
    
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}...")
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {camera_id} opened successfully!")
                break
            else:
                print(f"✗ Camera opened but couldn't read frame")
                cap.release()
                cap = None
        else:
            print(f"✗ Failed to open camera {camera_id}")
        
        if attempt < max_retries:
            print("Retrying in 2 seconds...")
            time.sleep(2)
    
    if cap is None or not cap.isOpened():
        print(f"\nError: Could not open camera {camera_id} after {max_retries} attempts")
        return
    
    if args.show_camera:
        print("Camera view enabled. Press 'q' in the camera window to close it and stop the program.")
    
    print(f"\nCamera running. Sending predictions every {CAPTURE_INTERVAL}s to {ENDPOINT_URL}")
    print("Press Ctrl+C to stop\n")
    
    last_capture_time = time.time()
    
    try:
        GUID = uuid.uuid4()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Could not read frame")
                continue
            
            current_time = time.time()
            
            if current_time - last_capture_time >= CAPTURE_INTERVAL:
                preds = predict(frame, model, device)
                print(f"Predictions: {preds}")
                data = {
                    "id": str(GUID),
                    "predictions": preds
                }
                send_to_server(data)
                
                last_capture_time = current_time 

            if args.show_camera:
                frame = cv2.flip(frame, 1)
                status_text = f"Preds Sent: {int(current_time - last_capture_time)}s ago"
                display_frame = cv2.resize(frame, (640, 480)) 
                cv2.putText(display_frame, status_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Real-Time Camera View (Press 'q' to stop)", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        if args.show_camera:
            cv2.destroyAllWindows()
        print("Camera released. Goodbye!")


if __name__ == "__main__":
    main()
