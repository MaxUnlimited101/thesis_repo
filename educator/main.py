import cv2
import torch
import torch.nn.functional as F
import requests
import time
from urllib.parse import urljoin
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import uvicorn
import pinggy
from contextlib import asynccontextmanager
from asyncio import Lock
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
from fastapi.staticfiles import StaticFiles
import webbrowser
import threading


# ============================================================================
# SHARED CONFIGURATION
# ============================================================================
MODEL_PATH = "model_efb0.pth"
CLASS_NAMES = ('angry', 'disgust', 'contempt', 'fear', 'happy', 'neutral', 
               'sad', 'surprise')
CAPTURE_INTERVAL = 5  # seconds
TUNNEL_URL = None
CLIENT_MODE = False
CLIENT_CONFIG = {}

FACE_EXTENSION_Y=50
FACE_EXTENSION_X=15


# ============================================================================
# SERVER (EDUCATOR) CODE
# ============================================================================
TOKEN = ""
predictions_log = []
predictions_lock = Lock()

colors = {
    'neutral': "#808991", 'happy': "#FFF700", 'sad': "#0B4CB3",
    'surprise': "#51B3F0", 'fear': "#1F6D2E", 'disgust': "#79299C",
    'angry': "#CF0E0E", 'contempt': "#C3A02F"
}

def init_server():
    """Initialize server by reading token"""
    try:
        with open("token.txt", "r") as f:
            global TOKEN
            TOKEN = f.read().strip()
            print(f"Token loaded: {TOKEN}")
    except FileNotFoundError:
        print("ERROR: token.txt not found. Exiting...")
        exit(1)


def setup_tunnel(token):
    """Start HTTP tunnel forwarding to localhost:8001"""
    tunnel = pinggy.start_tunnel(forwardto="localhost:8001", token=token)
    global TUNNEL_URL
    TUNNEL_URL = tunnel.urls[0]
    print(f"\n{'='*60}")
    print(f"🌐 TUNNEL URL: {TUNNEL_URL}")
    print(f"{'='*60}\n")
    print("Server is running. Access dashboard at http://localhost:8001")
    return tunnel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle"""
    if not CLIENT_MODE:
        init_server()
        tunnel = setup_tunnel(TOKEN)
        yield
        tunnel.stop()
    else:
        yield


app = FastAPI(lifespan=lifespan)


async def generate_plot(student_id: str = None, cumulative: bool = False):
    """Generate stacked area chart showing emotion distribution"""
    async with predictions_lock:
        if not predictions_log:
            return None
        
        if student_id:
            filtered_data = [(sid, ts, preds) for sid, ts, preds in predictions_log if sid == student_id]
        else:
            filtered_data = predictions_log
        
        if not filtered_data:
            return None
        
        filtered_data.sort(key=lambda x: x[1])
        timestamps = [entry[1] for entry in filtered_data]
        emotion_keys = list(filtered_data[0][2].keys())
        
        emotion_data = {key: [] for key in emotion_keys}
        for _, _, predictions in filtered_data:
            for key in emotion_keys:
                emotion_data[key].append(predictions.get(key, 0.0))
        
        emotion_arrays = [np.array(emotion_data[key]) for key in emotion_keys]
        if cumulative:
            emotion_arrays = [np.cumsum(arr) for arr in emotion_arrays]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        time_indices = list(range(1, len(timestamps) + 1))
        
        color_list = [colors.get(key.lower(), '#CCCCCC') for key in emotion_keys]
        
        ax.stackplot(time_indices, *emotion_arrays, labels=emotion_keys, colors=color_list, alpha=0.8)
        ax.set_xlabel('Reading index', fontsize=12)
        ylabel = 'Cumulative Emotion Count' if cumulative else 'Emotion Probability'
        ax.set_ylabel(ylabel, fontsize=12)
        
        title_suffix = f" - {student_id}" if student_id else " - All Students"
        title_prefix = "Cumulative " if cumulative else ""
        ax.set_title(f'{title_prefix}Emotion Distribution Over Readings{title_suffix}', fontsize=14, pad=20)
        
        if not cumulative:
            ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9, fontsize=10)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf


@app.post("/api/emotions")
async def receive_emotions(request: Request):
    """Receive emotion predictions from clients"""
    try:
        data = await request.json()
        timestamp = int(time.time())
        async with predictions_lock:
            predictions_log.append((data['id'], timestamp, data['predictions']))
        print(f"Received data from {data['id']} at {timestamp}")
        return JSONResponse({"status": "ok"}, status_code=200)
    except Exception as e:
        print("Error:", e)
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/statistics")
async def get_statistics():
    """Get overall statistics for dashboard"""
    async with predictions_lock:
        student_counts = defaultdict(int)
        for student_id, _, _ in predictions_log:
            student_counts[student_id] += 1
        return {
            "total_students": len(student_counts),
            "total_predictions": len(predictions_log),
            "active_sessions": len(student_counts),
            "students": dict(student_counts)
        }


@app.get("/api/plot/{student_id}")
async def get_student_plot(student_id: str, type: str = "regular"):
    """Get emotion plot for specific student"""
    plot_buffer = await generate_plot(student_id, cumulative=(type == "cumulative"))
    if plot_buffer is None:
        return JSONResponse({"error": "No data available"}, status_code=404)
    return StreamingResponse(plot_buffer, media_type="image/png")


@app.get("/api/plot")
async def get_all_students_plot(type: str = "regular"):
    """Get emotion plot for all students"""
    plot_buffer = await generate_plot(cumulative=(type == "cumulative"))
    if plot_buffer is None:
        return JSONResponse({"error": "No data available"}, status_code=404)
    return StreamingResponse(plot_buffer, media_type="image/png")


@app.get("/api/tunnel-url")
async def get_tunnel_url():
    """Get the tunnel URL"""
    return {"url": TUNNEL_URL if TUNNEL_URL else ""}


@app.post("/api/start-server")
async def start_server_mode():
    """Start server mode"""
    return {"status": "Server is already running"}


@app.post("/api/start-client")
async def start_client_mode(request: Request):
    """Start client mode"""
    data = await request.json()
    endpoint_url = data.get('url', '')
    show_camera = data.get('show_camera', False)
    
    if not endpoint_url:
        return JSONResponse({"error": "URL is required"}, status_code=400)
    
    # Start client in a separate thread
    def run_client_thread():
        run_client(endpoint_url, show_camera)
    
    thread = threading.Thread(target=run_client_thread, daemon=True)
    thread.start()
    
    return {"status": "Client started", "url": endpoint_url}


@app.get("/health")
async def health_check():
    return {"message": "ok"}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/dashboard")
async def dashboard():
    return FileResponse("static/index.html")


@app.get("/")
async def read_root():
    return FileResponse("static/menu.html")


# ============================================================================
# CLIENT CODE
# ============================================================================
def list_available_cameras(max_cameras=10):
    """Find available cameras"""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def select_camera():
    """Let user select camera"""
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


def init_client(endpoint_url):
    """Initialize client with model"""
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


def preprocess(frame, device='cpu'):
    """Preprocess frame for model"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def predict(frame, model, device='cpu'):
    """Predict emotions from frame"""
    model.eval()
    with torch.no_grad():
        inputs = preprocess(frame, device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


def send_to_server(data, endpoint_url):
    """Send predictions to server"""
    try:
        r = requests.post(urljoin(endpoint_url, "api/emotions"), json=data, timeout=5)
        print(f"Sent: {r.status_code}")
    except Exception as e:
        print(f"Error sending: {e}")


def run_client(endpoint_url, show_camera=False):
    """Run emotion detection client"""
    print("\n" + "="*60)
    print("  STARTING CLIENT MODE")
    print("="*60)
    print(f"\nConnecting to: {endpoint_url}")
    print("Initializing model...")
    
    model, device = init_client(endpoint_url)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    camera_id = select_camera()
    if camera_id is None:
        return
    
    print(f"\nOpening camera {camera_id}...")
    
    max_retries = 3
    cap = None
    
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}...")
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
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
    
    if show_camera:
        print("Camera view enabled. Press 'q' in the camera window to close it.")
    
    print(f"\nCamera running. Sending predictions every {CAPTURE_INTERVAL}s to {endpoint_url}")
    print("Press Ctrl+C to stop\n")
    
    last_capture_time = time.time()
    GUID = uuid.uuid4()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Could not read frame")
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            current_time = time.time()
            
            if current_time - last_capture_time >= CAPTURE_INTERVAL:
                preds = predict(frame, model, device)
                print(f"Predictions: {preds}")
                data = {"id": str(GUID), "predictions": preds}
                send_to_server(data, endpoint_url)
                last_capture_time = current_time
            
            if show_camera:
                for (x, y, w, h) in face:
                    cv2.rectangle(frame, (x - FACE_EXTENSION_X, y - FACE_EXTENSION_Y), (x + w + FACE_EXTENSION_X, y + h + FACE_EXTENSION_Y), (0, 255, 0), 2)
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
        if show_camera:
            cv2.destroyAllWindows()
        print("Camera released. Goodbye!")


# ============================================================================
# MAIN
# ============================================================================
def open_browser():
    """Open browser after short delay"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:8001', autoraise=True)


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("  EMOTION MONITORING SYSTEM")
    print("="*60)
    print("\nStarting web interface...")
    print("Opening browser at http://localhost:8001")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run("__main__:app", host="0.0.0.0", port=8001, reload=False)


if __name__ == "__main__":
    main()
