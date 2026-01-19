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
from matplotlib.ticker import MaxNLocator
import numpy as np
import io
from fastapi.staticfiles import StaticFiles
import webbrowser
import threading
import json

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
CLIENT_THREAD = None
CLIENT_THREAD_RUNNING = False


FACE_EXTENSION_Y=50
FACE_EXTENSION_X=15


# ============================================================================
# SERVER (EDUCATOR) CODE
# ============================================================================
TOKEN = ""
USER_TIMEOUT = 10
predictions_log = []
active_students = dict()
predictions_lock = Lock()

colors = {
    'neutral': "#808991", 'happy': "#FFF700", 'sad': "#0B4CB3",
    'surprise': "#51B3F0", 'fear': "#1F6D2E", 'disgust': "#79299C",
    'angry': "#CF0E0E", 'contempt': "#C3A02F"
}

colors_rgba = {
    'neutral': 'rgba(128, 137, 145, 1)',
    'happy': 'rgba(255, 247, 0, 1)',
    'sad': 'rgba(11, 76, 179, 1)',
    'surprise': 'rgba(81, 179, 240, 1)',
    'fear': 'rgba(31, 109, 46, 1)',
    'disgust': 'rgba(121, 41, 156, 1)',
    'angry': 'rgba(207, 14, 14, 1)',
    'contempt': 'rgba(195, 160, 47, 1)'
}

border_colors_rgba = {
    'neutral': 'rgba(90, 97, 105, 1)',
    'happy': 'rgba(230, 180, 0, 1)',
    'sad': 'rgba(8, 50, 140, 1)',
    'surprise': 'rgba(45, 140, 200, 1)',
    'fear': 'rgba(20, 80, 30, 1)',
    'disgust': 'rgba(90, 25, 120, 1)',
    'angry': 'rgba(160, 10, 10, 1)',
    'contempt': 'rgba(150, 120, 30, 1)'
}

def init_server():
    """Initialize server by reading token"""
    global TOKEN
    try:
        with open("token.txt", "r") as f:
            TOKEN = f.read().strip()
            print(f"Token loaded: {TOKEN}")
    except FileNotFoundError:
        print("WARN: token.txt not found. Continuing without token...")
        TOKEN = ""


def setup_tunnel(token):
    """Start HTTP tunnel forwarding to localhost:8001"""
    if not token:
        print("Skipping tunnel setup due to missing token...")
        return None
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
        if tunnel:
            tunnel.stop()
    else:
        yield


app = FastAPI(lifespan=lifespan)

def get_summarized_emotions(emotion_arrays):
    summed_per_emotion = [np.sum(arr) for arr in emotion_arrays]
    
    grand_total = sum(summed_per_emotion)
    
    if grand_total == 0:
        normalized_values = [0.0] * len(summed_per_emotion)
    else:
        normalized_values = [val / grand_total for val in summed_per_emotion]
    
    return normalized_values

async def generate_plot(student_id: str = None, is_cumulative: bool = False, is_summary: bool = False, fill: bool = True):
    """Generate chart showing emotion distribution"""
    async with predictions_lock:
        if not predictions_log:
            return json.dumps({"labels": [], "datasets": []})
        
        if student_id:
            filtered_data = [(sid, ts, preds) for sid, ts, preds in predictions_log if sid == student_id]
        else:
            filtered_data = predictions_log
        
        if not filtered_data:
            return json.dumps({"labels": [], "datasets": []})
        
        filtered_data.sort(key=lambda x: x[1])
        timestamps = [entry[1] for entry in filtered_data]
        emotion_keys = list(filtered_data[0][2].keys())
        
        emotion_data = {key: [] for key in emotion_keys}
        for _, _, predictions in filtered_data:
            for key in emotion_keys:
                emotion_data[key].append(predictions.get(key, 0.0))

        emotion_arrays = [np.array(emotion_data[key]) for key in emotion_keys]

        datasets = []
        if is_summary:
            summarized_values = get_summarized_emotions(emotion_arrays)
            
            labels = emotion_keys

            datasets = [{
                "label": emotion_keys,
                "data": summarized_values,
                "backgroundColor": [colors_rgba.get(k, "#808080") for k in emotion_keys],
                "borderWidth": 2,
            }]
            

        else:
            labels = [f"{i}" for i in range(len(emotion_arrays[0]))]        

            if is_cumulative:
                emotion_arrays = [np.cumsum(arr) for arr in emotion_arrays]
                total_final = sum(arr[-1] for arr in emotion_arrays)
                emotion_arrays = [arr / total_final for arr in emotion_arrays]
                
            for key, data_array in zip(emotion_keys, emotion_arrays):
                datasets.append({
                    "label": key,
                    "data": data_array.tolist(),
                    "backgroundColor": colors_rgba.get(key, "#808080"),
                    "fill": fill,
                    "borderWidth": 3,
                    "borderColor": border_colors_rgba.get(key, "#808080")
                })

        chart_data = {
            "labels": labels,
            "datasets": datasets
        }

        return json.dumps(chart_data)


@app.post("/api/emotions")
async def receive_emotions(request: Request):
    """Receive emotion predictions from clients"""
    try:
        data = await request.json()
        student_id = data['id']
        timestamp = int(time.time())
        async with predictions_lock:
            predictions_log.append((student_id, timestamp, data['predictions']))
            active_students[student_id] = timestamp
        print(f"Received data from {student_id} at {timestamp}")
        return JSONResponse({"status": "ok"}, status_code=200)
    except Exception as e:
        print("Error:", e)
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/statistics")
async def get_statistics():
    """Get overall statistics for dashboard"""
    async with predictions_lock:
        now = time.time()

        active_ids = [
            sid for sid, last_seen in active_students.items()
            if (now - last_seen) < USER_TIMEOUT
        ]

        student_counts = defaultdict(int)
        for student_id, _, _ in predictions_log:
            student_counts[student_id] += 1
        return {
            "total_students": len(student_counts),
            "total_predictions": len(predictions_log),
            "active_sessions": len(active_ids),
            "students": dict(student_counts)
        }


@app.get("/api/plot/{student_id}")
async def get_student_plot(student_id: str, is_cumulative: bool = False, is_summary: bool = False, fill : bool = True):
    """Get emotion plot for specific student"""
    plot_buffer = await generate_plot(student_id, is_cumulative, is_summary, fill)
    if plot_buffer is None:
        return JSONResponse({"error": "No data available"}, status_code=404)
    return StreamingResponse(plot_buffer, media_type="image/png")


@app.get("/api/plot")
async def get_all_students_plot(is_cumulative: bool = False, is_summary: bool = False, fill : bool = True):
    """Get emotion plot for all students"""
    plot_buffer = await generate_plot(is_cumulative, is_summary, fill)
    if plot_buffer is None:
        return JSONResponse({"error": "No data available"}, status_code=404)
    return StreamingResponse(plot_buffer, media_type="image/png")


@app.get("/api/tunnel-url")
async def get_tunnel_url():
    """Get the tunnel URL"""
    return {"url": TUNNEL_URL if TUNNEL_URL else "Error: No tunnel URL available, check whether token.txt exists."}


@app.post("/api/start-server")
async def start_server_mode():
    """Start server mode"""
    return {"status": "Server is already running"}


@app.post("/api/start-client")
async def start_client_mode(request: Request):
    """Start client mode"""
    global CLIENT_THREAD, CLIENT_THREAD_RUNNING
    
    data = await request.json()
    endpoint_url = data.get('url', '')
    show_camera = data.get('show_camera', False)
    
    if not endpoint_url:
        return JSONResponse({"error": "URL is required"}, status_code=400)
    
    if CLIENT_THREAD and CLIENT_THREAD.is_alive():
        return JSONResponse({"error": "Client already running"}, status_code=400)
    
    # Start client in a separate thread
    def run_client_thread():
        global CLIENT_THREAD_RUNNING
        CLIENT_THREAD_RUNNING = True
        run_client(endpoint_url, show_camera)
    
    CLIENT_THREAD = threading.Thread(target=run_client_thread, daemon=True)
    CLIENT_THREAD.start()
    
    return {"status": "Client started", "url": endpoint_url}


@app.post("/api/stop-client")
async def stop_client_mode():
    """Stop client mode"""
    global CLIENT_THREAD_RUNNING
    
    if not CLIENT_THREAD or not CLIENT_THREAD.is_alive():
        return JSONResponse({"error": "Client not running"}, status_code=400)
    
    CLIENT_THREAD_RUNNING = False
    return {"status": "Client stopped"}


@app.get("/api/client-status")
async def get_client_status():
    """Get client running status"""
    return {"running": CLIENT_THREAD and CLIENT_THREAD.is_alive()}


@app.get("/health")
async def health_check():
    return {"message": "ok"}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/dashboard")
async def dashboard():
    return FileResponse("static/index.html")


@app.get("/train")
async def train_page():
    return FileResponse("static/train.html")


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
    print("Click 'Stop Client' to stop\n")
    
    last_capture_time = time.time()
    GUID = uuid.uuid4()
    
    try:
        while CLIENT_THREAD_RUNNING:
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
# TRAIN CODE
# ============================================================================

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from contextlib import asynccontextmanager
from asyncio import Lock
import time
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import io
from datetime import datetime
from fastapi.staticfiles import StaticFiles
import datasets 
import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import precision_score, f1_score, roc_auc_score, accuracy_score
import threading
from pathlib import Path

# Training state manager
class TrainingManager:
    def __init__(self):
        self.is_training = False
        self.should_stop = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.config = {}
        self.plot_paths = []  # List of plot file paths
        self.model_path = "static/trained_model.pth"
        self.status_message = "Idle"
        self.lock = threading.Lock()
        
    def reset(self):
        with self.lock:
            self.is_training = False
            self.should_stop = False
            self.current_epoch = 0
            self.total_epochs = 0
            self.best_accuracy = 0.0
            self.train_losses = []
            self.val_losses = []
            self.train_metrics = defaultdict(list)
            self.val_metrics = defaultdict(list)
            self.config = {}
            self.status_message = "Idle"
    
    def get_status(self):
        with self.lock:
            return {
                "status": self.status_message,
                "is_training": self.is_training,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "best_accuracy": self.best_accuracy,
                "complete": not self.is_training and self.current_epoch > 0,
                "plot_urls": [f"/static/{Path(p).name}" for p in self.plot_paths if os.path.exists(p)]
            }

# Global training manager
training_manager = TrainingManager()

def get_dataset_class(dataset_name):
    """Get dataset class from datasets module"""
    dataset_mapping = {
        "AffectNet": "AffectNetDataset",
        "FER-2013": "FER2013Dataset",
        "NHFIER": "NHFIERDataset"
    }
    
    class_name = dataset_mapping.get(dataset_name)
    if class_name and hasattr(datasets, class_name):
        return getattr(datasets, class_name)
    return None


def get_model_class(model_name):
    """Get model class from models module"""
    if hasattr(models, model_name):
        return getattr(models, model_name)
    return None


def get_optimizer(optimizer_name, model_params, lr=0.001):
    """Get optimizer instance"""
    optimizer_map = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "AdamW": optim.AdamW,
        "RMSprop": optim.RMSprop,
        "Adagrad": optim.Adagrad
    }
    
    optimizer_cls = optimizer_map.get(optimizer_name, optim.Adam)
    if optimizer_name == "SGD":
        return optimizer_cls(model_params, lr=lr, momentum=0.9)
    return optimizer_cls(model_params, lr=lr)


def get_scheduler(scheduler_name, optimizer, num_epochs=10, lr=0.001):
    """Get learning rate scheduler"""
    scheduler_map = {
        "StepLR": lambda: optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1),
        "ReduceLROnPlateau": lambda: optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3),
        "CosineAnnealingLR": lambda: optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs),
        "ExponentialLR": lambda: optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
        "CyclicLR": lambda: optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/2, max_lr=lr, step_size_up=5),
        "OneCycleLR": lambda: optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=100)
    }
    
    scheduler_fn = scheduler_map.get(scheduler_name)
    return scheduler_fn() if scheduler_fn else None


def get_criterion(criterion_name):
    """Get loss function"""
    criterion_map = {
        "CrossEntropyLoss": nn.CrossEntropyLoss(),
        "NLLLoss": nn.NLLLoss(),
        "LabelSmoothingCrossEntropy": nn.CrossEntropyLoss(label_smoothing=0.1)
    }
    
    return criterion_map.get(criterion_name, nn.CrossEntropyLoss())


def calculate_metrics(y_true, y_pred, y_prob, metric_names):
    """Calculate specified metrics"""
    metrics = {}
    
    # Always calculate accuracy for internal tracking of best model
    accuracy = accuracy_score(y_true, y_pred)
    
    # Only add accuracy to metrics dict (for plotting) if user selected it
    if "accuracy" in metric_names:
        metrics["accuracy"] = accuracy
    
    if "precision" in metric_names:
        metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if "f1" in metric_names:
        metrics["f1"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if "roc_auc" in metric_names:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except:
            metrics["roc_auc"] = 0.0
    
    # Return both metrics dict and accuracy separately
    return metrics, accuracy


def generate_plot_for_train():
    """Generate training progress plots - creates multiple images if needed"""
    with training_manager.lock:
        if not training_manager.train_losses:
            return
        
        training_manager.plot_paths = []  # Reset plot paths
        epochs = range(1, len(training_manager.train_losses) + 1)
        
        # Collect all plots to generate: loss + all metrics
        plots_to_generate = [("loss", None)]  # (plot_type, metric_name)
        for metric_name in training_manager.train_metrics.keys():
            plots_to_generate.append(("metric", metric_name))
        
        # Generate plots in groups of 2 per image
        plots_per_image = 2
        num_images = (len(plots_to_generate) + plots_per_image - 1) // plots_per_image
        
        for img_idx in range(num_images):
            start_idx = img_idx * plots_per_image
            end_idx = min(start_idx + plots_per_image, len(plots_to_generate))
            plots_in_this_image = plots_to_generate[start_idx:end_idx]
            
            # Create figure with appropriate number of subplots
            fig, axes = plt.subplots(1, len(plots_in_this_image), figsize=(10 * len(plots_in_this_image), 5))
            if len(plots_in_this_image) == 1:
                axes = [axes]
            
            # Generate each subplot
            for ax_idx, (plot_type, metric_name) in enumerate(plots_in_this_image):
                ax = axes[ax_idx]
                
                if plot_type == "loss":
                    # Plot loss
                    ax.plot(epochs, training_manager.train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
                    if training_manager.val_losses:
                        ax.plot(epochs, training_manager.val_losses, 'r-o', label='Val Loss', linewidth=2, markersize=6)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title('Training and Validation Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    # Plot metric
                    ax.plot(epochs, training_manager.train_metrics[metric_name], 'b-o',
                           label=f'Train {metric_name.title()}', linewidth=2, markersize=6)
                    if metric_name in training_manager.val_metrics:
                        ax.plot(epochs, training_manager.val_metrics[metric_name], 'r-o',
                               label=f'Val {metric_name.title()}', linewidth=2, markersize=6)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(metric_name.title())
                    ax.set_title(f'{metric_name.title()} Progress')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = f"static/training_plot_{img_idx + 1}.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            training_manager.plot_paths.append(plot_path)


def train_model(config):
    """Main training function"""
    # Remove old plot files
    for old_plot in Path("static").glob("training_plot_*.png"):
        try:
            old_plot.unlink()
        except:
            pass
    try:
        training_manager.status_message = "Initializing..."
        
        # Set device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        
        # Data transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        training_manager.status_message = "Loading datasets..."
        train_datasets = []
        val_datasets = []
        test_datasets = []
        
        # Get split ratios from config
        train_split = config.get("train_split", 0.8)
        val_split = config.get("val_split", 0.1)
        test_split = config.get("test_split", 0.1)
        
        for dataset_name in config["datasets"]:
            dataset_cls = get_dataset_class(dataset_name)
            if dataset_cls:
                try:
                    # Check if dataset supports split parameter
                    if dataset_name == "FER-2013":
                        train_ds = dataset_cls(root_dir=dataset_name, split='train', transform=transform)
                        # For FER-2013, split test set into val and test
                        test_full = dataset_cls(root_dir=dataset_name, split='test', transform=transform)
                        val_size = int(len(test_full) * (val_split / (val_split + test_split)))
                        test_size = len(test_full) - val_size
                        val_ds, test_ds = torch.utils.data.random_split(
                            test_full,
                            [val_size, test_size],
                            generator=torch.Generator().manual_seed(42)
                        )
                    else:
                        # For datasets without splits, use full dataset and split manually
                        full_ds = dataset_cls(root_dir=dataset_name, transform=transform)
                        
                        # Split according to config ratios
                        train_size = int(train_split * len(full_ds))
                        val_size = int(val_split * len(full_ds))
                        test_size = len(full_ds) - train_size - val_size
                        
                        train_ds, val_ds, test_ds = torch.utils.data.random_split(
                            full_ds, 
                            [train_size, val_size, test_size],
                            generator=torch.Generator().manual_seed(42)
                        )
                    
                    train_datasets.append(train_ds)
                    val_datasets.append(val_ds)
                    test_datasets.append(test_ds)
                    print(f"Loaded {dataset_name}: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
                    
                except Exception as e:
                    print(f"Error loading {dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        if not train_datasets:
            training_manager.status_message = "Error: No datasets loaded"
            training_manager.is_training = False
            return
        
        # Combine datasets
        train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        test_dataset = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]
        
        # Get batch size from config
        batch_size = config.get("batch_size", 32)
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Initialize model
        training_manager.status_message = "Initializing model..."
        model_cls = get_model_class(config["model"])
        if not model_cls:
            training_manager.status_message = f"Error: Model {config['model']} not found"
            training_manager.is_training = False
            return
        
        model = model_cls().to(device)
        
        # Get learning rate and epochs from config
        learning_rate = config.get("learning_rate", 0.001)
        num_epochs = config.get("epochs", 10)
        
        # Optimizer, scheduler, criterion
        optimizer = get_optimizer(config["optimizer"], model.parameters(), lr=learning_rate)
        scheduler = get_scheduler(config["scheduler"], optimizer, num_epochs=num_epochs, lr=learning_rate)
        criterion = get_criterion(config["criterion"])
        
        # Training loop
        training_manager.total_epochs = num_epochs
        
        for epoch in range(num_epochs):
            if training_manager.should_stop:
                training_manager.status_message = "Training stopped by user"
                break
            
            training_manager.current_epoch = epoch + 1
            training_manager.status_message = f"Training epoch {epoch + 1}/{num_epochs}"
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            train_probs = []
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if training_manager.should_stop:
                    training_manager.status_message = "Training stopped by user"
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Collect predictions
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                train_probs.extend(probs.detach().cpu().numpy())
            
            train_loss /= len(train_loader)
            training_manager.train_losses.append(train_loss)
            
            # Calculate training metrics
            train_metrics, train_accuracy = calculate_metrics(
                np.array(train_labels),
                np.array(train_preds),
                np.array(train_probs),
                config["metrics"]
            )
            
            for metric_name, value in train_metrics.items():
                training_manager.train_metrics[metric_name].append(value)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            val_probs = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
            
            val_loss /= len(val_loader)
            training_manager.val_losses.append(val_loss)
            
            # Calculate validation metrics
            val_metrics, val_accuracy = calculate_metrics(
                np.array(val_labels),
                np.array(val_preds),
                np.array(val_probs),
                config["metrics"]
            )
            
            for metric_name, value in val_metrics.items():
                training_manager.val_metrics[metric_name].append(value)
            
            # Always update best accuracy (even if not selected for plotting)
            if val_accuracy > training_manager.best_accuracy:
                training_manager.best_accuracy = val_accuracy
                # Save best model
                torch.save(model, training_manager.model_path)
                print(f"Saved best model with accuracy: {training_manager.best_accuracy:.4f}")
            
            # Update scheduler
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Generate plot
            generate_plot_for_train()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
        
        # Save final model if not already saved
        if not os.path.exists(training_manager.model_path):
            torch.save(model, training_manager.model_path)
            print(f"Saved final model")
        
        training_manager.status_message = "Training completed"
        training_manager.is_training = False
        
    except Exception as e:
        training_manager.status_message = f"Error: {str(e)}"
        training_manager.is_training = False
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()


@app.post("/train/start")
async def start_training(request: Request, background_tasks: BackgroundTasks):
    """Start training with given configuration"""
    if training_manager.is_training:
        return JSONResponse(
            status_code=400,
            content={"error": "Training already in progress"}
        )
    
    try:
        config = await request.json()
        
        # Validate config
        required_fields = ["datasets", "model", "optimizer", "scheduler", "criterion", "metrics"]
        for field in required_fields:
            if field not in config or not config[field]:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Missing required field: {field}"}
                )
        
        # Reset training manager
        training_manager.reset()
        training_manager.is_training = True
        training_manager.config = config
        
        # Start training in background
        background_tasks.add_task(train_model, config)
        
        return JSONResponse(content={"message": "Training started", "config": config})
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/train/status")
async def get_training_status():
    """Get current training status"""
    return JSONResponse(content=training_manager.get_status())


@app.post("/train/stop")
async def stop_training():
    """Stop ongoing training"""
    if not training_manager.is_training:
        return JSONResponse(
            status_code=400,
            content={"error": "No training in progress"}
        )
    
    training_manager.should_stop = True
    return JSONResponse(content={"message": "Stopping training..."})


@app.get("/train/download")
async def download_model():
    """Download the trained model"""
    if not os.path.exists(training_manager.model_path):
        return JSONResponse(
            status_code=404,
            content={"error": "Model file not found"}
        )
    
    return FileResponse(
        path=training_manager.model_path,
        filename="trained_model.pth",
        media_type="application/octet-stream"
    )


# ============================================================================
# MAIN
# ============================================================================
def open_browser():
    """Open browser after short delay"""
    time.sleep(2)
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
