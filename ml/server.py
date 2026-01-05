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
from sklearn.metrics import precision_score, f1_score, roc_auc_score
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
        self.plot_path = "static/training_plot.png"
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
                "plot_url": f"/static/training_plot.png" if os.path.exists(self.plot_path) else None
            }

# Global training manager
training_manager = TrainingManager()

app = FastAPI()

def init():
    # Create plots directory if it doesn't exist
    os.makedirs("static", exist_ok=True)


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
    
    if "accuracy" in metric_names:
        metrics["accuracy"] = (y_true == y_pred).sum() / len(y_true)
    
    if "precision" in metric_names:
        metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if "f1" in metric_names:
        metrics["f1"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if "roc_auc" in metric_names:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except:
            metrics["roc_auc"] = 0.0
    
    return metrics


def generate_plot():
    """Generate training progress plot"""
    with training_manager.lock:
        if not training_manager.train_losses:
            return
        
        # Determine number of subplots needed
        num_metrics = len(training_manager.train_metrics)
        num_plots = 1 + num_metrics  # Loss + each metric
        
        fig, axes = plt.subplots(1, min(num_plots, 3), figsize=(15, 5))
        if num_plots == 1:
            axes = [axes]
        elif num_plots == 2:
            axes = [axes]
        
        # Plot loss
        ax_loss = axes[0] if num_plots > 1 else axes
        epochs = range(1, len(training_manager.train_losses) + 1)
        ax_loss.plot(epochs, training_manager.train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
        if training_manager.val_losses:
            ax_loss.plot(epochs, training_manager.val_losses, 'r-o', label='Val Loss', linewidth=2, markersize=6)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Training and Validation Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        
        # Plot first metric (usually accuracy)
        if num_plots > 1 and training_manager.train_metrics:
            metric_name = list(training_manager.train_metrics.keys())[0]
            ax_metric = axes[1]
            ax_metric.plot(epochs, training_manager.train_metrics[metric_name], 'b-o', 
                          label=f'Train {metric_name.title()}', linewidth=2, markersize=6)
            if metric_name in training_manager.val_metrics:
                ax_metric.plot(epochs, training_manager.val_metrics[metric_name], 'r-o',
                             label=f'Val {metric_name.title()}', linewidth=2, markersize=6)
            ax_metric.set_xlabel('Epoch')
            ax_metric.set_ylabel(metric_name.title())
            ax_metric.set_title(f'{metric_name.title()} Progress')
            ax_metric.legend()
            ax_metric.grid(True, alpha=0.3)
        
        # Plot second metric if available
        if num_plots > 2 and len(training_manager.train_metrics) > 1:
            metric_name = list(training_manager.train_metrics.keys())[1]
            ax_metric2 = axes[2]
            ax_metric2.plot(epochs, training_manager.train_metrics[metric_name], 'b-o',
                           label=f'Train {metric_name.title()}', linewidth=2, markersize=6)
            if metric_name in training_manager.val_metrics:
                ax_metric2.plot(epochs, training_manager.val_metrics[metric_name], 'r-o',
                              label=f'Val {metric_name.title()}', linewidth=2, markersize=6)
            ax_metric2.set_xlabel('Epoch')
            ax_metric2.set_ylabel(metric_name.title())
            ax_metric2.set_title(f'{metric_name.title()} Progress')
            ax_metric2.legend()
            ax_metric2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(training_manager.plot_path, dpi=100, bbox_inches='tight')
        plt.close()


def train_model(config):
    """Main training function"""
    os.remove(training_manager.plot_path) if os.path.exists(training_manager.plot_path) else None
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
            train_metrics = calculate_metrics(
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
            val_metrics = calculate_metrics(
                np.array(val_labels),
                np.array(val_preds),
                np.array(val_probs),
                config["metrics"]
            )
            
            for metric_name, value in val_metrics.items():
                training_manager.val_metrics[metric_name].append(value)
            
            # Update best accuracy
            if "accuracy" in val_metrics:
                if val_metrics["accuracy"] > training_manager.best_accuracy:
                    training_manager.best_accuracy = val_metrics["accuracy"]
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
            generate_plot()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if "accuracy" in val_metrics:
                print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        
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


@app.get("/health")
async def health_check():
    return {"message": "ok"}


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/train.html")


if __name__ == "__main__":
    init()
    uvicorn.run("server:app", host="0.0.0.0", port=9001, reload=True)
