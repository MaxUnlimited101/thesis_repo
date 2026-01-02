import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize 
import tqdm
from torch.utils.data import Subset, Dataset
from sklearn.model_selection import train_test_split

import datasets
import models

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

def get_data_loaders(batch_size, dataset: datasets.BaseEmotionDataset, val_split=0.2, num_workers=0):
    """Create data loaders for training and validation datasets."""
    # Define transformations
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset.transform = transform_train
    # Split dataset into train and validation sets
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, random_state=42, 
        stratify=[dataset[i][1] for i in indices]
    )

    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(
        np.array(train_indices).reshape(-1, 1), 
        [dataset[i][1] for i in train_indices]
    )

    # Create subset datasets
    train_dataset = Subset(dataset, X_resampled.flatten())
    val_dataset = Subset(dataset, val_indices)

    # Update val_dataset to use validation transforms
    val_dataset.dataset.transform = transform_val

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    return train_loader, val_loader
    
    
# Create wrapper for transforms
class TransformWrapper:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
        
    def __len__(self):
        return len(self.dataset)


def get_train_validation_test_split_dataloaders(dataset: Dataset, val_split=0.1, test_split=0.1, batch_size=64, num_workers=0):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset.transform = transform_train
    # Split dataset into train and validation and test sets
    print(f"Total dataset size: {len(dataset)} samples (before oversampling)")
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(
        indices, test_size=(val_split + test_split), random_state=42, 
        stratify=[dataset[i][1] for i in indices]
    )
    val_size = int(len(dataset) * val_split / (val_split + test_split))
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42, # 50-50 split of the temp set
        stratify=[dataset[i][1] for i in temp_indices]
    )
    oversampler = RandomOverSampler(random_state=42)
    train_resampled, _ = oversampler.fit_resample(
        np.array(train_indices).reshape(-1, 1),
        [dataset[i][1] for i in train_indices]
    )
    
    train_subset = Subset(dataset, train_resampled.flatten())
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    train_dataset = TransformWrapper(train_subset, transform_train)
    val_dataset = TransformWrapper(val_subset, transform_val)
    test_dataset = TransformWrapper(test_subset, transform_val)
    
    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val
    test_dataset.dataset.transform = transform_val
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


def train_model_with_test(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs=10, device='cpu'):
    model = model.to(device)
    
    print("Starting training...")
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    train_precisions = []
    val_precisions = []
    train_roc_aucs = []
    val_roc_aucs = []
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        train_predictions = []
        train_labels = []
        train_probabilities = []

        for batch_idx, (inputs, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for metrics
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_probabilities.extend(probabilities.detach().cpu().numpy())

        # Update scheduler after epoch
        scheduler.step()
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Convert to numpy arrays
        train_predictions = np.array(train_predictions)
        train_labels = np.array(train_labels)
        train_probabilities = np.array(train_probabilities)
        
        # Calculate F1, Precision, and ROC AUC for training
        train_f1 = f1_score(train_labels, train_predictions, average='weighted', zero_division=0)
        train_precision = precision_score(train_labels, train_predictions, average='weighted', zero_division=0)
        train_f1_scores.append(train_f1)
        train_precisions.append(train_precision)
        
        # Calculate ROC AUC for training
        try:
            unique_labels = sorted(np.unique(train_labels))
            if len(unique_labels) > 1:
                train_labels_bin = label_binarize(train_labels, classes=unique_labels)
                train_proba_filtered = train_probabilities[:, unique_labels]
                
                if len(unique_labels) == 2:
                    train_roc_auc = roc_auc_score(train_labels_bin, train_proba_filtered[:, 1])
                else:
                    train_roc_auc = roc_auc_score(train_labels_bin, train_proba_filtered, 
                                                 multi_class='ovr', average='weighted')
            else:
                train_roc_auc = 0.0
        except Exception as e:
            print(f"Warning: Could not calculate training ROC AUC: {e}")
            train_roc_auc = 0.0
        
        train_roc_aucs.append(train_roc_auc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_labels = []
        val_probabilities = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for metrics
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probabilities.extend(probabilities.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Convert to numpy arrays
        val_predictions = np.array(val_predictions)
        val_labels = np.array(val_labels)
        val_probabilities = np.array(val_probabilities)
        
        # Calculate F1, Precision, and ROC AUC for validation
        val_f1 = f1_score(val_labels, val_predictions, average='weighted', zero_division=0)
        val_precision = precision_score(val_labels, val_predictions, average='weighted', zero_division=0)
        val_f1_scores.append(val_f1)
        val_precisions.append(val_precision)
        
        # Calculate ROC AUC for validation
        try:
            unique_labels = sorted(np.unique(val_labels))
            if len(unique_labels) > 1:
                val_labels_bin = label_binarize(val_labels, classes=unique_labels)
                val_proba_filtered = val_probabilities[:, unique_labels]
                
                if len(unique_labels) == 2:
                    val_roc_auc = roc_auc_score(val_labels_bin, val_proba_filtered[:, 1])
                else:
                    val_roc_auc = roc_auc_score(val_labels_bin, val_proba_filtered, 
                                               multi_class='ovr', average='weighted')
            else:
                val_roc_auc = 0.0
        except Exception as e:
            print(f"Warning: Could not calculate validation ROC AUC: {e}")
            val_roc_auc = 0.0
        
        val_roc_aucs.append(val_roc_auc)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Prec: {train_precision:.4f}, ROC-AUC: {train_roc_auc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Prec: {val_precision:.4f}, ROC-AUC: {val_roc_auc:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, "best_model_all.pth")
            patience_counter = 0
            print(f"  ✓ New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        print("-" * 50)
    
    # Final evaluation on test set
    model = torch.load("best_model_all.pth")
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Store for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Calculate detailed test metrics
    print("\nDetailed Test Results:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=datasets.emotions))
    
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_f1_scores': train_f1_scores,
        'val_f1_scores': val_f1_scores,
        'train_precisions': train_precisions,
        'val_precisions': val_precisions,
        'train_roc_aucs': train_roc_aucs,
        'val_roc_aucs': val_roc_aucs,
        'test_accuracy': test_acc,
        'best_val_acc': best_val_acc,
        'num_epochs_trained': len(train_losses)
    }
    
    # Save metrics to JSON with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_filename = f"training_metrics_{timestamp}.json"
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTraining metrics saved to: {metrics_filename}")
    
    torch.save(model, "final_model_all.pth")
    
    return model, metrics



def main():
    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    NUM_WORKERS = 0
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    
    print(f"Using device: {DEVICE}")

    all_datasets = [datasets.AffectNetDataset(root_dir="AffectNet"), 
                    datasets.NHFIERDataset(root_dir="NHFIER"),
                    datasets.FER2013Dataset(root_dir="FER-2013")]
    dataset = datasets.combine_datasets(all_datasets)

    # Get data loaders
    # This takes ~5 minutes
    train_loader, val_loader, test_loader = get_train_validation_test_split_dataloaders(dataset, VAL_SPLIT, TEST_SPLIT, BATCH_SIZE, NUM_WORKERS)

    # Initialize model, loss function, and optimizer
    model = models.EfficientNetB0(pretrained=True)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Train the model
    trained_model = train_model_with_test(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, NUM_EPOCHS, DEVICE)
    print("Training complete. Model saved.")
    
if __name__ == "__main__":
    main()
