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


import datasets
from datasets import NHFIERDataset
import models
from models import ResNet18
from torch.utils.data import Subset, Dataset
from sklearn.model_selection import train_test_split

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    
# Create wrapper for transforms
class TransformWrapper(Dataset):
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


def get_validation_test_split_dataloaders(dataset: Dataset, val_split=0.1, test_split=0.1, batch_size=64, num_workers=0):
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
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(
        indices, test_size=(val_split + test_split), random_state=42, 
        stratify=[dataset[i][1] for i in indices]
    )

    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42, # 50-50 split of the temp set
        stratify=[dataset[i][1] for i in temp_indices]
    )
    
    train_subset = Subset(dataset, train_indices)
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


def create_roc_curve_plot(y_true_bin, y_proba_filtered, unique_labels, roc_auc_weighted, roc_auc_macro, model_path):
    """Create ROC curve visualization for multi-class classification."""
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    from itertools import cycle
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for different classes
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 
                   'purple', 'brown', 'pink', 'gray', 'olive'])
    
    # Plot 1: Individual ROC curves for each class
    for i, (class_idx, color) in enumerate(zip(unique_labels, colors)):
        if len(unique_labels) > 2:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba_filtered[:, i])
            roc_auc_class = auc(fpr, tpr)
        else:
            # Binary classification case
            if i == 1:  # Positive class
                fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba_filtered[:, 1])
                roc_auc_class = auc(fpr, tpr)
            else:
                continue
        
        ax1.plot(fpr, tpr, color=color, lw=2,
                label=f'{datasets.emotions[class_idx]} (AUC = {roc_auc_class:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves for Each Class')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Summary metrics comparison
    metrics_names = ['Weighted AUC', 'Macro AUC']
    metrics_values = [roc_auc_weighted, roc_auc_macro]
    colors_bar = ['skyblue', 'lightcoral']
    
    bars = ax2.bar(metrics_names, metrics_values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax2.set_ylim([0, 1.0])
    ax2.set_ylabel('AUC Score')
    ax2.set_title('AUC-ROC Comparison: Weighted vs Macro')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add horizontal reference lines
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
    ax2.axhline(y=0.9, color='blue', linestyle='--', alpha=0.5, label='Excellent (0.9)')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    model_name = model_path.split('/')[-1].split('.')[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"roc_analysis_{model_name}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nROC analysis plot saved as: {plot_filename}")
    
    plt.show()


def create_auc_comparison_chart(model_results):
    """Create a comparison chart of AUC-ROC (macro) scores for different models."""
    model_names = list(model_results.keys())
    auc_macro_scores = [model_results[model]['auc_roc_macro'] for model in model_names]
    auc_weighted_scores = [model_results[model]['auc_roc_weighted'] for model in model_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, auc_weighted_scores, width, label='AUC-ROC (Weighted)', 
                   color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, auc_macro_scores, width, label='AUC-ROC (Macro)', 
                   color='lightcoral', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('AUC-ROC Score')
    ax.set_title('AUC-ROC Comparison: Weighted vs Macro Averaging')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add reference lines
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
    ax.axhline(y=0.9, color='blue', linestyle='--', alpha=0.5, label='Excellent (0.9)')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"auc_macro_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nAUC macro comparison chart saved as: {plot_filename}")
    
    plt.show()


def fin_test(test_loader, model_path, device='cpu'):
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []  # Add this to store probability scores
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get probability scores for AUC calculation
            probabilities = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Store for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())  # Store probabilities

    test_acc = test_correct / test_total
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Calculate detailed test metrics
    print("\nDetailed Test Results:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=datasets.emotions))
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate AUC-ROC properly using probability scores
    roc_auc = 0.0
    roc_auc_macro = 0.0
    try:
        # Get unique labels present in the test set
        unique_labels = sorted(np.unique(all_labels))
        print(f"Classes present in test set: {[datasets.emotions[i] for i in unique_labels]}")
        
        if len(unique_labels) > 1:
            # For multi-class AUC-ROC, we need to binarize the labels
            from sklearn.preprocessing import label_binarize
            
            # Binarize the labels for one-vs-rest AUC calculation
            y_true_bin = label_binarize(all_labels, classes=unique_labels)
            
            # Filter probabilities to only include classes present in test set
            y_proba_filtered = all_probabilities[:, unique_labels]
            
            if len(unique_labels) == 2:
                # Binary classification case
                roc_auc = roc_auc_score(y_true_bin, y_proba_filtered[:, 1])
                roc_auc_macro = roc_auc  # For binary, weighted and macro are the same
            else:
                # Multi-class classification case
                roc_auc = roc_auc_score(y_true_bin, y_proba_filtered, 
                                      multi_class='ovr', average='weighted')
                roc_auc_macro = roc_auc_score(y_true_bin, y_proba_filtered, 
                                            multi_class='ovr', average='macro')
            print(f"Final Test ROC AUC (macro): {roc_auc_macro:.4f}")
            
            # Per-class AUC-ROC
            print("\nPer-class AUC-ROC:")
            per_class_auc = []
            for i, class_idx in enumerate(unique_labels):
                if len(unique_labels) > 2:
                    class_auc = roc_auc_score(y_true_bin[:, i], y_proba_filtered[:, i])
                    print(f"  {datasets.emotions[class_idx]}: {class_auc:.4f}")
                    per_class_auc.append(class_auc)
                elif len(unique_labels) == 2:
                    if i == 1:  # Positive class in binary case
                        class_auc = roc_auc_score(y_true_bin, y_proba_filtered[:, 1])
                        print(f"  {datasets.emotions[class_idx]}: {class_auc:.4f}")
                        per_class_auc.append(class_auc)
            
            # Create ROC curve visualization
            create_roc_curve_plot(y_true_bin, y_proba_filtered, unique_labels, 
                                roc_auc, roc_auc_macro, model_path)
            
        else:
            print("Only one class present in test set - AUC-ROC not meaningful")
            roc_auc = 0.5
            
    except Exception as e:
        print(f"Error calculating AUC-ROC: {e}")
        roc_auc = 0.0
    
    # Additional comprehensive metrics
    try:
        # Precision, Recall, F1 scores
        precision_weighted = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        print(f"\nComprehensive Test Metrics:")
        print(f"Accuracy: {test_acc:.6f}")
        print(f"Precision (weighted): {precision_weighted:.6f}")
        print(f"Recall (weighted): {recall_weighted:.6f}")
        print(f"F1-Score (weighted): {f1_weighted:.6f}")
        print(f"AUC-ROC (weighted): {roc_auc:.6f}")
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Save metrics to JSON
        metrics = {
            'test_accuracy': float(test_acc),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'auc_roc_weighted': float(roc_auc),
            'auc_roc_macro': float(roc_auc_macro),
            'confusion_matrix': cm.tolist(),
            'present_classes': [datasets.emotions[i] for i in unique_labels],
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        with open(f"test_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to JSON file")
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating additional metrics: {e}")
        return {'test_accuracy': test_acc, 'auc_roc': roc_auc}


def main():
    # Configuration
    BATCH_SIZE = 32
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
    print("Preparing data loaders...")
    _, _, test_loader = get_validation_test_split_dataloaders(dataset, VAL_SPLIT, TEST_SPLIT, BATCH_SIZE, NUM_WORKERS)

    # Test the model on test set
    print("Starting final testing...")
    fin_test(test_loader, model_path="model_efb0.pth", device=DEVICE.type)
    
if __name__ == "__main__":
    main()