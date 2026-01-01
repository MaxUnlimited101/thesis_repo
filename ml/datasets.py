import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import re

emotions = ('angry', 'disgust', 'contempt', 'fear', 'happy', 'neutral', 
            'sad', 'surprise')

# Emotion to index mapping
emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}

class BaseEmotionDataset(Dataset):
    """Base class for emotion datasets with common functionality"""
    
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.emotion_to_idx = emotion_to_idx
        self.emotions = emotions
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """Override in child classes to load dataset-specific samples"""
        raise NotImplementedError("Subclasses must implement _load_samples method")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Apply target transform
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    
    def get_class_counts(self):
        """Get counts for each emotion class"""
        counts = {emotion: 0 for emotion in self.emotions}
        for _, label in self.samples:
            if isinstance(label, int):
                emotion = self.emotions[label]
            else:
                emotion = label
            counts[emotion] += 1
        return counts


class AffectNetDataset(BaseEmotionDataset):
    """AffectNet dataset class - uses CSV labels file"""
    
    def __init__(self, root_dir, transform=None, target_transform=None):
        super().__init__(root_dir, transform, target_transform)
    
    def _load_samples(self):
        """Load samples from AffectNet CSV labels file"""
        labels_file = os.path.join(self.root_dir, 'affectnet', 'labels.csv')
        
        if not os.path.exists(labels_file):
            print(f"Warning: Labels file not found at {labels_file}")
            return
        
        # Load CSV
        df = pd.read_csv(labels_file)
        
        for _, row in df.iterrows():
            image_path = os.path.join(self.root_dir, 'affectnet', row['pth'])
            emotion_name = row['label'].lower()
            
            # Map emotion name to our standard emotions
            emotion_mapping = {
                'anger': 'angry',
                'happiness': 'happy',
                'sadness': 'sad'
            }
            
            if emotion_name in emotion_mapping:
                emotion_name = emotion_mapping[emotion_name]
            
            # Only include samples with valid emotions
            if emotion_name in self.emotion_to_idx and os.path.exists(image_path):
                label = self.emotion_to_idx[emotion_name]
                self.samples.append((image_path, label))
        
        print(f"AffectNet: Loaded {len(self.samples)} samples")


class CKPlusDataset(BaseEmotionDataset):
    """CK+ dataset class - organized in emotion folders"""
    
    def __init__(self, root_dir, transform=None, target_transform=None):
        super().__init__(root_dir, transform, target_transform)
    
    def _load_samples(self):
        """Load samples from CK+ emotion folders"""
        # CK+ uses 'sadness' instead of 'sad' 
        ck_emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        
        for emotion_folder in ck_emotions:
            folder_path = os.path.join(self.root_dir, emotion_folder)
            
            if not os.path.exists(folder_path):
                continue
            
            # Map CK+ emotion names to our standard names
            emotion_name = emotion_folder
            if emotion_name == 'anger':
                emotion_name = 'angry'
            elif emotion_name == 'sadness':
                emotion_name = 'sad'
            
            # Skip if not in our emotion list
            if emotion_name not in self.emotion_to_idx:
                continue
                
            label = self.emotion_to_idx[emotion_name]
            
            # Get all image files in the folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    self.samples.append((image_path, label))
        
        print(f"CK+: Loaded {len(self.samples)} samples")


class FER2013Dataset(BaseEmotionDataset):
    """FER-2013 dataset class - organized in train/test folders with emotion subfolders"""
    
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.split = split  # 'train' or 'test'
        super().__init__(root_dir, transform, target_transform)
    
    def _load_samples(self):
        """Load samples from FER-2013 train/test emotion folders"""
        split_path = os.path.join(self.root_dir, self.split)
        
        if not os.path.exists(split_path):
            print(f"Warning: Split directory not found at {split_path}")
            return
        
        # FER-2013 uses our standard emotion names
        for emotion_name in self.emotions:
            folder_path = os.path.join(split_path, emotion_name)
            
            if not os.path.exists(folder_path):
                continue
                
            label = self.emotion_to_idx[emotion_name]
            
            # Get all image files in the folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    self.samples.append((image_path, label))
        
        print(f"FER-2013 ({self.split}): Loaded {len(self.samples)} samples")


class JAFFEDataset(BaseEmotionDataset):
    """JAFFE dataset class - files named with emotion codes"""
    
    def __init__(self, root_dir, transform=None, target_transform=None):
        super().__init__(root_dir, transform, target_transform)
    
    def _load_samples(self):
        """Load samples from JAFFE with emotion code parsing"""
        jaffe_path = os.path.join(self.root_dir, 'jaffe')
        
        if not os.path.exists(jaffe_path):
            print(f"Warning: JAFFE directory not found at {jaffe_path}")
            return
        
        # JAFFE emotion code mapping
        # Format: XX.YY#.###.tiff where YY is emotion code
        emotion_codes = {
            'AN': 'angry',    # Anger
            'DI': 'disgust',  # Disgust
            'FE': 'fear',     # Fear
            'HA': 'happy',    # Happy
            'NE': 'neutral',  # Neutral
            'SA': 'sad',      # Sadness
            'SU': 'surprise'  # Surprise
        }
        
        for filename in os.listdir(jaffe_path):
            if filename.lower().endswith('.tiff'):
                # Parse emotion from filename (e.g., "KA.AN1.39.tiff" -> "AN" -> "angry")
                match = re.match(r'[A-Z]{2}\.([A-Z]{2})\d+\.\d+\.tiff', filename)
                
                if match:
                    emotion_code = match.group(1)
                    if emotion_code in emotion_codes:
                        emotion_name = emotion_codes[emotion_code]
                        if emotion_name in self.emotion_to_idx:
                            label = self.emotion_to_idx[emotion_name]
                            image_path = os.path.join(jaffe_path, filename)
                            self.samples.append((image_path, label))
        
        print(f"JAFFE: Loaded {len(self.samples)} samples")


class KDEFDataset(BaseEmotionDataset):
    """KDEF dataset class - organized in emotion folders"""
    
    def __init__(self, root_dir, transform=None, target_transform=None):
        super().__init__(root_dir, transform, target_transform)
    
    def _load_samples(self):
        """Load samples from KDEF emotion folders"""
        # KDEF uses our standard emotion names except no 'contempt'
        kdef_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        for emotion_name in kdef_emotions:
            folder_path = os.path.join(self.root_dir, emotion_name)
            
            if not os.path.exists(folder_path):
                continue
            
            # Skip if not in our emotion list  
            if emotion_name not in self.emotion_to_idx:
                continue
                
            label = self.emotion_to_idx[emotion_name]
            
            # Get all image files in the folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    self.samples.append((image_path, label))
        
        print(f"KDEF: Loaded {len(self.samples)} samples")


class NHFIERDataset(BaseEmotionDataset):
    """NHFIER dataset class - organized in emotion folders"""
    
    def __init__(self, root_dir, transform=None, target_transform=None):
        super().__init__(root_dir, transform, target_transform)
    
    def _load_samples(self):
        """Load samples from NHFIER emotion folders"""
        # NHFIER emotion folder names to our standard names mapping
        nhfier_emotion_mapping = {
            'anger': 'angry',
            'contempt': 'contempt',
            'disgust': 'disgust',
            'fear': 'fear',
            'happiness': 'happy',
            'neutrality': 'neutral',
            'sadness': 'sad',
            'surprise': 'surprise'
        }
        
        for folder_name, emotion_name in nhfier_emotion_mapping.items():
            folder_path = os.path.join(self.root_dir, folder_name)
            
            if not os.path.exists(folder_path):
                continue
            
            # Skip if not in our emotion list
            if emotion_name not in self.emotion_to_idx:
                continue
                
            label = self.emotion_to_idx[emotion_name]
            
            # Get all image files in the folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    self.samples.append((image_path, label))
        
        print(f"NHFIER: Loaded {len(self.samples)} samples")


class RAFDBDataset(BaseEmotionDataset):
    """RAF-DB dataset class - uses CSV labels with numerical emotion codes"""
    
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.split = split  # 'train' or 'test'
        super().__init__(root_dir, transform, target_transform)
    
    def _load_samples(self):
        """Load samples from RAF-DB CSV labels file"""
        # RAF-DB label mapping (1-indexed in CSV)
        rafdb_label_mapping = {
            1: 'surprise',
            2: 'fear', 
            3: 'disgust',
            4: 'happy',
            5: 'sad',
            6: 'angry',
            7: 'neutral'
        }
        
        # Load appropriate CSV file
        labels_file = os.path.join(self.root_dir, f'{self.split}_labels.csv')
        
        if not os.path.exists(labels_file):
            print(f"Warning: Labels file not found at {labels_file}")
            return
        
        # Load CSV
        df = pd.read_csv(labels_file)
        
        for _, row in df.iterrows():
            # Image path construction
            image_filename = row['image']
            image_path = os.path.join(self.root_dir, 'DATASET', self.split, str(row['label']), image_filename)
            
            # Map numerical label to emotion name
            emotion_label_num = int(row['label'])
            if emotion_label_num in rafdb_label_mapping:
                emotion_name = rafdb_label_mapping[emotion_label_num]
                
                # Only include samples with valid emotions and existing files
                if emotion_name in self.emotion_to_idx and os.path.exists(image_path):
                    label = self.emotion_to_idx[emotion_name]
                    self.samples.append((image_path, label))
        
        print(f"RAF-DB ({self.split}): Loaded {len(self.samples)} samples")


def combine_datasets(datasets):
    """Combine multiple datasets into one"""
    combined = torch.utils.data.ConcatDataset(datasets)
    print(f"Combined dataset size: {len(combined)} samples")
    return combined


# Convenience functions to create datasets
def create_affectnet_dataset(root_dir, transform=None):
    """Create AffectNet dataset"""
    return AffectNetDataset(root_dir, transform=transform)

def create_ckplus_dataset(root_dir, transform=None):
    """Create CK+ dataset"""
    return CKPlusDataset(root_dir, transform=transform)

def create_fer2013_dataset(root_dir, split='train', transform=None):
    """Create FER-2013 dataset"""
    return FER2013Dataset(root_dir, split=split, transform=transform)

def create_jaffe_dataset(root_dir, transform=None):
    """Create JAFFE dataset"""
    return JAFFEDataset(root_dir, transform=transform)

def create_kdef_dataset(root_dir, transform=None):
    """Create KDEF dataset"""
    return KDEFDataset(root_dir, transform=transform)

def create_nhfier_dataset(root_dir, transform=None):
    """Create NHFIER dataset"""  
    return NHFIERDataset(root_dir, transform=transform)

def create_rafdb_dataset(root_dir, split='train', transform=None):
    """Create RAF-DB dataset"""
    return RAFDBDataset(root_dir, split=split, transform=transform)

def get_default_transform():
    """Get default image transforms for emotion recognition"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

# Example usage and testing
if __name__ == "__main__":
    base_path = "/Users/maksymnosal/thesis_app"
    
    # Test all datasets
    print("Testing dataset classes...")
    
    # AffectNet
    affectnet = create_affectnet_dataset(
        os.path.join(base_path, "AffectNet"), 
        transform=get_default_transform()
    )
    print(f"AffectNet samples: {len(affectnet)}")
    print(f"AffectNet class counts: {affectnet.get_class_counts()}")
    
    # CK+
    ckplus = create_ckplus_dataset(
        os.path.join(base_path, "CK+"), 
        transform=get_default_transform()
    )
    print(f"CK+ samples: {len(ckplus)}")
    print(f"CK+ class counts: {ckplus.get_class_counts()}")
    
    # FER-2013 Train
    fer_train = create_fer2013_dataset(
        os.path.join(base_path, "FER-2013"), 
        split='train',
        transform=get_default_transform()
    )
    print(f"FER-2013 train samples: {len(fer_train)}")
    print(f"FER-2013 train class counts: {fer_train.get_class_counts()}")
    
    # FER-2013 Test
    fer_test = create_fer2013_dataset(
        os.path.join(base_path, "FER-2013"), 
        split='test',
        transform=get_default_transform()
    )
    print(f"FER-2013 test samples: {len(fer_test)}")
    print(f"FER-2013 test class counts: {fer_test.get_class_counts()}")
    
    # JAFFE
    jaffe = create_jaffe_dataset(
        os.path.join(base_path, "JAFFE"), 
        transform=get_default_transform()
    )
    print(f"JAFFE samples: {len(jaffe)}")
    print(f"JAFFE class counts: {jaffe.get_class_counts()}")
    
    # KDEF
    kdef = create_kdef_dataset(
        os.path.join(base_path, "KDEF"),
        transform=get_default_transform()
    )
    print(f"KDEF samples: {len(kdef)}")
    print(f"KDEF class counts: {kdef.get_class_counts()}")
    
    # NHFIER
    nhfier = create_nhfier_dataset(
        os.path.join(base_path, "NHFIER"),
        transform=get_default_transform()
    )
    print(f"NHFIER samples: {len(nhfier)}")
    print(f"NHFIER class counts: {nhfier.get_class_counts()}")
    
    # RAF-DB Train
    rafdb_train = create_rafdb_dataset(
        os.path.join(base_path, "RAF-DB"),
        split='train',
        transform=get_default_transform()
    )
    print(f"RAF-DB train samples: {len(rafdb_train)}")
    print(f"RAF-DB train class counts: {rafdb_train.get_class_counts()}")
    
    # RAF-DB Test
    rafdb_test = create_rafdb_dataset(
        os.path.join(base_path, "RAF-DB"),
        split='test',
        transform=get_default_transform()
    )
    print(f"RAF-DB test samples: {len(rafdb_test)}")
    print(f"RAF-DB test class counts: {rafdb_test.get_class_counts()}")
    
    # Test loading a sample from each dataset
    print("\nTesting sample loading...")
    
    if len(affectnet) > 0:
        image, label = affectnet[0]
        print(f"AffectNet sample - Image shape: {image.shape}, Label: {label} ({emotions[label]})")
    
    if len(ckplus) > 0:
        image, label = ckplus[0]
        print(f"CK+ sample - Image shape: {image.shape}, Label: {label} ({emotions[label]})")
    
    if len(fer_train) > 0:
        image, label = fer_train[0]
        print(f"FER-2013 sample - Image shape: {image.shape}, Label: {label} ({emotions[label]})")
    
    if len(jaffe) > 0:
        image, label = jaffe[0]
        print(f"JAFFE sample - Image shape: {image.shape}, Label: {label} ({emotions[label]})")
    
    if len(kdef) > 0:
        image, label = kdef[0]
        print(f"KDEF sample - Image shape: {image.shape}, Label: {label} ({emotions[label]})")
    
    if len(nhfier) > 0:
        image, label = nhfier[0]
        print(f"NHFIER sample - Image shape: {image.shape}, Label: {label} ({emotions[label]})")
    
    if len(rafdb_train) > 0:
        image, label = rafdb_train[0]
        print(f"RAF-DB sample - Image shape: {image.shape}, Label: {label} ({emotions[label]})")

