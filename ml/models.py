import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, ResNet18_Weights, \
    resnet34, ResNet34_Weights, resnet50, ResNet50_Weights
from efficientnet_pytorch import EfficientNet


class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 8)

    def forward(self, x):
        return self.backbone(x)
    

class ResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet34, self).__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = resnet34(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 8)

    def forward(self, x):
        return self.backbone(x)
    
    
class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 8)

    def forward(self, x):
        return self.backbone(x)
    
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8, input_shape=(3, 224, 224)):
        super(SimpleCNN, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
    

class EfficientNetB0(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB0, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
        self.backbone._fc = nn.Linear(self.backbone._fc.in_features, 8)

    def forward(self, x):
        return self.backbone(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        self.backbone = torchvision.models.mobilenet_v2(pretrained=pretrained)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
    

class SimpleCNNv2(nn.Module):
    def __init__(self, num_classes=8, input_shape=(3, 224, 224)):
        super(SimpleCNNv2, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
    

class VGG11(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(VGG11, self).__init__()
        self.backbone = torchvision.models.vgg11(pretrained=pretrained)
        self.backbone.classifier[6] = nn.Linear(self.backbone.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)