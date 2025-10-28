import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetB0(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB0, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
        self.backbone._fc = nn.Linear(self.backbone._fc.in_features, 8)

    def forward(self, x):
        return self.backbone(x)