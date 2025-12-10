import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import requests
import sys
sys.path.append('.')
from student import (
    preprocess, predict, send_to_server, 
    list_available_cameras, select_camera
)


# ============== UNIT TESTS ==============

class TestPreprocessing:
    def test_preprocess_shape(self):
        """Test that preprocessing produces correct tensor shape"""
        # Create a dummy BGR frame (OpenCV format)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = preprocess(frame, device='cpu')
        
        # Should be (1, 3, 224, 224) - batch, channels, height, width
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == torch.float32
    
    def test_preprocess_normalization(self):
        """Test that values are normalized to [0, 1]"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        result = preprocess(frame, device='cpu')
        
        assert result.max() <= 1.0
        assert result.min() >= 0.0
        assert torch.allclose(result, torch.ones_like(result), atol=0.01)
    
    def test_preprocess_color_conversion(self):
        """Test BGR to RGB conversion"""
        # Create frame with distinct BGR values
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue channel in BGR
        
        result = preprocess(frame, device='cpu')
        
        # After BGR->RGB conversion, blue should be in channel 2
        # result shape is (1, 3, 224, 224)
        # Channel 2 should have highest values
        assert result[0, 2].mean() > result[0, 0].mean()
        assert result[0, 2].mean() > result[0, 1].mean()
