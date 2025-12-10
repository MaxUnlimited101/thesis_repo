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


class TestPrediction:
    @pytest.fixture
    def mock_model(self):
        """Create a mock model"""
        model = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)
        
        # Mock output: batch_size=1, num_classes=8
        mock_output = torch.randn(1, 8)
        model.return_value = mock_output
        
        return model
    
    def test_predict_output_format(self, mock_model):
        """Test that predict returns correct format"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = predict(frame, mock_model, device='cpu')
        
        # Should return dict with 8 emotions
        assert isinstance(result, dict)
        assert len(result) == 8
        assert all(isinstance(v, float) for v in result.values())
    
    def test_predict_probabilities_sum_to_one(self, mock_model):
        """Test that probabilities approximately sum to 1"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = predict(frame, mock_model, device='cpu')
        
        total_prob = sum(result.values())
        assert abs(total_prob - 1.0) < 0.01
    
    def test_predict_emotion_names(self, mock_model):
        """Test that all expected emotion names are present"""
        from student import CLASS_NAMES
        
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = predict(frame, mock_model, device='cpu')
        
        for emotion in CLASS_NAMES:
            assert emotion in result


class TestSendToServer:
    @patch('student.requests.post')
    def test_send_successful(self, mock_post):
        """Test successful data transmission"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        test_data = {
            "id": "test_student",
            "predictions": {"happy": 0.8}
        }
        
        # Should not raise exception
        send_to_server(test_data)
        
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs['json'] == test_data
        assert call_kwargs['timeout'] == 5
    
    @patch('student.requests.post')
    def test_send_failure(self, mock_post):
        """Test handling of network failure"""
        mock_post.side_effect = requests.exceptions.RequestException("Network error")
        
        test_data = {"id": "test", "predictions": {}}
        
        # Should handle exception gracefully
        send_to_server(test_data)  # Should not raise
    
    @patch('student.requests.post')
    def test_send_timeout(self, mock_post):
        """Test handling of timeout"""
        mock_post.side_effect = requests.exceptions.Timeout("Timeout")
        
        test_data = {"id": "test", "predictions": {}}
        
        # Should handle timeout gracefully
        send_to_server(test_data)
