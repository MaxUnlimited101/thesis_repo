import matplotlib
matplotlib.use('Agg')
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import io
from PIL import Image

from educator import app, predictions_log, predictions_lock, calculate_overall_statistics, generate_plot


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def clean_predictions():
    """Clear predictions_log before and after each test"""
    async with predictions_lock:
        predictions_log.clear()
    yield
    async with predictions_lock:
        predictions_log.clear()


# ============== UNIT TESTS ==============

class TestHealthCheck:
    def test_health_endpoint(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"message": "ok"}


class TestEmotionsEndpoint:
    @pytest.mark.asyncio
    async def test_receive_valid_emotions(self, client, clean_predictions):
        """Test receiving valid emotion data"""
        test_data = {
            "id": "student_123",
            "predictions": {
                "happy": 0.8,
                "neutral": 0.1,
                "sad": 0.05,
                "angry": 0.05
            }
        }
        
        response = client.post("/api/emotions", json=test_data)
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        
        # Verify data was stored
        async with predictions_lock:
            assert len(predictions_log) == 1
            student_id, timestamp, predictions = predictions_log[0]
            assert student_id == "student_123"
            assert predictions == test_data["predictions"]
            assert isinstance(timestamp, int)
    
    @pytest.mark.asyncio
    async def test_receive_invalid_json(self, client):
        """Test handling of invalid JSON"""
        response = client.post(
            "/api/emotions",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_receive_multiple_predictions(self, client, clean_predictions):
        """Test receiving multiple predictions from same student"""
        student_id = "student_456"
        
        for i in range(3):
            test_data = {
                "id": student_id,
                "predictions": {"happy": 0.5 + i * 0.1}
            }
            response = client.post("/api/emotions", json=test_data)
            assert response.status_code == 200
        
        async with predictions_lock:
            assert len(predictions_log) == 3
            assert all(sid == student_id for sid, _, _ in predictions_log)
