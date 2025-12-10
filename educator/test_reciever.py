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
