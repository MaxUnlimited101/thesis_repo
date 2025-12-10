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


class TestStatisticsEndpoint:
    @pytest.mark.asyncio
    async def test_statistics_empty(self, client, clean_predictions):
        """Test statistics with no data"""
        response = client.get("/api/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_students"] == 0
        assert data["total_predictions"] == 0
        assert data["active_sessions"] == 0
    
    @pytest.mark.asyncio
    async def test_statistics_with_data(self, client, clean_predictions):
        """Test statistics with sample data"""
        # Add test data
        async with predictions_lock:
            predictions_log.append(("student_1", 1000, {"happy": 0.8}))
            predictions_log.append(("student_1", 1005, {"happy": 0.7}))
            predictions_log.append(("student_2", 1010, {"sad": 0.6}))
        
        response = client.get("/api/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_students"] == 2
        assert data["total_predictions"] == 3
        assert data["active_sessions"] == 2
        assert data["students"]["student_1"] == 2
        assert data["students"]["student_2"] == 1


class TestCalculateOverallStatistics:
    @pytest.mark.asyncio
    async def test_empty_log(self, clean_predictions):
        """Test statistics calculation with empty log"""
        stats = await calculate_overall_statistics()
        assert stats == {}
    
    @pytest.mark.asyncio
    async def test_single_student(self, clean_predictions):
        """Test statistics for single student"""
        async with predictions_lock:
            predictions_log.append(("student_1", 1000, {"happy": 0.8, "sad": 0.2}))
            predictions_log.append(("student_1", 1005, {"happy": 0.6, "sad": 0.4}))
        
        stats = await calculate_overall_statistics()
        
        assert "student_1" in stats
        assert stats["student_1"]["happy"] == pytest.approx(0.7)
        assert stats["student_1"]["sad"] == pytest.approx(0.3)
    
    @pytest.mark.asyncio
    async def test_multiple_students(self, clean_predictions):
        """Test statistics for multiple students"""
        async with predictions_lock:
            predictions_log.append(("student_1", 1000, {"happy": 1.0}))
            predictions_log.append(("student_2", 1005, {"happy": 0.5}))
        
        stats = await calculate_overall_statistics()
        
        assert len(stats) == 2
        assert stats["student_1"]["happy"] == 1.0
        assert stats["student_2"]["happy"] == 0.5


class TestPlotGeneration:
    @pytest.mark.asyncio
    async def test_generate_plot_empty_log(self, clean_predictions):
        """Test plot generation with empty log"""
        result = await generate_plot()
        assert result is None
    
    @pytest.mark.asyncio
    async def test_generate_plot_with_data(self, clean_predictions):
        """Test successful plot generation"""
        async with predictions_lock:
            predictions_log.append(("student_1", 1000, {
                "happy": 0.8, "sad": 0.1, "neutral": 0.1
            }))
            predictions_log.append(("student_1", 1005, {
                "happy": 0.6, "sad": 0.2, "neutral": 0.2
            }))
        
        plot_buffer = await generate_plot()
        
        assert plot_buffer is not None
        assert isinstance(plot_buffer, io.BytesIO)
        
        # Verify it's a valid image
        plot_buffer.seek(0)
        img = Image.open(plot_buffer)
        assert img.format == "PNG"
    
    @pytest.mark.asyncio
    async def test_generate_plot_specific_student(self, clean_predictions):
        """Test plot generation for specific student"""
        async with predictions_lock:
            predictions_log.append(("student_1", 1000, {"happy": 0.8}))
            predictions_log.append(("student_2", 1005, {"sad": 0.7}))
        
        plot_buffer = await generate_plot(student_id="student_1")
        assert plot_buffer is not None
        
        # Test non-existent student
        plot_buffer = await generate_plot(student_id="student_999")
        assert plot_buffer is None
    
    @pytest.mark.asyncio
    async def test_generate_cumulative_plot(self, clean_predictions):
        """Test cumulative plot generation"""
        async with predictions_lock:
            predictions_log.append(("student_1", 1000, {"happy": 0.5}))
            predictions_log.append(("student_1", 1005, {"happy": 0.5}))
        
        plot_buffer = await generate_plot(cumulative=True)
        assert plot_buffer is not None


class TestPlotEndpoints:
    @pytest.mark.asyncio
    async def test_plot_endpoint_no_data(self, client, clean_predictions):
        """Test plot endpoint with no data"""
        response = client.get("/api/plot")
        assert response.status_code == 404
        assert response.json()["error"] == "No data available"
    
    @pytest.mark.asyncio
    async def test_plot_endpoint_with_data(self, client, clean_predictions):
        """Test plot endpoint with data"""
        async with predictions_lock:
            predictions_log.append(("student_1", 1000, {"happy": 0.8}))
        
        response = client.get("/api/plot")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
    
    @pytest.mark.asyncio
    async def test_student_plot_endpoint(self, client, clean_predictions):
        """Test student-specific plot endpoint"""
        async with predictions_lock:
            predictions_log.append(("student_123", 1000, {"happy": 0.8}))
        
        response = client.get("/api/plot/student_123")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
    
    @pytest.mark.asyncio
    async def test_cumulative_plot_endpoint(self, client, clean_predictions):
        """Test cumulative plot endpoint"""
        async with predictions_lock:
            predictions_log.append(("student_1", 1000, {"happy": 0.5}))
        
        response = client.get("/api/plot?type=cumulative")
        assert response.status_code == 200


# # ============== INTEGRATION TESTS ==============

class TestEndToEndWorkflow:
    @pytest.mark.asyncio
    async def test_complete_workflow(self, client, clean_predictions):
        """Test complete workflow: receive data -> check stats -> get plot"""
        # Step 1: Send emotion data
        for i in range(3):
            data = {
                "id": "student_alpha",
                "predictions": {
                    "happy": 0.5 + i * 0.1,
                    "neutral": 0.3,
                    "sad": 0.2 - i * 0.05
                }
            }
            response = client.post("/api/emotions", json=data)
            assert response.status_code == 200
        
        # Step 2: Check statistics
        response = client.get("/api/statistics")
        assert response.status_code == 200
        stats = response.json()
        assert stats["total_students"] == 1
        assert stats["total_predictions"] == 3
        
        # Step 3: Get plot
        response = client.get("/api/plot/student_alpha")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


class TestConcurrentRequests:
    @pytest.mark.asyncio
    async def test_concurrent_emotion_submissions(self, client, clean_predictions):
        """Test handling concurrent emotion submissions"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def send_emotion(student_id):
            data = {
                "id": student_id,
                "predictions": {"happy": 0.5}
            }
            return client.post("/api/emotions", json=data)
        
        # Send 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(send_emotion, f"student_{i}")
                for i in range(10)
            ]
            responses = [f.result() for f in futures]
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        
        # Verify all data was stored
        async with predictions_lock:
            assert len(predictions_log) == 10


class TestDataPersistence:
    @pytest.mark.asyncio
    async def test_data_accumulation(self, client, clean_predictions):
        """Test that data accumulates correctly over time"""
        students = ["alice", "bob", "charlie"]
        
        for _ in range(5):
            for student in students:
                data = {
                    "id": student,
                    "predictions": {"happy": 0.7}
                }
                client.post("/api/emotions", json=data)
        
        response = client.get("/api/statistics")
        stats = response.json()
        
        assert stats["total_students"] == 3
        assert stats["total_predictions"] == 15
        for student in students:
            assert stats["students"][student] == 5
