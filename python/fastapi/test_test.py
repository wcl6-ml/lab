import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np
from serve.app import app

# --- FIXTURES ---

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_model():
    mock = Mock()
    # Simulate Isolation Forest: negative values are anomalies
    mock.predict.return_value = np.array([-0.5, 0.5, 0.2, 0.2]) 
    return mock

@pytest.fixture
def auth_headers():
    return {"API_KEY": TEST_API_KEY}

# 1. Grab the key from CI env, or use a local fallback
TEST_API_KEY = os.getenv("API_KEY", "ci-test-dummy-key")

@pytest.fixture(autouse=True)
def sync_app_api_key():
    """
    Force the app's internal API_KEY variable to match our TEST_API_KEY.
    This solves the 'None != key' issue during import.
    """
    with patch("serve.app.API_KEY", TEST_API_KEY):
        yield

@pytest.fixture
def sample_request_payload():
    """Sample prediction request."""
    return {
        "features": [
            [0.1] * 29,
            [0.2] * 29,
            [0.3] * 29,
            [0.4] * 29
        ],
        "batch_id": "test_batch_001"  # ADD THIS LINE
    }

# --- TEST CASES ---

def test_predict_success(client, mock_model, sample_request_payload, auth_headers):
    """Test successful prediction."""
    from datetime import datetime
    
    # Create mock drift detector
    mock_drift_detector = MagicMock()
    mock_drift_detector.feature_names = [f'V{i}' for i in range(1, 29)]
    mock_drift_detector.detect_drift.return_value = {
        'overall_psi': 0.05,
        'drift_detected': False
    }
    
    with patch('serve.app.model', mock_model), \
        patch('serve.app.model_metadata', {'version': '1.0', 'startup_time': datetime.now()}), \
        patch('serve.app.drift_detector', mock_drift_detector), \
        patch('builtins.open', MagicMock()):
        
        response = client.post("/predict",
                                json=sample_request_payload, 
                                headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "predictions" in data
        assert "anomaly_scores" in data
        assert "model_version" in data
        assert "inference_time_ms" in data
        assert "psi_score" in data
        assert "batch_id" in data
        
        # Check predictions
        assert len(data["predictions"]) == 4
        assert all(p in [0, 1] for p in data["predictions"])
        
        # Verify model.predict was called
        mock_model.predict.assert_called_once()

# def test_scenario_unauthorized(client):
#     """Scenario: Request without proper API Key"""
#     payload = {"features": [[0.1]*29]}
#     # No headers provided
#     response = client.post("/predict", json=payload)
    
#     assert response.status_code == 401
#     assert response.json()["detail"] == "Invalid API key"

# def test_scenario_model_not_found(client, valid_headers):
#     """Scenario: API is running but model failed to load"""
#     with patch('serve.app.model', None):
#         payload = {"features": [[0.1]*29], "batch_id": "lab_002"}
#         response = client.post("/predict", json=payload, headers=valid_headers)
        
#         # 503 Service Unavailable is the standard for missing dependencies
#         assert response.status_code == 503
#         assert "Model not loaded" in response.json()["detail"]

