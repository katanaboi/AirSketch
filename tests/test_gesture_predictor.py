import pytest
import numpy as np
from gesture_predictor import GesturePredictor
from unittest.mock import patch, MagicMock

@pytest.fixture
def predictor():
    with patch('gesture_predictor.tf.lite.Interpreter', create=True):
        with patch('gesture_predictor.joblib.load') as mock_joblib:
            with patch('builtins.open', MagicMock()):
                with patch('gesture_predictor.json.load') as mock_json:
                    mock_joblib.return_value.classes_ = ['Fist', 'Palm', 'Pen']
                    mock_json.return_value = {"threshold": 0.1}
                    p = GesturePredictor()
                    p.load_models()
                    return p

def test_prepare_landmarks(predictor, mock_landmarks):
    # landmarks has 21 points
    data = predictor._prepare_landmarks(mock_landmarks)
    
    assert data is not None
    assert data.shape == (1, 42)
    # Check normalization: wrist (landmark 0) should be (0,0) in the internal array
    # The internal array is reshaped to (21, 2) then flattened.
    # points[0] and points[1] are x, y of wrist.
    # Since we subtract wrist from all points, they should be 0.
    assert data[0][0] == 0.0
    assert data[0][1] == 0.0

def test_classify(predictor):
    # Mock classifier behavior
    predictor.classifier = MagicMock()
    predictor.classifier.get_output_details.return_value = [{"index": 0}]
    # Mocking softmax output for 3 classes: Fist, Palm, Pen
    # idx 2 is 'Pen'
    predictor.classifier.get_tensor.return_value = np.array([[0.1, 0.1, 0.8]])
    
    gesture, confidence = predictor._classify(np.zeros((1, 42)))
    
    assert gesture == 'Pen'
    assert confidence == 0.8

def test_predict_with_threshold_valid(predictor, mock_landmarks):
    with patch.object(predictor, '_get_reconstruction_error', return_value=0.05):
        with patch.object(predictor, '_classify', return_value=('Palm', 0.9)):
            gesture, confidence = predictor.predict_with_threshold(mock_landmarks)
            assert gesture == 'Palm'
            assert confidence == 0.9

def test_predict_with_threshold_anomaly(predictor, mock_landmarks):
    # Error 0.2 > threshold 0.1
    with patch.object(predictor, '_get_reconstruction_error', return_value=0.2):
        gesture, confidence = predictor.predict_with_threshold(mock_landmarks)
        assert gesture == '?'
        assert confidence == 0.0
