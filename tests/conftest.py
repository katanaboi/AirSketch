import pytest
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def mock_image():
    """Provides a blank 480x640 BGR image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_landmarks():
    """Provides a mock MediaPipe HandLandmarks object."""
    class MockLandmark:
        def __init__(self, x, y, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    mock_obj = MagicMock()
    # Create 21 landmarks (standard for MediaPipe hands)
    mock_obj.landmark = [MockLandmark(0.5, 0.5) for _ in range(21)]
    
    # Thumb tip (4) and Thumb MCP (2) are used for drawing logic
    mock_obj.landmark[4] = MockLandmark(0.6, 0.4)
    mock_obj.landmark[2] = MockLandmark(0.5, 0.5)
    
    return mock_obj
