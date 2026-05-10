import pytest
import numpy as np
from drawing_handler import DrawingHandler
from unittest.mock import patch, MagicMock

def test_drawing_handler_init():
    handler = DrawingHandler(480, 640)
    assert handler.canvas.shape == (480, 640, 3)
    assert handler.prev_point is None

def test_clear_canvas():
    handler = DrawingHandler(100, 100)
    handler.canvas.fill(255)
    handler.clear_canvas()
    assert np.all(handler.canvas == 0)

def test_reset_drawing_state():
    handler = DrawingHandler(100, 100)
    handler.prev_point = (50, 50)
    handler.reset_drawing_state()
    assert handler.prev_point is None

@patch('cv2.line')
def test_handle_drawing(mock_line, mock_landmarks):
    handler = DrawingHandler(480, 640)
    
    # First call: initializes prev_point
    handler.handle_drawing(mock_landmarks, 1)
    assert handler.prev_point is not None
    assert not mock_line.called # No line on first point
    
    # Second call: should draw line
    prev_point = handler.prev_point
    # Move the thumb tip to trigger distance > 4
    mock_landmarks.landmark[4].x += 0.1
    
    handler.handle_drawing(mock_landmarks, 2)
    assert mock_line.called
    # Check that it draws from prev_point
    args, kwargs = mock_line.call_args
    assert args[1] == prev_point

@patch('cv2.circle')
def test_handle_erasing(mock_circle, mock_landmarks):
    handler = DrawingHandler(480, 640)
    handler.handle_erasing(mock_landmarks, 1)
    
    assert mock_circle.called
    # Eraser should draw on canvas (args[0] is handler.canvas)
    assert np.array_equal(mock_circle.call_args[0][0], handler.canvas)
    # Color should be black (0, 0, 0)
    assert mock_circle.call_args[0][3] == (0, 0, 0)

@patch('cv2.circle')
def test_draw_thumb_tip_indicator(mock_circle, mock_landmarks, mock_image):
    handler = DrawingHandler(480, 640)
    handler.draw_thumb_tip_indicator(mock_image, mock_landmarks)
    
    # Should draw two circles (dot and outline)
    assert mock_circle.call_count == 2
    # Verify first circle is on the provided image, not the canvas
    assert mock_circle.call_args_list[0][0][0] is mock_image
