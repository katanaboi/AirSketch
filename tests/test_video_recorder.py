import pytest
import os
from video_recorder import VideoRecorder
from unittest.mock import patch, MagicMock

def test_video_recorder_init():
    recorder = VideoRecorder()
    assert recorder.is_recording is False
    assert recorder.video_writer is None
    assert recorder.output_dir == "recordings"

@patch('cv2.VideoWriter')
@patch('os.makedirs')
def test_start_recording(mock_makedirs, mock_video_writer_class):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_video_writer_class.return_value = mock_writer
    
    recorder = VideoRecorder()
    filename = recorder.start_recording(640, 480)
    
    assert recorder.is_recording is True
    assert recorder.video_writer is not None
    assert "recordings" in filename
    assert mock_makedirs.called
    assert mock_video_writer_class.called

def test_stop_recording():
    recorder = VideoRecorder()
    mock_writer = MagicMock()
    recorder.video_writer = mock_writer
    recorder.is_recording = True
    
    recorder.stop_recording()
    
    assert recorder.is_recording is False
    assert recorder.video_writer is None
    assert mock_writer.release.called

def test_write_frame():
    recorder = VideoRecorder()
    mock_writer = MagicMock()
    recorder.video_writer = mock_writer
    recorder.is_recording = True
    
    mock_frame = MagicMock()
    recorder.write_frame(mock_frame)
    
    mock_writer.write.assert_called_with(mock_frame)

def test_write_frame_not_recording():
    recorder = VideoRecorder()
    mock_writer = MagicMock()
    recorder.video_writer = mock_writer
    recorder.is_recording = False
    
    mock_frame = MagicMock()
    recorder.write_frame(mock_frame)
    
    assert not mock_writer.write.called

@patch.object(VideoRecorder, 'start_recording')
@patch.object(VideoRecorder, 'stop_recording')
def test_toggle_recording(mock_stop, mock_start):
    recorder = VideoRecorder()
    
    # Toggle on
    recorder.is_recording = False
    recorder.toggle_recording(640, 480)
    mock_start.assert_called_with(640, 480)
    
    # Toggle off
    recorder.is_recording = True
    recorder.toggle_recording(640, 480)
    assert mock_stop.called
