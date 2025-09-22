import os
import cv2
from datetime import datetime

class VideoRecorder:
    def __init__(self):
        self.is_recording = False
        self.video_writer = None
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)

    def start_recording(self, frame_width, frame_height, fps=20.0):
        """Start video recording"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"hand_gesture_{timestamp}.mp4")

        for codec in ["mp4v", "h264", "XVID"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
            if self.video_writer.isOpened():
                break
        else:
            print("Error: Could not open video writer with any codec.")
            return None

        self.is_recording = True
        print(f"Started recording: {filename}")
        return filename

    def stop_recording(self):
        """Stop video recording"""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print("Stopped recording")

    def write_frame(self, frame):
        """Write frame to video file"""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def toggle_recording(self, frame_width, frame_height):
        """Toggle video recording on/off"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording(frame_width, frame_height)