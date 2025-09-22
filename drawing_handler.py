import cv2
import numpy as np

ERASER_SIZE = 70

class DrawingHandler:
    def __init__(self, frame_height, frame_width):
        self.canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        self.prev_point = None
        
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.fill(0)
        
    def draw_thumb_tip_indicator(self, image, hand_landmarks):
        """Draw purple dot at thumb tip position with offset"""
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        
        offset_x = (thumb_tip.x - thumb_mcp.x) * 0.15
        offset_y = (thumb_tip.y - thumb_mcp.y) * 0.15
        
        h, w = image.shape[:2]
        dot_x = int((1.0 - (thumb_tip.x + offset_x)) * w)
        dot_y = int((thumb_tip.y + offset_y) * h)
        
        cv2.circle(image, (dot_x, dot_y), 8, (128, 0, 128), -1)
        cv2.circle(image, (dot_x, dot_y), 10, (255, 255, 255), 2)

    def handle_drawing(self, hand_landmarks, frame_counter, draw_interval=1):
        """Handle finger drawing functionality"""
        if frame_counter % draw_interval != 0:
            return self.prev_point
        
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        
        offset_x = (thumb_tip.x - thumb_mcp.x) * 0.15
        offset_y = (thumb_tip.y - thumb_mcp.y) * 0.15
        
        h, w = self.canvas.shape[:2]
        current_x = int((1.0 - (thumb_tip.x + offset_x)) * w)
        current_y = int((thumb_tip.y + offset_y) * h)
        
        if self.prev_point is not None:
            smooth_factor = 0.85
            finger_x = int(self.prev_point[0] * (1 - smooth_factor) + current_x * smooth_factor)
            finger_y = int(self.prev_point[1] * (1 - smooth_factor) + current_y * smooth_factor)
            
            distance = ((finger_x - self.prev_point[0])**2 + (finger_y - self.prev_point[1])**2)**0.5
            if distance > 4:
                cv2.line(self.canvas, self.prev_point, (finger_x, finger_y), (50, 255, 50), 8)
        else:
            finger_x, finger_y = current_x, current_y
        
        self.prev_point = (finger_x, finger_y)
        return self.prev_point

    def handle_erasing(self, hand_landmarks, frame_counter, draw_interval=1):
        """Handle eraser functionality"""
        if frame_counter % draw_interval != 0:
            return self.prev_point
        
        landmarks = [hand_landmarks.landmark[i] for i in [7, 8, 11, 12]]
        
        center_x = sum(lm.x for lm in landmarks) / 4
        center_y = sum(lm.y for lm in landmarks) / 4
        
        h, w = self.canvas.shape[:2]
        eraser_x = int((1.0 - center_x) * w)
        eraser_y = int(center_y * h)
        
        if self.prev_point is not None:
            smooth_factor = 0.85
            eraser_x = int(self.prev_point[0] * (1 - smooth_factor) + eraser_x * smooth_factor)
            eraser_y = int(self.prev_point[1] * (1 - smooth_factor) + eraser_y * smooth_factor)
        
        cv2.circle(self.canvas, (eraser_x, eraser_y), ERASER_SIZE, (0, 0, 0), -1)
        
        self.prev_point = (eraser_x, eraser_y)
        return self.prev_point

    def draw_eraser_indicator(self, image, hand_landmarks):
        """Draw eraser circle indicator"""
        landmarks = [hand_landmarks.landmark[i] for i in [7, 8, 11, 12]]
        
        center_x = sum(lm.x for lm in landmarks) / 4
        center_y = sum(lm.y for lm in landmarks) / 4
        
        h, w = image.shape[:2]
        circle_x = int((1.0 - center_x) * w)
        circle_y = int(center_y * h)
        
        cv2.circle(image, (circle_x, circle_y), ERASER_SIZE, (0, 0, 255), 3)
        cv2.circle(image, (circle_x, circle_y), 5, (0, 0, 255), -1)

    def reset_drawing_state(self):
        """Reset drawing state when gesture changes"""
        self.prev_point = None