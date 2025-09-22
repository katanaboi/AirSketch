import cv2
import numpy as np

class UIHandler:
    @staticmethod
    def draw_prediction_on_hand(flipped_image, hand_landmarks, prediction, width, hand_label):
        """Draw prediction text near the hand with smooth positioning"""
        if not hand_landmarks:
            return

        height, width = flipped_image.shape[:2]

        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]

        center_x = int((1.0 - (wrist.x + middle_mcp.x) / 2) * width)
        center_y = int((wrist.y + middle_mcp.y) / 2 * height)

        text_x = center_x - 50
        text_y = center_y - 40

        text_x = max(10, min(text_x, width - 200))
        text_y = max(30, min(text_y, height - 20))

        # If no prediction, just show hand label
        if not prediction:
            cv2.putText(flipped_image, hand_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (200, 200, 100), 2, cv2.LINE_AA)
            return

        text_size = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        rect_x1 = text_x - 15
        rect_y1 = text_y - text_size[1] - 15
        rect_x2 = text_x + text_size[0] + 15
        rect_y2 = text_y + 10

        # Shadow effect
        shadow_offset = 3
        cv2.rectangle(flipped_image, (rect_x1 + shadow_offset, rect_y1 + shadow_offset),
                     (rect_x2 + shadow_offset, rect_y2 + shadow_offset), (0, 0, 0), -1)

        # Background with gradient
        overlay = flipped_image.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (40, 40, 40), -1)

        gradient = np.zeros_like(overlay[rect_y1:rect_y2, rect_x1:rect_x2])
        for i in range(gradient.shape[0]):
            alpha = i / gradient.shape[0]
            gradient[i, :] = (0, int(80 * (1 - alpha)), int(120 * (1 - alpha)))

        overlay[rect_y1:rect_y2, rect_x1:rect_x2] = cv2.addWeighted(
            overlay[rect_y1:rect_y2, rect_x1:rect_x2], 0.7, gradient, 0.3, 0)

        cv2.addWeighted(overlay, 0.8, flipped_image, 0.2, 0, flipped_image)

        # Border
        cv2.rectangle(flipped_image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 200, 255), 2)

        # Text
        cv2.putText(flipped_image, prediction, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Hand label
        hand_text_y = text_y + 30
        cv2.putText(flipped_image, hand_label, (text_x, hand_text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (200, 200, 100), 2, cv2.LINE_AA)

        # Connection line
        cv2.line(flipped_image, (center_x, center_y), (text_x + text_size[0] // 2, rect_y2),
                (200, 200, 200), 1, cv2.LINE_AA)

    @staticmethod
    def draw_legend(image, detection_mode, drawing_mode, dataset_mode):
        """Draw Apple-style legend"""
        h, w = image.shape[:2]
        
        # Legend background
        legend_w, legend_h = 280, 120
        x = w - legend_w - 20
        y = 20
        
        # Rounded rectangle background
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + legend_w, y + legend_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
        
        # Border
        cv2.rectangle(image, (x, y), (x + legend_w, y + legend_h), (200, 200, 200), 1)
        
        # Title
        cv2.putText(image, "AirSketch", (x + 15, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mode indicators
        modes = [
            ("W", "Drawing", drawing_mode, (0, 150, 255)),
            ("D", "Dataset", dataset_mode, (255, 150, 0)),
            ("ESC", "Exit", False, (255, 100, 100))
        ]
        
        for i, (key, name, active, color) in enumerate(modes):
            y_pos = y + 45 + i * 22
            
            # Key badge
            badge_color = color if active else (100, 100, 100)
            cv2.rectangle(image, (x + 15, y_pos - 12), (x + 35, y_pos + 2), badge_color, -1)
            cv2.putText(image, key, (x + 18, y_pos - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Mode name
            text_color = (255, 255, 255) if active else (150, 150, 150)
            cv2.putText(image, name, (x + 45, y_pos - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

    @staticmethod
    def draw_status_indicators(image, models_loaded, predictions, is_recording, fps):
        """Draw various status indicators on the image"""
        # FPS counter
        cv2.putText(image, f"FPS: {int(fps)}", (10, image.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Recording indicator
        if is_recording:
            cv2.circle(image, (image.shape[1] - 20, 20), 10, (0, 0, 255), -1)
            cv2.putText(image, "REC", (image.shape[1] - 40, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    @staticmethod
    def draw_data_collection_indicator(image, is_saving):
        """Draw data collection indicator"""
        if is_saving:
            cv2.circle(image, (50, 150), 15, (0, 255, 0), -1)
            cv2.putText(image, "SAVING", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)