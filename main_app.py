import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

from gesture_predictor import FastGesturePredictor
from utils import save_to_csv

os.makedirs("models/tflite", exist_ok=True)
gp = FastGesturePredictor(tflite_model_path="models/tflite/gesture_classifier.tflite")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class DatasetCreator:
    def __init__(self):
        self.dataset_mode = False
        self.manual_mode = False
        self.current_label = ""
        self.collecting = False
        self.frame_counter = 0

    def display_instructions(self, image):
        """Display current mode and instructions on the image"""
        height, width = image.shape[:2]

        cv2.rectangle(image, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (width - 10, 120), (255, 255, 255), 2)

        mode_text = "DATASET MODE: ON" if self.dataset_mode else "DATASET MODE: OFF"
        mode_color = (0, 255, 0) if self.dataset_mode else (0, 0, 255)
        cv2.putText(
            image, mode_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2
        )

        if self.dataset_mode:
            if self.manual_mode:
                if self.current_label:
                    status_text = f"MANUAL: {self.current_label} - Press SPACE to capture"
                    status_color = (0, 255, 0)
                else:
                    status_text = "Enter label first"
                    status_color = (255, 255, 255)
            else:
                if self.collecting:
                    status_text = f"COLLECTING: {self.current_label}"
                    status_color = (0, 255, 255)
                else:
                    status_text = "READY - Enter label and press SPACE to start"
                    status_color = (255, 255, 255)

            cv2.putText(
                image,
                status_text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                1,
            )
            cv2.putText(
                image,
                "Instructions:",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                image,
                "SPACE: Start/Stop | 'M': Manual | 'L': Change Label | ESC: Exit",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
        else:
            cv2.putText(
                image,
                "Press 'D' to enter Dataset Creation Mode",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                image,
                "ESC: Exit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    def get_label_input(self):
        """Get label input from user via console with confirmation"""
        if not self.collecting and not self.manual_mode:
            print("\n" + "=" * 50)
            print("DATASET CREATION MODE")
            print("=" * 50)

            while True:
                label = input(
                    "Enter label for data collection (or 'quit' to exit dataset mode): "
                ).strip()

                if label.lower() == "quit":
                    self.dataset_mode = False
                    return False
                elif label:
                    print(f"\nYou entered: '{label}'")
                    confirm = input("Is this correct? (y/n/edit): ").strip().lower()

                    if confirm == "y" or confirm == "yes":
                        self.current_label = label
                        print(f"✓ Label confirmed: '{self.current_label}'")
                        print(
                            "Press SPACE in the video window to start collecting data..."
                        )
                        return True
                    elif confirm == "edit" or confirm == "e":
                        new_label = input(f"Edit label (current: '{label}'): ").strip()
                        if new_label:
                            label = new_label
                            print(f"\nUpdated label: '{label}'")
                            confirm2 = input("Is this correct? (y/n): ").strip().lower()
                            if confirm2 == "y" or confirm2 == "yes":
                                self.current_label = label
                                print(f"✓ Label confirmed: '{self.current_label}'")
                                print(
                                    "Press SPACE in the video window to start collecting data..."
                                )
                                return True
                            else:
                                continue
                        else:
                            print("Empty label. Please try again.")
                            continue
                    elif confirm == "n" or confirm == "no":
                        print("Please enter the label again.")
                        continue
                    else:
                        print(
                            "Please enter 'y' for yes, 'n' for no, or 'edit' to modify the label."
                        )
                        continue
                else:
                    print("Invalid label. Please enter a valid label.")
                    continue
        elif self.manual_mode:
            label = input("Enter label for manual capture: ").strip()
            if label:
                self.current_label = label
                print(f"✓ Manual mode label: '{self.current_label}'")
                return True
        return True


def handle_drawing(hand_landmarks, drawing_canvas, drawing_mode, prev_point, frame_counter, draw_interval=4):
    """Handle finger drawing functionality with dynamic offset and smoothing"""
    if not drawing_mode:
        return None
    
    if frame_counter % draw_interval != 0:
        return prev_point
    
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    
    # Calculate offset relative to thumb tip to thumb MCP direction
    offset_x = (thumb_tip.x - thumb_mcp.x) * 0.15
    offset_y = (thumb_tip.y - thumb_mcp.y) * 0.15
    
    h, w = drawing_canvas.shape[:2]
    current_x = int((1.0 - (thumb_tip.x + offset_x)) * w)
    current_y = int((thumb_tip.y + offset_y) * h)
    
    # Apply smoothing to reduce shakiness
    if prev_point is not None:
        smooth_factor = 0.3  # Much more smoothing (70% previous, 30% current)
        finger_x = int(prev_point[0] * (1 - smooth_factor) + current_x * smooth_factor)
        finger_y = int(prev_point[1] * (1 - smooth_factor) + current_y * smooth_factor)
        cv2.line(drawing_canvas, prev_point, (finger_x, finger_y), (50, 255, 50), 4)
    else:
        finger_x, finger_y = current_x, current_y
    
    return (finger_x, finger_y)


def draw_prediction_on_hand(
    flipped_image, hand_landmarks, prediction, width, hand_label
):
    """Draw prediction text near the hand with smooth positioning"""
    if hand_landmarks and prediction:
        height, width = flipped_image.shape[:2]

        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]

        center_x = int((1.0 - (wrist.x + middle_mcp.x) / 2) * width)
        center_y = int((wrist.y + middle_mcp.y) / 2 * height)

        text_x = center_x - 50
        text_y = center_y - 40

        text_x = max(10, min(text_x, width - 200))
        text_y = max(30, min(text_y, height - 20))

        text_size = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        rect_x1 = text_x - 15
        rect_y1 = text_y - text_size[1] - 15
        rect_x2 = text_x + text_size[0] + 15
        rect_y2 = text_y + 10

        shadow_offset = 3
        cv2.rectangle(
            flipped_image,
            (rect_x1 + shadow_offset, rect_y1 + shadow_offset),
            (rect_x2 + shadow_offset, rect_y2 + shadow_offset),
            (0, 0, 0),
            -1,
        )

        overlay = flipped_image.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (40, 40, 40), -1)

        gradient = np.zeros_like(overlay[rect_y1:rect_y2, rect_x1:rect_x2])
        for i in range(gradient.shape[0]):
            alpha = i / gradient.shape[0]
            gradient[i, :] = (0, int(80 * (1 - alpha)), int(120 * (1 - alpha)))

        overlay[rect_y1:rect_y2, rect_x1:rect_x2] = cv2.addWeighted(
            overlay[rect_y1:rect_y2, rect_x1:rect_x2], 0.7, gradient, 0.3, 0
        )

        cv2.addWeighted(overlay, 0.8, flipped_image, 0.2, 0, flipped_image)

        cv2.rectangle(
            flipped_image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 200, 255), 2
        )

        cv2.putText(
            flipped_image,
            prediction,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        hand_text_y = text_y + 30
        cv2.putText(
            flipped_image,
            hand_label,
            (text_x, hand_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 100),
            2,
            cv2.LINE_AA,
        )

        cv2.line(
            flipped_image,
            (center_x, center_y),
            (text_x + text_size[0] // 2, rect_y2),
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


class VideoRecorder:
    def __init__(self):
        self.is_recording = False
        self.video_writer = None
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)

    def start_recording(self, frame_width, frame_height, fps=20.0):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"hand_gesture_{timestamp}.mp4")

        for codec in ["mp4v", "h264", "XVID"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, fps, (frame_width, frame_height)
            )
            if self.video_writer.isOpened():
                break
        else:
            print("Error: Could not open video writer with any codec.")
            return None

        self.is_recording = True
        print(f"Started recording: {filename}")
        return filename

    def stop_recording(self):
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print("Stopped recording")

    def write_frame(self, frame):
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def toggle_recording(self, frame_width, frame_height):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording(frame_width, frame_height)


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    dataset_creator = DatasetCreator()
    video_recorder = VideoRecorder()

    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Drawing variables
    drawing_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    prev_point = None

    frame_counter = 0
    prev_frame_time = 0
    new_frame_time = 0

    print("Hand Landmark Dataset Creator")
    print("=" * 40)
    print("Controls:")
    print("- 'D' key: Toggle Dataset Creation Mode")
    print("- SPACE: Start/Stop data collection (Auto) / Capture sample (Manual)")
    print("- 'M' key: Toggle Manual/Automatic capture mode")
    print("- 'L' key: Change label (Manual mode only)")
    print("- 'R' key: Start/Stop video recording")
    print("- Pen gesture: Automatic drawing mode")
    print("- 'C' key: Clear drawing")
    print("- ESC: Exit program")
    print("\nStarting camera...")

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2,  # Limit to one hand for better performance
    ) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if frame_counter % 1 == 0:
                results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            predictions = {}
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # Handle predictions and data collection
                    if not dataset_creator.dataset_mode:
                        try:
                            prediction = gp.predict(hand_landmarks)
                            if prediction is not None:
                                gesture, confidence = prediction
                                predictions[i] = str(gesture)
                                print(f"Gesture: {gesture}, Accuracy: {confidence:.2%}")
                            else:
                                predictions[i] = "No gesture detected"
                        except Exception as e:
                            print(f"Error in gesture prediction: {e}")
                            predictions[i] = "Error"

                    # Automatic data collection (original functionality)
                    if (
                        dataset_creator.collecting
                        and dataset_creator.current_label
                        and not dataset_creator.manual_mode
                        and frame_counter % 1 == 0
                    ):
                        frame_counter += 1
                        print(
                            f"Saving frame {frame_counter} with label: {dataset_creator.current_label}"
                        )
                        save_to_csv(
                            hand_landmarks,
                            dataset_creator.current_label,
                            "hand_landmarks_dataset",
                            "auto"
                        )
                        cv2.circle(image, (50, 150), 15, (0, 255, 0), -1)
                        cv2.putText(
                            image,
                            "SAVING",
                            (70, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )



            flipped_image = cv2.flip(image, 1)
            video_recorder.write_frame(flipped_image)

            # Draw predictions on flipped image
            if results.multi_hand_landmarks and not dataset_creator.dataset_mode:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    try:
                        hand_label = (
                            "Right"
                            if results.multi_handedness[i].classification[0].label
                            == "Left"
                            else "Left"
                        )  # Because the image is mirrored the label needs to be reverse

                        # Handle drawing based on gesture prediction
                        is_pen_gesture = predictions.get(i, "").lower() == "pen"
                        prev_point = handle_drawing(hand_landmarks, drawing_canvas, is_pen_gesture, prev_point, frame_counter)

                        draw_prediction_on_hand(
                            flipped_image,
                            hand_landmarks,
                            predictions.get(i, "No prediction"),
                            flipped_image.shape[1],
                            hand_label,
                        )
                    except Exception as e:
                        print(f"Error drawing prediction: {e}")

            # Overlay drawing canvas on flipped image
            flipped_image = cv2.add(flipped_image, drawing_canvas)
            
            dataset_creator.display_instructions(flipped_image)
            
            # Show drawing status based on pen gesture detection
            pen_detected = any(predictions.get(i, "").lower() == "pen" for i in predictions)
            if pen_detected:
                cv2.putText(flipped_image, "PEN DETECTED - DRAWING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

            cv2.putText(
                flipped_image,
                f"FPS: {int(fps)}",
                (10, flipped_image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            if video_recorder.is_recording:
                cv2.circle(
                    flipped_image,
                    (flipped_image.shape[1] - 20, 20),
                    10,
                    (0, 0, 255),
                    -1,
                )
                cv2.putText(
                    flipped_image,
                    "REC",
                    (flipped_image.shape[1] - 40, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            cv2.imshow("Hand Landmark Dataset Creator", flipped_image)

            key = cv2.waitKey(1) & 0xFF

            # Key handlers
            if key == 27:  # ESC
                print("\nExiting program...")
                if video_recorder.is_recording:
                    video_recorder.stop_recording()
                break

            elif key in [ord("d"), ord("D")]:  # Toggle dataset mode
                dataset_creator.dataset_mode = not dataset_creator.dataset_mode
                dataset_creator.collecting = False
                dataset_creator.current_label = ""
                dataset_creator.manual_mode = False
                frame_counter = 0

                mode_status = "ENTERED" if dataset_creator.dataset_mode else "EXITED"
                print(f"\n{'=' * 50}\n{mode_status} DATASET CREATION MODE\n{'=' * 50}")

            elif key in [ord("m"), ord("M")] and dataset_creator.dataset_mode:  # Toggle manual mode
                dataset_creator.manual_mode = not dataset_creator.manual_mode
                dataset_creator.collecting = False
                dataset_creator.current_label = ""
                
                if dataset_creator.manual_mode:
                    print("Switched to MANUAL capture mode")
                    dataset_creator.get_label_input()
                else:
                    print("Switched to AUTOMATIC capture mode")

            elif key == ord(" ") and dataset_creator.dataset_mode:  # Space key
                if dataset_creator.manual_mode:
                    # Manual capture
                    if not dataset_creator.current_label:
                        dataset_creator.get_label_input()
                    elif results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            dataset_creator.frame_counter += 1
                            save_to_csv(hand_landmarks, dataset_creator.current_label, "hand_landmarks_dataset", "manual")
                            print(f"Captured sample {dataset_creator.frame_counter} for '{dataset_creator.current_label}'")
                            break
                    else:
                        print("No hand detected - cannot capture sample")
                else:
                    # Automatic mode (original functionality)
                    if not dataset_creator.collecting:
                        if dataset_creator.get_label_input() and dataset_creator.current_label:
                            dataset_creator.collecting = True
                            frame_counter = 0
                            print(f"\nStarted collecting data for label: '{dataset_creator.current_label}'")
                            print("Press SPACE again to stop collecting...")
                    else:
                        dataset_creator.collecting = False
                        print(f"\nStopped collecting data for label: '{dataset_creator.current_label}'")
                        print(f"Total frames collected: {frame_counter}")
                        dataset_creator.current_label = ""
                        frame_counter = 0
                        
                        if input("Continue with another label? (y/n): ").strip().lower() != "y":
                            dataset_creator.dataset_mode = False
                            print("Exited dataset creation mode.")

            elif key in [ord("r"), ord("R")]:  # Toggle recording
                video_recorder.toggle_recording(frame_width, frame_height)
            

            
            elif key in [ord("l"), ord("L")] and dataset_creator.dataset_mode and dataset_creator.manual_mode:  # Change label in manual mode
                dataset_creator.get_label_input()
            
            elif key in [ord("c"), ord("C")]:  # Clear drawing
                drawing_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                print("Drawing cleared")

    cap.release()
    if video_recorder.is_recording:
        video_recorder.stop_recording()
    cv2.destroyAllWindows()
    print("Program ended.")


if __name__ == "__main__":
    main()
