import os
import time  # Added for FPS calculation
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# from classifier import predict_gesture
# from classifier2 import GesturePredictor
from gesture_predictor import FastGesturePredictor
from utils import extract_hand_landmark_points, save_to_csv

gp = FastGesturePredictor(tflite_model_path="models/tflite/gesture_classifier.tflite")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class DatasetCreator:
    def __init__(self):
        self.dataset_mode = False
        self.current_label = ""
        self.collecting = False
        self.frame_counter = 0

    def display_instructions(self, image):
        """Display current mode and instructions on the image"""
        height, width = image.shape[:2]

        # Background rectangle for text
        cv2.rectangle(image, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (width - 10, 120), (255, 255, 255), 2)

        # Mode status
        mode_text = "DATASET MODE: ON" if self.dataset_mode else "DATASET MODE: OFF"
        mode_color = (0, 255, 0) if self.dataset_mode else (0, 0, 255)
        cv2.putText(
            image, mode_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2
        )

        if self.dataset_mode:
            # Collection status
            if self.collecting:
                status_text = f"COLLECTING: {self.current_label}"
                status_color = (0, 255, 255)  # Yellow
            else:
                status_text = "READY - Enter label and press SPACE to start"
                status_color = (255, 255, 255)  # White

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
                "SPACE: Start/Stop | 'D': Toggle Dataset Mode | ESC: Exit",
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
        if not self.collecting:
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
                    # Confirm the label with the user
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
                                continue  # Start over
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
        return True


def draw_prediction_on_hand(
    flipped_image, hand_landmarks, prediction, width, hand_label
):
    """Draw prediction text near the hand with smooth positioning"""
    if hand_landmarks and prediction:
        # Get image dimensions
        height, width = flipped_image.shape[:2]

        # Calculate hand center (using wrist and middle finger MCP as reference)
        wrist = hand_landmarks.landmark[0]  # Wrist landmark
        middle_mcp = hand_landmarks.landmark[9]  # Middle finger MCP

        # Convert normalized coordinates to pixel coordinates
        # Account for the fact that the image will be flipped for display
        center_x = int(
            (1.0 - (wrist.x + middle_mcp.x) / 2) * width
        )  # Flip X coordinate
        center_y = int((wrist.y + middle_mcp.y) / 2 * height)

        # Position text above the hand center
        text_x = center_x - 50
        text_y = center_y - 40

        # Ensure text stays within image bounds
        text_x = max(10, min(text_x, width - 200))
        text_y = max(30, min(text_y, height - 20))

        # Create background rectangle for better readability
        text_size = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        rect_x1 = text_x - 15
        rect_y1 = text_y - text_size[1] - 15
        rect_x2 = text_x + text_size[0] + 15
        rect_y2 = text_y + 10

        # Draw rounded rectangle background with shadow effect
        # Shadow
        shadow_offset = 3
        cv2.rectangle(
            flipped_image,
            (rect_x1 + shadow_offset, rect_y1 + shadow_offset),
            (rect_x2 + shadow_offset, rect_y2 + shadow_offset),
            (0, 0, 0),
            -1,
        )

        # Main background with gradient effect
        overlay = flipped_image.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (40, 40, 40), -1)

        # Add gradient effect (darker at bottom)
        gradient = np.zeros_like(overlay[rect_y1:rect_y2, rect_x1:rect_x2])
        for i in range(gradient.shape[0]):
            alpha = i / gradient.shape[0]
            gradient[i, :] = (0, int(80 * (1 - alpha)), int(120 * (1 - alpha)))

        overlay[rect_y1:rect_y2, rect_x1:rect_x2] = cv2.addWeighted(
            overlay[rect_y1:rect_y2, rect_x1:rect_x2], 0.7, gradient, 0.3, 0
        )

        cv2.addWeighted(overlay, 0.8, flipped_image, 0.2, 0, flipped_image)

        # Draw border with rounded corners
        cv2.rectangle(
            flipped_image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 200, 255), 2
        )

        # Draw prediction text with better styling
        cv2.putText(
            flipped_image,
            prediction,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),  # White color
            2,
            cv2.LINE_AA,
        )

        # Draw hand label (Left/Right) below the prediction
        hand_text_y = text_y + 30
        cv2.putText(
            flipped_image,
            hand_label,
            (text_x, hand_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 100),  # Light yellow color
            2,
            cv2.LINE_AA,
        )

        # Draw a subtle connecting line to hand
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

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start_recording(self, frame_width, frame_height, fps=20.0):
        """Start recording video in MP4 format"""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"hand_gesture_{timestamp}.mp4")

        # Define the codec for MP4 and create VideoWriter object
        # Try different codecs for MP4 compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
        self.video_writer = cv2.VideoWriter(
            filename, fourcc, fps, (frame_width, frame_height)
        )

        if not self.video_writer.isOpened():
            print("Error: Could not open video writer with MP4V codec. Trying H264...")
            fourcc = cv2.VideoWriter_fourcc(*"h264")  # Alternative codec
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, fps, (frame_width, frame_height)
            )

            if not self.video_writer.isOpened():
                print(
                    "Error: Could not open video writer with H264 codec. Trying XVID..."
                )
                fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Fallback codec
                self.video_writer = cv2.VideoWriter(
                    filename, fourcc, fps, (frame_width, frame_height)
                )

                if not self.video_writer.isOpened():
                    print("Error: Could not open video writer with any codec.")
                    return None

        self.is_recording = True

        print(f"Started recording: {filename}")
        return filename

    def stop_recording(self):
        """Stop recording video"""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print("Stopped recording")

    def write_frame(self, frame):
        """Write a frame to the video file if recording"""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def toggle_recording(self, frame_width, frame_height):
        """Toggle recording state"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording(frame_width, frame_height)


def main():
    # Suppress TensorFlow and absl warnings
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    # Suppress absl warnings
    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    dataset_creator = DatasetCreator()
    video_recorder = VideoRecorder()

    # For webcam input:
    cap = cv2.VideoCapture(0)
    # Let the camera keep its native resolution for better accuracy
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_counter = 0
    prev_frame_time = 0
    new_frame_time = 0

    print("Hand Landmark Dataset Creator")
    print("=" * 40)
    print("Controls:")
    print("- 'D' key: Toggle Dataset Creation Mode")
    print("- SPACE: Start/Stop data collection")
    print("- 'R' key: Start/Stop video recording")
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

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time

            # Process the image
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Only process every other frame to improve performance
            if frame_counter % 1 == 0:
                results = hands.process(image)
            else:
                # Reuse previous results for skipped frames
                pass

            # Draw the hand annotations on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Process hand landmarks
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get hand label (Left or Right)
                    hand_label = results.multi_handedness[i].classification[0].label

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # Only predict gestures when NOT in dataset mode
                    if not dataset_creator.dataset_mode:
                        try:
                            prediction = gp.predict(hand_landmarks)
                            # Convert prediction to string if it's not None
                            if prediction is not None:
                                prediction = str(prediction)
                            else:
                                prediction = "No gesture detected"
                        except Exception as e:
                            print(f"Error in gesture prediction: {e}")
                            prediction = "Error"

                    # Save data if in collection mode
                    if dataset_creator.collecting and dataset_creator.current_label:
                        frame_counter += 1

                        # Save every 5th frame to avoid too much data
                        if frame_counter % 1 == 0:
                            print(
                                f"Saving frame {frame_counter} with label: {dataset_creator.current_label}"
                            )

                            # Extract and save landmark points
                            landmark_points = extract_hand_landmark_points(
                                hand_landmarks
                            )
                            save_to_csv(
                                hand_landmarks,
                                dataset_creator.current_label,
                                "hand_landmarks_dataset",
                            )

                            # Visual feedback - draw a green circle when saving
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

            # Flip the image horizontally for a selfie-view display
            flipped_image = cv2.flip(image, 1)

            # Write frame to video if recording
            video_recorder.write_frame(flipped_image)

            # Draw prediction on the flipped image (after flipping)
            if results.multi_hand_landmarks and not dataset_creator.dataset_mode:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    try:
                        # Get hand label (Left or Right)
                        hand_label = results.multi_handedness[i].classification[0].label

                        prediction = gp.predict(hand_landmarks)
                        if prediction is not None:
                            prediction = str(prediction[0])
                        else:
                            prediction = "No gesture detected"

                        # Draw on the flipped image with correct coordinates
                        draw_prediction_on_hand(
                            flipped_image,
                            hand_landmarks,
                            prediction,
                            flipped_image.shape[1],
                            hand_label,
                        )
                    except Exception as e:
                        print(f"Error drawing prediction: {e}")

            # Display instructions and status on the flipped image
            dataset_creator.display_instructions(flipped_image)

            # Display FPS on screen
            cv2.putText(
                flipped_image,
                f"FPS: {int(fps)}",
                (10, flipped_image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # Display recording status
            if video_recorder.is_recording:
                # Draw red circle to indicate recording
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

            # Show the image
            cv2.imshow("Hand Landmark Dataset Creator", flipped_image)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF  # Reduced wait time for better responsiveness

            if key == 27:  # ESC key
                print("\nExiting program...")
                # Stop recording if active
                if video_recorder.is_recording:
                    video_recorder.stop_recording()
                break

            elif key == ord("d") or key == ord("D"):  # Toggle dataset mode
                dataset_creator.dataset_mode = not dataset_creator.dataset_mode
                dataset_creator.collecting = False
                dataset_creator.current_label = ""
                frame_counter = 0

                if dataset_creator.dataset_mode:
                    print("\n" + "=" * 50)
                    print("ENTERED DATASET CREATION MODE")
                    print("=" * 50)
                else:
                    print("\n" + "=" * 50)
                    print("EXITED DATASET CREATION MODE")
                    print("=" * 50)

            elif key == ord(" "):  # SPACE key
                if dataset_creator.dataset_mode:
                    if not dataset_creator.collecting:
                        # Try to get label and start collecting
                        if dataset_creator.get_label_input():
                            if dataset_creator.current_label:
                                dataset_creator.collecting = True
                                frame_counter = 0
                                print(
                                    f"\nStarted collecting data for label: '{dataset_creator.current_label}'"
                                )
                                print("Press SPACE again to stop collecting...")
                    else:
                        # Stop collecting
                        dataset_creator.collecting = False
                        print(
                            f"\nStopped collecting data for label: '{dataset_creator.current_label}'"
                        )
                        print(f"Total frames collected: {frame_counter}")
                        dataset_creator.current_label = ""
                        frame_counter = 0

                        # Ask if user wants to continue with another label
                        continue_choice = (
                            input("Continue with another label? (y/n): ")
                            .strip()
                            .lower()
                        )
                        if continue_choice != "y":
                            dataset_creator.dataset_mode = False
                            print("Exited dataset creation mode.")

            elif key == ord("r") or key == ord("R"):  # Toggle recording
                video_recorder.toggle_recording(frame_width, frame_height)

    cap.release()
    # Make sure to release video writer if still recording
    if video_recorder.is_recording:
        video_recorder.stop_recording()
    cv2.destroyAllWindows()
    print("Program ended.")


if __name__ == "__main__":
    main()
