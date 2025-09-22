import os
import time
import cv2
import mediapipe as mp

from gesture_predictor import initialize_models, predict_gesture
from drawing_handler import DrawingHandler
from dataset_creator import DatasetCreator
from video_recorder import VideoRecorder
from ui_handler import UIHandler

os.makedirs("models/tflite", exist_ok=True)

# Initialize models
models_loaded = initialize_models()
print("Models loaded successfully" if models_loaded else "Warning: Could not load models - dataset mode only")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    # Initialize components
    dataset_creator = DatasetCreator()
    video_recorder = VideoRecorder()
    
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    drawing_handler = DrawingHandler(frame_height, frame_width)
    ui_handler = UIHandler()
    
    # App modes
    detection_mode = True  # Default mode
    drawing_mode = False
    
    frame_counter = 0
    prev_frame_time = 0

    print("AirSketch Application")
    print("=" * 40)
    print("Controls:")
    print("- Default: Detection Mode (landmarks + labels visible)")
    print("- 'D': Toggle Dataset Mode | SPACE: Start/Stop collection")
    print("- 'W': Toggle Drawing Mode (drawing with pen/eraser)")
    print("- 'M': Toggle Manual mode | 'L': Change label | 'R': Record")
    print("- 'C': Clear drawing | ESC: Exit")
    print("\nStarting camera...")

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2
    ) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time

            # Process image
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frame_counter += 1

            # Handle predictions and drawing
            predictions = {}
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks only in detection mode or dataset mode
                    if detection_mode or dataset_creator.dataset_mode:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                    # Gesture prediction (only in drawing mode)
                    if drawing_mode and models_loaded:
                        try:
                            prediction = predict_gesture(hand_landmarks)
                            if prediction:
                                gesture, confidence = prediction
                                predictions[i] = str(gesture)
                            else:
                                predictions[i] = "?"
                        except Exception as e:
                            print(f"Error in prediction: {e}")
                            predictions[i] = "?"

                    # Data collection
                    if dataset_creator.collect_frame_data(hand_landmarks, frame_counter):
                        ui_handler.draw_data_collection_indicator(image, True)

            # Flip image and handle drawing
            flipped_image = cv2.flip(image, 1)
            video_recorder.write_frame(flipped_image)

            # Handle different modes
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    try:
                        hand_label = ("Right" if results.multi_handedness[i].classification[0].label == "Left" else "Left")
                        
                        if drawing_mode and not dataset_creator.dataset_mode:
                            # Drawing mode: handle drawing/erasing and show only pen/eraser indicators
                            is_pen = models_loaded and predictions.get(i, "").lower() == "pen"
                            is_eraser = models_loaded and predictions.get(i, "").lower() == "eraser"
                            
                            if is_pen:
                                drawing_handler.handle_drawing(hand_landmarks, frame_counter)
                                drawing_handler.draw_thumb_tip_indicator(flipped_image, hand_landmarks)
                            elif is_eraser:
                                drawing_handler.handle_erasing(hand_landmarks, frame_counter)
                                drawing_handler.draw_eraser_indicator(flipped_image, hand_landmarks)
                            else:
                                drawing_handler.reset_drawing_state()

                            # Show prediction label only if it's not '?'
                            prediction = predictions.get(i, "?")
                            if prediction != "?":
                                ui_handler.draw_prediction_on_hand(
                                    flipped_image, hand_landmarks, 
                                    prediction, 
                                    flipped_image.shape[1], hand_label
                                )
                        elif detection_mode and not dataset_creator.dataset_mode:
                            # Detection mode: show landmarks and hand labels only
                            ui_handler.draw_prediction_on_hand(
                                flipped_image, hand_landmarks, 
                                "", 
                                flipped_image.shape[1], hand_label
                            )
                        elif dataset_creator.dataset_mode:
                            # Dataset mode: show landmarks and labels
                            ui_handler.draw_prediction_on_hand(
                                flipped_image, hand_landmarks, 
                                predictions.get(i, "No prediction"), 
                                flipped_image.shape[1], hand_label
                            )
                    except Exception as e:
                        print(f"Error handling mode: {e}")

            # Overlay drawing canvas
            flipped_image = cv2.add(flipped_image, drawing_handler.canvas)
            
            # Display UI elements
            if dataset_creator.dataset_mode:
                dataset_creator.display_instructions(flipped_image)
            else:
                # Show Apple-style legend
                ui_handler.draw_legend(flipped_image, detection_mode, drawing_mode, dataset_creator.dataset_mode)
            
            ui_handler.draw_status_indicators(flipped_image, models_loaded, predictions, 
                                            video_recorder.is_recording, fps)

            cv2.imshow("Hand Landmark Dataset Creator", flipped_image)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\nExiting...")
                if video_recorder.is_recording:
                    video_recorder.stop_recording()
                break
            elif key in [ord("d"), ord("D")]:
                dataset_creator.toggle_dataset_mode()
            elif key in [ord("m"), ord("M")]:
                dataset_creator.toggle_manual_mode()
            elif key == ord(" "):
                dataset_creator.handle_space_key(results)
            elif key in [ord("r"), ord("R")]:
                video_recorder.toggle_recording(frame_width, frame_height)
            elif key in [ord("l"), ord("L")] and dataset_creator.dataset_mode and dataset_creator.manual_mode:
                dataset_creator.get_label_input()
            elif key in [ord("c"), ord("C")]:
                drawing_handler.clear_canvas()
                print("Drawing cleared")
            elif key in [ord("w"), ord("W")]:
                if not dataset_creator.dataset_mode:
                    drawing_mode = not drawing_mode
                    detection_mode = not drawing_mode
                    mode_name = "DRAWING" if drawing_mode else "DETECTION"
                    print(f"Switched to {mode_name} mode")

    cap.release()
    if video_recorder.is_recording:
        video_recorder.stop_recording()
    cv2.destroyAllWindows()
    print("Program ended.")

if __name__ == "__main__":
    main()
