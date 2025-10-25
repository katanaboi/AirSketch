import os
import time
import cv2
import mediapipe as mp
import traceback
import socket
import struct
import numpy as np

from gesture_predictor import initialize_models, predict_gesture, predict_gesture_no_threshold
from drawing_predictor import initialize_drawing_models, predict_drawing_gesture
from drawing_handler import DrawingHandler
from dataset_creator import DatasetCreator
from video_recorder import VideoRecorder
from ui_handler import UIHandler

os.makedirs("models/tflite", exist_ok=True)
os.makedirs("drawing_models/tflite", exist_ok=True)

HOST_IP = '0.0.0.0'
HOST_PORT = 9999

# Initialize models
detection_models_loaded = initialize_models()
drawing_models_loaded = initialize_drawing_models()
print(f"Detection models: {'✓' if detection_models_loaded else '✗'}")
print(f"Drawing models: {'✓' if drawing_models_loaded else '✗'}")
print("Note: Detection mode uses threshold by default. Edit main_app.py line ~104 to toggle.")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def recv_all(sock, count):
    """Helper function to reliably receive a specific number of bytes."""
    buf = b''
    while count > 0:
        try:
            new_buf = sock.recv(count)
            if not new_buf: return None  # Connection closed
        except Exception as e:
            print(f"Error during recv: {e}")
            return None  # Treat other errors as disconnect

        buf += new_buf
        count -= len(new_buf)
    return buf


# --- NEW: Helper function from your server ---
def send_message(sock, message: str):
    """Encodes and sends a string message with a 4-byte length prefix."""
    try:
        message_bytes = message.encode('utf-8')
        length_prefix = struct.pack('<I', len(message_bytes))
        sock.sendall(length_prefix)
        sock.sendall(message_bytes)
    except (BrokenPipeError, ConnectionResetError) as e:
        print(f"Client disconnected during send: {e}")
        raise  # Re-raise to break the loop
    except Exception as e:
        print(f"Error sending message: {e}")
        raise  # Re-raise other errors


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
    ui_handler = UIHandler()

    # --- MODIFIED: We no longer use a local webcam ---
    # cap = cv2.VideoCapture(0)
    # We will get frame_width and frame_height from the first received frame
    frame_width, frame_height = None, None
    drawing_handler = None

    # App modes
    landmark_mode = False  # Default mode - just landmarks
    detection_mode = True  # Custom detection mode
    drawing_mode = False  # Drawing mode

    frame_counter = 0
    prev_frame_time = 0

    print("AirSketch Application")
    print("=" * 40)
    print("Controls:")
    print("- Default: Landmark Mode (landmarks + hand labels)")
    print("- 'Q': Detection Mode (custom gestures) | 'D': Dataset Mode")
    print("- 'W': Drawing Mode (pen/eraser) | 'C': Clear drawing")
    print("- 'M': Manual mode | 'L': Change label | 'R': Record | ESC: Exit")
    print("\nStarting camera...")

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5,
            max_num_hands=2
    ) as hands:

        # --- NEW: Outer server loop ---
        conn = None
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((HOST_IP, HOST_PORT))
                    sock.listen(1)

                    print(f"[*] AirSketch Server listening on {HOST_IP}:{HOST_PORT}")
                    conn, addr = sock.accept()

                    with conn:
                        print(f"[*] Connected by {addr}")
                        send_message(conn, "MSG:AirSketch Server connected")

                        # --- MODIFIED: This is now the inner client loop ---
                        while True:

                            # --- NEW: Receive Image Data ---
                            size_data = recv_all(conn, 4)
                            if not size_data: break
                            img_size = struct.unpack('<I', size_data)[0]

                            if img_size > 10 * 1024 * 1024:
                                print(f"Warning: Large image size ({img_size}). Disconnecting.")
                                break

                            img_data = recv_all(conn, img_size)
                            if not img_data: break

                            # --- NEW: Process Image ---
                            np_array = np.frombuffer(img_data, dtype=np.uint8)
                            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                            if image is None:
                                print("Failed to decode image.")
                                continue

                            # --- NEW: Initialize handler on first frame ---
                            if drawing_handler is None:
                                frame_height, frame_width, _ = image.shape
                                drawing_handler = DrawingHandler(frame_height, frame_width)
                                print(f"Stream started: {frame_width}x{frame_height}")

                            # --- This is the original AirSketch logic ---

                            # Calculate FPS
                            new_frame_time = time.time()
                            fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
                            prev_frame_time = new_frame_time

                            # Process image
                            image.flags.writeable = False
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            results = hands.process(image_rgb)
                            image.flags.writeable = True
                            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # No need, already BGR
                            frame_counter += 1

                            # Handle predictions and drawing
                            predictions = {}
                            if results.multi_hand_landmarks:
                                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                                    # Draw landmarks (not in drawing mode)
                                    if not drawing_mode:
                                        mp_drawing.draw_landmarks(
                                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing_styles.get_default_hand_landmarks_style(),
                                            mp_drawing_styles.get_default_hand_connections_style()
                                        )

                                    # Gesture predictions
                                    if drawing_mode and drawing_models_loaded:
                                        try:
                                            prediction = predict_drawing_gesture(hand_landmarks)
                                            if prediction:
                                                gesture, confidence = prediction
                                                predictions[i] = str(gesture)
                                                # --- NEW: Send gesture back to client ---
                                                send_message(conn, f"GES:{str(gesture)}")
                                            else:
                                                predictions[i] = "?"
                                        except Exception as e:
                                            predictions[i] = "?"
                                    elif detection_mode and detection_models_loaded:
                                        try:
                                            prediction = predict_gesture(hand_landmarks)  # WITH threshold
                                            #prediction = predict_gesture_no_threshold(hand_landmarks)  # WITHOUT threshold

                                            if prediction:
                                                gesture, confidence = prediction
                                                predictions[i] = str(gesture)
                                                # --- NEW: Send gesture back to client ---
                                                send_message(conn, f"GES:{str(gesture)}")
                                                print(f"Ges:{str(gesture)}")
                                            else:
                                                predictions[i] = "?"
                                        except Exception as e:
                                            predictions[i] = "?"
                                            print("--- PREDICTION FAILED ---")
                                            traceback.print_exc()
                                            print("-------------------------")

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
                                        hand_label = ("Right" if results.multi_handedness[i].classification[
                                                                     0].label == "Left" else "Left")

                                        if drawing_mode and not dataset_creator.dataset_mode:
                                            is_pen = drawing_models_loaded and predictions.get(i, "").lower() == "pen"
                                            is_eraser = drawing_models_loaded and predictions.get(i,
                                                                                                  "").lower() == "eraser"

                                            if is_pen:
                                                drawing_handler.handle_drawing(hand_landmarks, frame_counter)
                                                drawing_handler.draw_thumb_tip_indicator(flipped_image, hand_landmarks)
                                            elif is_eraser:
                                                drawing_handler.handle_erasing(hand_landmarks, frame_counter)
                                                drawing_handler.draw_eraser_indicator(flipped_image, hand_landmarks)
                                            else:
                                                drawing_handler.reset_drawing_state()

                                            prediction = predictions.get(i, "?")
                                            if prediction != "?":
                                                ui_handler.draw_prediction_on_hand(
                                                    flipped_image, hand_landmarks,
                                                    prediction,
                                                    flipped_image.shape[1], hand_label
                                                )
                                        elif detection_mode and not dataset_creator.dataset_mode:
                                            prediction = predictions.get(i, "?")
                                            ui_handler.draw_prediction_on_hand(
                                                flipped_image, hand_landmarks,
                                                prediction,
                                                flipped_image.shape[1], hand_label
                                            )
                                        elif landmark_mode and not dataset_creator.dataset_mode:
                                            ui_handler.draw_prediction_on_hand(
                                                flipped_image, hand_landmarks,
                                                "",
                                                flipped_image.shape[1], hand_label
                                            )
                                        elif dataset_creator.dataset_mode:
                                            ui_handler.draw_prediction_on_hand(
                                                flipped_image, hand_landmarks,
                                                predictions.get(i, "No prediction"),
                                                flipped_image.shape[1], hand_label
                                            )
                                    except Exception as e:
                                        print(f"Error handling mode: {e}")

                            # Overlay drawing canvas
                            # --- NEW: Check if handler is initialized ---
                            if drawing_handler:
                                flipped_image = cv2.add(flipped_image, drawing_handler.canvas)

                            # Display UI elements
                            if dataset_creator.dataset_mode:
                                dataset_creator.display_instructions(flipped_image)
                            else:
                                ui_handler.draw_legend(flipped_image, landmark_mode, detection_mode, drawing_mode,
                                                       dataset_creator.dataset_mode)

                            ui_handler.draw_status_indicators(flipped_image, detection_models_loaded,
                                                              drawing_models_loaded, predictions,
                                                              video_recorder.is_recording, fps)

                            cv2.imshow("AirSketch Server (Receiving from Quest 3)", flipped_image)

                            # Handle key presses
                            key = cv2.waitKey(1) & 0xFF

                            if key == 27:  # ESC
                                print("\nExiting...")
                                if video_recorder.is_recording:
                                    video_recorder.stop_recording()
                                raise KeyboardInterrupt  # --- MODIFIED: to break all loops

                            # --- All other key handlers remain the same ---
                            elif key in [ord("d"), ord("D")]:
                                dataset_creator.toggle_dataset_mode()
                            elif key in [ord("m"), ord("M")]:
                                dataset_creator.toggle_manual_mode()
                            elif key == ord(" "):
                                dataset_creator.handle_space_key(results)
                            elif key in [ord("r"), ord("R")]:
                                if frame_width and frame_height:  # Check if dims are set
                                    video_recorder.toggle_recording(frame_width, frame_height)
                                else:
                                    print("Waiting for stream to start...")
                            elif key in [ord("l"),
                                         ord("L")] and dataset_creator.dataset_mode and dataset_creator.manual_mode:
                                dataset_creator.get_label_input()
                            elif key in [ord("c"), ord("C")]:
                                if drawing_handler:  # Check if handler is set
                                    drawing_handler.clear_canvas()
                                    print("Drawing cleared")
                            elif key in [ord("w"), ord("W")]:
                                if not dataset_creator.dataset_mode:
                                    drawing_mode = True;
                                    detection_mode = False;
                                    landmark_mode = False
                                    print("Switched to DRAWING mode")
                            elif key in [ord("q"), ord("Q")]:
                                if not dataset_creator.dataset_mode:
                                    detection_mode = True;
                                    drawing_mode = False;
                                    landmark_mode = False
                                    print("Switched to DETECTION mode")
                            elif key in [ord("e"), ord("E")]:
                                if not dataset_creator.dataset_mode:
                                    landmark_mode = True;
                                    detection_mode = False;
                                    drawing_mode = False
                                    print("Switched to LANDMARK mode")

            # --- NEW: Exception handling from your server ---
            except ConnectionResetError:
                print("Client connection was forcibly closed.")
            except BrokenPipeError:
                print("Client connection broken.")
            except socket.timeout:
                print("Socket operation timed out.")
            except KeyboardInterrupt:
                print("Server stopped by user (Ctrl+C or 'q').")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print("--- Traceback ---")
                traceback.print_exc()
                print("-----------------")
            finally:
                if conn: conn.close(); print("Connection closed.")
                # Reset handler for new connection
                drawing_handler = None
                frame_width, frame_height = None, None
                print("Waiting for new connection...")
                time.sleep(1)

    # --- MODIFIED: Cleanup ---
    # cap.release() # No longer exists
    if video_recorder.is_recording:
        video_recorder.stop_recording()
    cv2.destroyAllWindows()
    print("Program ended.")


if __name__ == "__main__":
    main()