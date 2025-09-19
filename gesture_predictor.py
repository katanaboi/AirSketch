import joblib
import numpy as np
import tensorflow as tf

from utils import extract_hand_landmark_points


class FastGesturePredictor:
    def __init__(self, tflite_model_path):
        import os
        os.makedirs("models", exist_ok=True)
        
        try:
            self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            self.interpreter.allocate_tensors()
        except Exception as e:
            raise RuntimeError(f"Failed to load TFLite model: {str(e)}")

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        try:
            label_encoder = joblib.load("models/label_encoder.pkl")
            self.class_names = label_encoder.classes_
        except FileNotFoundError:
            raise FileNotFoundError(
                "label_encoder.pkl not found. Please ensure the file exists in the correct directory."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load label encoder: {str(e)}")

    def predict(self, landmarks):
        # Prepare the landmarks (same as before)
        landmark_points = extract_hand_landmark_points(landmarks)

        if len(landmark_points) != 42:
            return "Error", 0.0

        # Convert and normalize
        landmark_array = np.array(landmark_points, dtype=np.float32)
        landmarks_reshaped = landmark_array.reshape(21, 2)

        # Normalize to wrist
        wrist_x = landmarks_reshaped[0, 0]
        wrist_y = landmarks_reshaped[0, 1]
        landmarks_reshaped[:, 0] -= wrist_x
        landmarks_reshaped[:, 1] -= wrist_y

        # Prepare for model
        input_data = landmarks_reshaped.flatten().reshape(1, -1).astype(np.float32)

        # Make prediction with fast model
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Get result
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]

        return predicted_class, float(confidence)


try:
    FAST_PREDICTOR = FastGesturePredictor("models/tflite/gesture_classifier.tflite")
except Exception as e:
    print(f"Warning: Could not initialize gesture predictor: {e}")
    FAST_PREDICTOR = None

def predict_gesture(landmarks):
    if FAST_PREDICTOR is None:
        return "Model not loaded", 0.0
    return FAST_PREDICTOR.predict(landmarks)
