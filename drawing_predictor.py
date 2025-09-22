import joblib
import numpy as np
import tensorflow as tf
import json
from typing import Tuple
from utils import extract_hand_landmark_points


class DrawingPredictor:
    def __init__(self):
        self.classifier = None
        self.autoencoder = None
        self.class_names = None
        self.threshold = None
        
    def load_models(self) -> bool:
        try:
            self.classifier = tf.lite.Interpreter("drawing_models/tflite/gesture_classifier.tflite")
            self.classifier.allocate_tensors()
            
            self.autoencoder = tf.lite.Interpreter("drawing_models/tflite/autoencoder.tflite")
            self.autoencoder.allocate_tensors()
            
            self.class_names = joblib.load("drawing_models/label_encoder.pkl").classes_
            with open("drawing_models/threshold.json") as f:
                self.threshold = json.load(f)["threshold"]
            return True
        except Exception:
            return False
    
    def _prepare_landmarks(self, landmarks):
        points = extract_hand_landmark_points(landmarks)
        if len(points) != 42:
            return None
        
        array = np.array(points, dtype=np.float32).reshape(21, 2)
        array -= array[0]  # Normalize to wrist
        return array.flatten().reshape(1, -1)
    
    def _get_reconstruction_error(self, data):
        if not self.autoencoder:
            return 0.0
        
        input_details = self.autoencoder.get_input_details()
        output_details = self.autoencoder.get_output_details()
        
        self.autoencoder.set_tensor(input_details[0]["index"], data)
        self.autoencoder.invoke()
        reconstruction = self.autoencoder.get_tensor(output_details[0]["index"])
        
        return float(np.mean(np.square(data - reconstruction)))
    
    def _classify(self, data):
        if not self.classifier:
            return "Error", 0.0
        
        input_details = self.classifier.get_input_details()
        output_details = self.classifier.get_output_details()
        
        self.classifier.set_tensor(input_details[0]["index"], data)
        self.classifier.invoke()
        predictions = self.classifier.get_tensor(output_details[0]["index"])
        
        idx = np.argmax(predictions[0])
        confidence = float(predictions[0][idx])
        gesture = self.class_names[idx]
        
        return gesture, confidence
    
    def predict(self, landmarks) -> Tuple[str, float]:
        data = self._prepare_landmarks(landmarks)
        if data is None:
            return "Invalid", 0.0
        
        error = self._get_reconstruction_error(data)
        if error > self.threshold:
            return "?", 0.0
        
        return self._classify(data)


_drawing_predictor = DrawingPredictor()


def initialize_drawing_models() -> bool:
    return _drawing_predictor.load_models()


def predict_drawing_gesture(landmarks) -> Tuple[str, float]:
    return _drawing_predictor.predict(landmarks)