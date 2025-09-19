Model: gesture_classifier
TensorFlow version: 2.19.1
Files included:
- gesture_classifier_complete.keras: Complete model (RECOMMENDED for loading)
- gesture_classifier_complete.h5: Complete model (legacy H5 format)
- gesture_classifier_savedmodel/: TensorFlow SavedModel format
- gesture_classifier.weights.h5: Model weights only
- gesture_classifier_architecture.json: Model architecture
- gesture_classifier_history.json: Training history
- gesture_classifier_info.json: Model metadata

Loading instructions:
- Use .keras file: model = tf.keras.models.load_model('path/to/model.keras')
- Use .h5 file: model = tf.keras.models.load_model('path/to/model.h5')
- Use SavedModel: model = tf.saved_model.load('path/to/savedmodel')
