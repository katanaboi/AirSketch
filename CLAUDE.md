# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is AirSketch

AirSketch is a hand gesture recognition and sign language application for Meta Quest 3 mixed reality. It uses MediaPipe hand tracking and TensorFlow Lite models for real-time gesture classification. There are two entry points:

- **Local mode** (`main_app.py`): Uses the local webcam directly. Good for development and dataset collection. Read-only for prediction — it loads models but does not write them.
- **Server mode** (`AirSketch_server.py`): Receives camera frames over TCP from a Unity client running on Meta Quest 3. Sends gesture predictions back to the client via length-prefixed messages (`GES:<gesture>`). Defaults to Detection mode on startup.

## Running the App

```bash
pip install -r requirements.txt

# Local webcam mode
python main_app.py

# Server mode (receives frames from Quest 3 / Unity)
python AirSketch_server.py
```

The server listens on `0.0.0.0:9999`. The Unity client connects, streams JPEG-encoded frames (4-byte little-endian size prefix + image data), and receives back gesture strings in the same length-prefixed format.

Both modes open an OpenCV window with keyboard-driven mode switching (E=Landmark, Q=Detection, W=Drawing, D=Dataset, C=Clear canvas, R=Record, ESC=Exit).

## Architecture

There are two parallel ML pipelines that share the same structure but use separate model files:

1. **Detection pipeline** (`gesture_predictor.py` + `models/`): General gesture recognition (e.g. thumbs_up, peace, fist). Used in Detection mode (Q).
2. **Drawing pipeline** (`drawing_predictor.py` + `drawing_models/`): Classifies hand poses as "pen" or "eraser". Used in Drawing mode (W).

Both pipelines follow the same pattern:
- A **TFLite classifier** predicts the gesture class from 42 wrist-normalized landmark features (21 landmarks x 2 coords).
- A **TFLite autoencoder** provides anomaly detection — if reconstruction error exceeds a threshold (`threshold.json`), the prediction is rejected as "?".
- A **label encoder** (`label_encoder.pkl`) maps class indices to names.

Key difference: `gesture_predictor.py` also applies scale normalization (divides by max distance from wrist), while `drawing_predictor.py` only does wrist-relative normalization. This must stay consistent with how each model was trained.

### Main loop (`main_app.py` / `AirSketch_server.py`)

Both entry points share the same core loop: receive a frame, run MediaPipe hand detection, dispatch to the active mode's handler. The image is **flipped horizontally** before display (and drawing overlay), so all coordinate transforms in `DrawingHandler` and `UIHandler` use `1.0 - x` to map from MediaPipe's coordinate space to the flipped display.

`AirSketch_server.py` wraps this in a TCP server that accepts one client at a time. It uses a length-prefixed binary protocol (`<4-byte LE uint32 size><payload>`) for both receiving frames and sending gesture results. The server auto-reconnects when a client disconnects. It defaults to `detection_mode = True` (vs `landmark_mode = True` in local mode).

### Supporting modules

- `drawing_handler.py` — Canvas overlay, pen drawing (tracks thumb tip with smoothing), eraser (uses average of landmarks 7,8,11,12).
- `dataset_creator.py` — Interactive data collection (auto/manual modes), saves landmarks to CSV via `utils.save_to_csv`. CSV is always opened in append mode, so new sessions add rows rather than overwriting — the HUD shows `DB: <label> -> <total> | session: +<delta>` so you can see existing vs. newly captured samples at a glance.
- `ui_handler.py` — HUD rendering (predictions, FPS, recording indicator, legend).
- `video_recorder.py` — MP4 recording of the display output.
- `utils.py` — Landmark extraction functions and CSV I/O. The `extract_hand_landmark_points` function (42 features, 2D) is the shared input format for both pipelines.

## Training

Two equivalent paths — they share hyperparameters so accuracy is comparable:

**Notebooks** (`notebooks/`, step-by-step / exploratory):
1. `1. data-exploration.ipynb` — Analyze collected gesture datasets
2. `2. neural_network.ipynb` — Train the gesture classifier
3. `3. auto_encoder.ipynb` — Train the anomaly detection autoencoder and compute threshold

**`train.py`** (reproducible, end-to-end): runs the full pipeline — normalization, classifier training, autoencoder training, threshold computation, TFLite conversion, and evaluation plots — into a timestamped `models/runs/<timestamp>/` directory.

```bash
python train.py --dataset data/hand_landmarks_dataset.csv
python train.py --dataset data/hand_landmarks_dataset.csv --output-dir models/runs/custom
```

Full workflow: collect data with Dataset mode (D) -> audit with `python inspect_dataset.py` (per-class counts, imbalance ratio, near-duplicate estimate) -> train via notebooks OR `train.py` -> copy the `tflite/` outputs plus `label_encoder.pkl` and `threshold.json` into `models/` (detection) or `drawing_models/` (drawing).

## Model File Locations

- Detection models: `models/tflite/gesture_classifier.tflite`, `models/tflite/autoencoder.tflite`, `models/label_encoder.pkl`, `models/threshold.json`
- Drawing models: `drawing_models/tflite/gesture_classifier.tflite`, `drawing_models/tflite/autoencoder.tflite`, `drawing_models/label_encoder.pkl`, `drawing_models/threshold.json`
- Datasets: `data/hand_landmarks_dataset.csv` (auto), `data/hand_landmarks_dataset_manual.csv` (manual)
