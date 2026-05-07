# AirSketch: Local Server vs Cloud Server vs On-Device — Comparison

## Current Architecture

- **Quest 3 Unity client** captures camera frames, JPEG-encodes them, and streams them over TCP to a local PC
- **AirSketch_server.py** (local PC) receives frames, runs MediaPipe hand detection + TFLite gesture classification, sends back `GES:<gesture>` strings
- Models are tiny: 20-44KB TFLite files, 42-float input (21 landmarks × 2 coords)
- Protocol: 4-byte LE length prefix + payload, raw TCP socket on port 9999

---

## Option A: Cloud Server

**Effort:** Low — mostly infrastructure and networking changes

**What to change:**
1. Deploy `AirSketch_server.py` on a cloud VM (e.g., AWS EC2, GCP, Azure)
2. Replace raw TCP with WebSocket or HTTPS for NAT/firewall traversal
3. Add authentication (API key or token) so the server isn't open
4. Add TLS encryption for data in transit
5. Unity client connects to cloud IP instead of LAN IP

**Trade-offs:**

| Aspect | Assessment |
|--------|------------|
| Latency | Bad — full JPEG frames over internet adds 50-200ms+ round-trip |
| Bandwidth | ~500KB-2MB per frame × 30fps = high data usage |
| Cost | VM running 24/7 or serverless cold starts |
| Offline | Doesn't work without internet |
| Multi-user | Easy to support multiple clients |
| Model updates | Easy — just update server, no APK rebuild |

**Verdict:** Only makes sense if you plan to add much heavier models (e.g., large vision transformers) that can't run on Quest 3. For current 20-44KB TFLite models, this is overkill.

---

## Option B: On-Device in Unity (Quest 3) — Recommended

**Effort:** Medium — requires Unity integration work, but eliminates the server entirely

**What to change:**
1. Use **Meta Quest hand tracking API** (OpenXR) instead of MediaPipe — Quest 3 already tracks hand landmarks natively, no need to stream camera frames for hand detection
2. Map Quest hand tracking landmarks to your model's 42-float format (21 landmarks × 2 coords, wrist-normalized)
3. Run TFLite inference on-device using a Unity inference runtime (see options below)
4. Bundle the model files (~112KB total) in the APK's StreamingAssets

### Model Files to Bundle

| File | Path | Size | Purpose |
|------|------|------|---------|
| Gesture classifier | `models/tflite/gesture_classifier.tflite` | 25KB | Classifies hand pose into gesture classes (thumbs_up, peace, fist, etc.) |
| Gesture autoencoder | `models/tflite/autoencoder.tflite` | 44KB | Anomaly detection — rejects unknown gestures if reconstruction error is too high |
| Drawing classifier | `drawing_models/tflite/gesture_classifier.tflite` | 23KB | Classifies hand pose as "pen" or "eraser" |
| Drawing autoencoder | `drawing_models/tflite/autoencoder.tflite` | 20KB | Anomaly detection for drawing gestures |

Plus support files:
- `models/label_encoder.pkl` / `drawing_models/label_encoder.pkl` — maps class indices to gesture names
- `models/threshold.json` / `drawing_models/threshold.json` — autoencoder reconstruction error threshold

### Unity Inference Runtime Options

**Option 1: Unity Sentis (formerly Barracuda) — Recommended**
- Unity's built-in neural network inference engine (no third-party dependency)
- Requires converting `.tflite` → `.onnx` format first (one-time step):
  ```bash
  pip install tf2onnx
  python -m tf2onnx.convert --tflite models/tflite/gesture_classifier.tflite --output models/gesture_classifier.onnx
  ```
- Works on Quest 3 (CPU and GPU backends)
- Example C# inference:
  ```csharp
  var model = ModelLoader.Load("gesture_classifier.onnx");
  var worker = new Worker(model, BackendType.GPUCompute);
  var input = new Tensor<float>(new TensorShape(1, 42), landmarks);
  worker.Schedule(input);
  var output = worker.PeekOutput().ReadbackAndClone();
  ```

**Option 2: TensorFlow Lite C# Plugin**
- Use a native TFLite shared library (`.so` for Android/Quest) with C# bindings
- Keeps `.tflite` files as-is — no conversion needed
- More manual setup: need `libtensorflowlite.so` for Android ARM64
- Community plugins like [tf-lite-unity-sample](https://github.com/asus4/tf-lite-unity-sample) exist

**Option 3: MediaPipe Unity Plugin**
- [homuler/MediaPipeUnityPlugin](https://github.com/homuler/MediaPipeUnityPlugin) brings MediaPipe to Unity
- Overkill since Quest 3 already provides hand tracking

### Landmark Mapping Concern

Quest 3 hand tracking provides 26 joint positions (XrHandJointEXT) in 3D world space. The models expect 21 MediaPipe landmarks in 2D normalized coords. Options:
- **Project Quest 3D landmarks to 2D and normalize** (simpler, may work with existing models)
- **Retrain models on Quest landmark data** (more robust)

**Trade-offs:**

| Aspect | Assessment |
|--------|------------|
| Latency | Excellent — everything local, sub-millisecond inference |
| Bandwidth | Zero — no network needed |
| Cost | Zero — no server |
| Offline | Works completely offline |
| Model updates | Requires APK rebuild (or implement hot-loading from a URL) |
| Hand tracking | Quest native tracking may differ slightly from MediaPipe — may need mapping layer or retraining |

---

## Option C: Hybrid

Run inference on-device for real-time feedback, but optionally sync to a cloud server for:
- Logging / analytics
- A/B testing new models (download updated TFLite files)
- Heavier processing that doesn't need to be real-time

Most flexible long-term but highest upfront effort.

---

## Recommendation Summary

| | Cloud Server | On-Device | Hybrid |
|---|---|---|---|
| **Best for** | Heavy models, multi-user | Current tiny models | Long-term product |
| **Effort** | Low | Medium | High |
| **Latency** | Poor | Excellent | Excellent |
| **Cost** | Ongoing | None | Low |
| **Recommendation** | Not recommended for current use case | **Strongly recommended** | Future consideration |

**Bottom line:** Move to on-device inference. The 20-44KB models with 42-float input are trivially fast on Quest 3 hardware, and the headset already provides hand tracking — streaming camera frames to a PC server is unnecessary overhead.
