# Emotion & Face Recognition System — Complete Project Log

> **Project:** Assistive Vision System for the Visually Impaired  
> **Platform:** Raspberry Pi 4 Model B (target) / Windows (development)  
> **Student:** Ahmed Ali (Emotion Recognition + Face Recognition)  
> **Teammate:** Ismail Mohsen (Voice Assistant Integration)  
> **Date Range:** April 16 – May 13, 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1: Initial Setup & Environment](#2-phase-1-initial-setup--environment)
3. [Phase 2: First Emotion Model (Baseline)](#3-phase-2-first-emotion-model-baseline)
4. [Phase 3: Model Improvements & Training](#4-phase-3-model-improvements--training)
5. [Phase 4: Face Recognition Module](#5-phase-4-face-recognition-module)
6. [Phase 5: Integration & Logic System](#6-phase-5-integration--logic-system)
7. [Phase 6: System Optimization](#7-phase-6-system-optimization)
8. [Phase 7: Advanced Model Training (v3-v6)](#8-phase-7-advanced-model-training-v3-v6)
9. [Phase 8: Final Results & Deployment Prep](#9-phase-8-final-results--deployment-prep)
10. [GitHub & Google Drive Upload](#10-github--google-drive-upload)
11. [Technical Specifications](#11-technical-specifications)
12. [Challenges & Solutions Summary](#12-challenges--solutions-summary)
13. [Voice Commands Reference](#13-voice-commands-reference)
14. [File Structure](#14-file-structure)
15. [Final Model Comparison](#15-final-model-comparison)

---

## 1. Project Overview

### Goal
Build an assistive vision system for visually impaired users that:
- Detects faces in real-time
- Recognizes registered faces
- Reads emotions from facial expressions
- Speaks results via Text-to-Speech (TTS)
- Accepts voice commands via Speech-to-Text (STT)
- Runs on Raspberry Pi 4 Model B

### Key Features
| Feature | Description |
|---------|-------------|
| **Face Detection** | MTCNN (primary) + Haar Cascade (fallback) |
| **Face Recognition** | Facenet512 with LBP Liveness Detection |
| **Emotion Recognition** | Custom CNN (7 classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) |
| **Audio Fallback** | Analyzes voice when face confidence is low |
| **Voice Commands** | Wake word "Vision" + command recognition |
| **Offline STT** | Vosk fallback when no internet |
| **Multi-face Support** | Handles multiple faces with announcement queue |
| **Logging** | CSV session logs |

---

## 2. Phase 1: Initial Setup & Environment

### Environment Setup
```bash
# Python version
python --version  # Python 3.12.3

# Install dependencies
pip install opencv-python fer tensorflow pyttsx3 numpy sounddevice librosa
pip install mtcnn
pip install tensorflow-hub
pip install seaborn
pip install vosk
```

### Camera Configuration
```python
CONFIG = {
    "camera_index": 0,        # 0 = laptop camera, 1 = external
    "frame_width": 640,
    "frame_height": 480,
    "target_fps": 15,
}
```

### Issues Faced
| Issue | Solution |
|-------|----------|
| librosa installation failed | Installed individually with `--upgrade` |
| llvmlite download interrupted | Re-ran `pip install librosa` |

---

## 3. Phase 2: First Emotion Model (Baseline)

### Dataset
- **FER-2013**: 28,709 training images + 7,178 test images
- 7 classes: angry, disgust, fear, happy, neutral, sad, surprise
- Grayscale 48×48 pixels

### Initial Model (CNN v1)
- Architecture: Simple CNN
- Epochs: 50
- **Final Accuracy: 66.54%**

### Model File
```
D:\emotion_recognition Acc 66.5\models\emotion_model.keras
```

### Test Results (Per-Class)
| Emotion | Accuracy |
|---------|----------|
| Happy | 85% |
| Surprise | 84% |
| Neutral | 75% |
| Disgust | 69% |
| Angry | 64% |
| Sad | 45% |
| Fear | 35% |

### Key Issue Identified
- Fear and Sad confusion due to similar facial expressions
- This is a known limitation of FER-2013 dataset

---

## 4. Phase 3: Model Improvements & Training

### CNN v2 (100 epochs)
- Continued training from v1
- Added brightness augmentation
- Lower learning rate (0.0005)
- **Accuracy: 65.87%** (slightly lower than v1)
- **Conclusion**: v1 was better; model reached its ceiling

### MobileNetV2 Attempt (Failed)
- Transfer learning from ImageNet
- Issue: MobileNetV2 trained on natural color images, FER-2013 is grayscale faces
- Result: Started at 17%, reached only 48% in Phase 1
- **Conclusion**: Transfer learning not suitable for FER-2013

### Smoothing System Added
```python
CONFIG = {
    "smoothing_window": 10,      # Last 10 results
    "min_stable_ratio": 0.5,     # 50% agreement needed
    "inference_every_n": 5,      # Analyze every 5 frames
    "tts_cooldown_sec": 6.0,     # Speak every 6 seconds
    "confidence_threshold": 0.45, # Lowered from 0.65
}
```

### Dynamic Threshold
- Normal light: threshold = 45%
- Low light (brightness < 80): threshold = 35%

---

## 5. Phase 4: Face Recognition Module

### Technologies Used
| Component | Technology | Accuracy |
|-----------|------------|----------|
| Face Detection | MTCNN | High accuracy |
| Face Recognition | Facenet512 | 99.65% (LFW benchmark) |
| Liveness Detection | LBP (Local Binary Patterns) | Threshold > 18.0 |
| Matching | Cosine Distance | Threshold = 0.50 |

### Registration Flow
1. System detects face
2. Asks: "Do you want to register this person?"
3. User says: "Yes"
4. System asks: "Please say the name"
5. User says name
6. Face embedding saved to database

### Database
- File: `face_data.pkl`
- Stores: Name + face embedding vector
- Format: Python pickle

---

## 6. Phase 5: Integration & Logic System

### Logic Flow
```
Person Appears
    ↓
Blocked? → YES → Silent (ignore completely)
    ↓ NO
Known? → YES → "[Name] looks [Emotion]"
    ↓ NO
"Unknown person, they look [Emotion]"
    ↓
Continuous emotion reading every 6 seconds
    ↓
Voice Commands Available Anytime:
    - "Vision" → "Yes?" (activate)
    - "Register" → Register current person
    - "Block" / "Unblock" → Manage blocked list
    - "Who" → Identify current person
    - "Quiet" / "Speak" → Control announcements
    - "Stop" → Stop current speech
    - "List" / "Delete" → Database management
```

### Multi-Face Support
- Detects all faces in frame
- Sorts by size (closest to camera first)
- Announces emotions one by one with queue system
- Waits for each announcement to finish before next

### TTS (Text-to-Speech)
- Windows: `win32com` (SAPI5)
- Raspberry Pi: `espeak` (to be implemented)
- Shared module between Face and Emotion systems

### STT (Speech-to-Text) Strategy
| Mode | Technology | Use Case |
|------|------------|----------|
| Online | Google Speech API | High accuracy when internet available |
| Offline | Vosk (small-en-us-0.15) | Fallback when no internet |
| Auto-switch | Ping check every 10s | Seamless transition |

### Wake Word System
1. System listens for "Vision"
2. Responds: "Yes?"
3. Listens for command
4. Executes command

---

## 7. Phase 6: System Optimization

### Config Optimizations
```python
# config.py
INFERENCE_EVERY_N = 6           # Was 4 (less CPU load)
CONFIDENCE_MIN_LOW = 0.40       # Was 0.35 (less audio fallback)
BRIGHTNESS_THRESHOLD = 70       # Was 80 (less light sensitivity)
SMOOTHING_WINDOW = 8            # Was 12 (faster response)
AUDIO_RECORD_SEC = 2.0          # Was 2.5 (faster)
TTS_RATE = 150                  # Was 140 (slightly faster speech)
VOSK_ENABLED = True             # Toggle offline mode
```

### Performance Improvements
| Metric | Before | After |
|--------|--------|-------|
| CPU Load | High | Reduced |
| Audio Fallback | Frequent | Reduced |
| Response Time | Slower | Faster |
| TTS Speed | Slower | Faster |

---

## 8. Phase 7: Advanced Model Training (v3-v6)

### CNN v3 (Residual CNN)
- **Architecture**: Residual blocks + Global Average Pooling
- **Datasets**: FER-2013 + RAF-DB (40,980 images)
- **Epochs**: Up to 100 with Early Stopping
- **Training Time**: ~8 hours on CPU
- **FER-2013 Test Accuracy: 68.67%**
- **Validation Accuracy (FER+RAF+CK+): 69.46%**

### CNN v4 (MobileNetV2)
- **Result**: Failed (24% accuracy)
- **Reason**: Transfer learning incompatible with grayscale faces

### CNN v5 (EfficientNetB3)
- **Result**: Failed during training
- **Reason**: Memory/optimization issues

### CNN v6 (VGG16 + CBAM Attention) ⭐
- **Architecture**: VGG16 from scratch + CBAM attention modules
- **Datasets**: FERPlus + RAF-DB + CK+ (~42,500 images)
- **Optimizer**: SGD + Nesterov momentum
- **LR Schedule**: Cosine Annealing (300 epochs)
- **Augmentation**: Mixup + Rotation + Flip + Zoom + Brightness
- **Special Features**:
  - Class weights for imbalanced data
  - Test-Time Augmentation (TTA) — 5 passes
- **Standard Accuracy: 72.05%**
- **TTA Accuracy: 74.43%**

### Ensemble Attempt
- v3 + v6 combined
- Result: 81.35% (but simpler to use v3 + TTA = 81.50%)
- **Note**: These high numbers are on validation set, not FER-2013 test

### Google Colab Training
- Used for v3, v4, v5, v6
- GPU: T4 (free tier)
- Epoch time: ~2-3 minutes (vs ~30 min on CPU)
- Downloaded models as `.h5` and `.tflite`

---

## 9. Phase 8: Final Results & Deployment Prep

### Final Model Selected
**CNN v3** (cnn_v3_best.h5)
- **FER-2013 Test Accuracy: 68.67%**
- **Improvement over baseline: +2.8%**
- **File size**: ~60 MB
- **Format**: HDF5 (.h5)

### Why Not v6 (74.43%)?
- The 74.43% was measured on validation set (FERPlus+RAF+CK+)
- On FER-2013 test set, v3 achieved 68.67% which is the realistic metric
- v6 is more complex and slower for real-time inference

### Model Conversion for Pi
```python
# Convert to TFLite for Raspberry Pi
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save as .tflite file
```

### Raspberry Pi Deployment Plan
1. **TTS**: Change `win32com` → `espeak`
2. **Paths**: Update to `/home/pi/...`
3. **Camera**: Adjust index and resolution
4. **Autostart**: Add to systemd or rc.local
5. **Performance**: Reduce inference frequency, lower resolution

---

## 10. GitHub & Google Drive Upload

### GitHub Repository
- **URL**: https://github.com/AhmedAli40/Assistive-Vision-System.git
- **Type**: Private
- **Files**: All code (18 files)
- **Excluded**: Models, logs, data files (via .gitignore)

### Google Drive Folder
- **URL**: https://drive.google.com/drive/folders/1FHg9-D0uk08p9xlptWRWBJ5LGU-Uq7oe
- **Contents**:
  - `emotion_fixed.h5` (working model)
  - `emotion_model.keras` (original model)
  - `cnn_v3_best.h5` (best model)
  - `vosk-model/` (offline STT)
  - `face_data.pkl` (face database)
  - Training charts and confusion matrices

### .gitignore
```
models/
logs/
face_data.pkl
blocked.json
__pycache__/
*.pyc
*.h5
*.pkl
```

---

## 11. Technical Specifications

### Development Hardware
| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i7-7600U @ 2.80GHz |
| Cores | 2 cores / 4 threads |
| RAM | 16 GB |
| GPU | None (Intel HD Graphics 620) |
| OS | Windows 10/11 |

### Target Hardware (Raspberry Pi 4B)
| Component | Specification |
|-----------|--------------|
| CPU | Broadcom BCM2711, Quad-core Cortex-A72 |
| RAM | 4 GB |
| GPU | VideoCore VI |
| OS | Raspberry Pi OS (64-bit) |

### Software Versions
| Library | Version |
|---------|---------|
| Python | 3.12.3 |
| TensorFlow | 2.21.0 |
| Keras | 3.x (via tf_keras) |
| OpenCV | 4.9.0 |
| NumPy | 1.26.4 |
| DeepFace | Latest |
| Vosk | 0.3.45 |

---

## 12. Challenges & Solutions Summary

| # | Challenge | Solution |
|---|-----------|----------|
| 1 | librosa installation failed | Installed individually with retries |
| 2 | Camera using external instead of laptop | Changed `camera_index` from 1 to 0 |
| 3 | Emotions changing too fast | Added smoothing system (10-frame window) |
| 4 | TTS speaking too frequently | Increased cooldown from 3s to 6s |
| 5 | Low confidence in low light | Dynamic threshold (45% → 35%) |
| 6 | Model not loading (batch_shape error) | Converted from .keras to .h5 format |
| 7 | tf_keras overriding keras | Loaded model before DeepFace initialization |
| 8 | STT blocking camera | Moved STT to background thread with delay |
| 9 | Microphone calibration crashing | Added try-except with default threshold |
| 10 | Multi-face announcement chaos | Implemented announcement queue system |
| 11 | TTS and STT conflicts | Unified TTS module, wake word system |
| 12 | Offline mode needed | Integrated Vosk as fallback |
| 13 | EfficientNetB0 failing on FER-2013 | Switched to custom CNN from scratch |
| 14 | Training too slow on CPU | Used Google Colab with T4 GPU |
| 15 | Model version confusion | Documented all versions with clear naming |

---

## 13. Voice Commands Reference

### Wake Word
- Say: **"Vision"**
- Response: **"Yes?"**
- Then say any command below

### Registration Commands
| Command | Action |
|---------|--------|
| "register" | Register current person |
| "delete" | Delete a registered person |
| "list" | List all registered names |

### Blocking Commands
| Command | Action |
|---------|--------|
| "block" | Block current person permanently |
| "unblock" | Unblock a person |

### Information Commands
| Command | Action |
|---------|--------|
| "who" | Identify current person and emotion |

### Control Commands
| Command | Action |
|---------|--------|
| "quiet" | Stop all announcements |
| "speak" | Resume announcements |
| "stop" | Stop current speech immediately |

---

## 14. File Structure

```
Assistive-Vision-System/
│
├── main.py                      # Entry point
├── config.py                    # All settings
├── logic_controller.py          # System logic brain
├── convert_model.py             # Model conversion utility
├── install.bat                  # Install dependencies
├── run.bat                      # Run the system
├── README.md                    # Setup instructions
│
├── models/                      # MODELS FROM GOOGLE DRIVE
│   ├── emotion_fixed.h5         # Current working model
│   ├── cnn_v3_best.h5           # Best trained model (68.67%)
│   ├── cnn_v3.tflite            # TFLite for Pi deployment
│   └── vosk-model/              # Offline STT model
│       ├── am/
│       ├── conf/
│       ├── graph/
│       ├── ivector/
│       └── README
│
├── face/                        # FACE RECOGNITION MODULE
│   ├── __init__.py
│   ├── face_db.py              # Database management
│   ├── face_processor.py       # Face detection & recognition
│   └── registration.py         # Registration flow
│
├── emotion/                     # EMOTION RECOGNITION MODULE
│   ├── __init__.py
│   ├── face_detector.py        # MTCNN + emotion CNN
│   ├── audio_detector.py       # Audio fallback analysis
│   └── display.py              # Visualization
│
├── shared/                      # SHARED COMPONENTS
│   ├── __init__.py
│   ├── tts.py                  # Text-to-Speech (win32com/espeak)
│   └── stt.py                  # Speech-to-Text (Google + Vosk)
│
├── logs/                        # Auto-generated CSV logs
├── face_data.pkl               # Auto-generated face database
└── blocked.json                # Auto-generated blocked list
```

---

## 15. Final Model Comparison

### All Models Trained
| Version | Architecture | Dataset | Epochs | FER-2013 Test | Val Set | Status |
|---------|-------------|---------|--------|---------------|---------|--------|
| Baseline | Simple CNN | FER-2013 | 50 | **65.87%** | — | ✅ Used initially |
| v1 | CNN | FER-2013 | 50 | 66.54% | — | ✅ Best single on FER |
| v2 | CNN (continued) | FER-2013 | 100 | 65.87% | — | ⚠️ Lower than v1 |
| v3 | Residual CNN | FER-2013 + RAF-DB | ~70 | **68.67%** | 69.46% | ✅ **FINAL SELECTED** |
| v4 | MobileNetV2 | FER+RAF+CK+ | 10 | ~24% | 67.52% | ❌ Failed |
| v5 | EfficientNetB3 | FER+RAF+CK+ | — | — | — | ❌ Failed |
| v6 | VGG16 + CBAM | FERPlus+RAF+CK+ | 69 | — | 72.05% / 74.43% TTA | ✅ Good but complex |

### Key Metrics (Final Model: CNN v3)
| Metric | Value |
|--------|-------|
| Overall Accuracy | 68.67% |
| Happy Accuracy | ~87% |
| Surprise Accuracy | ~84% |
| Neutral Accuracy | ~75% |
| Angry Accuracy | ~64% |
| Disgust Accuracy | ~56% |
| Sad Accuracy | ~48% |
| Fear Accuracy | ~38% |
| Model Size | ~60 MB |
| Inference Time | ~50ms per frame |

### Per-Class Accuracy (FER-2013 Test Set)
```
              precision    recall  f1-score   support
       Angry       0.55      0.64      0.59       958
     Disgust       0.48      0.69      0.56       111
        Fear       0.60      0.35      0.44      1024
       Happy       0.90      0.85      0.87      1774
     Neutral       0.54      0.75      0.63      1233
         Sad       0.57      0.45      0.51      1247
    Surprise       0.73      0.84      0.78       831
    accuracy                           0.66      7178
   macro avg       0.62      0.65      0.63      7178
weighted avg       0.67      0.66      0.65      7178
```

---

## Dataset Details

### FER-2013
- Source: Kaggle
- Images: 28,709 train + 7,178 test
- Format: Grayscale 48×48
- Classes: 7 (angry, disgust, fear, happy, neutral, sad, surprise)
- Issues: Some mislabeled images, low resolution

### RAF-DB
- Source: Kaggle
- Images: ~12,271 train
- Format: Color, various sizes
- Classes: 7 (mapped to FER-2013 labels)
- Quality: Higher quality, real-world photos

### CK+ (Extended Cohn-Kanade)
- Source: Kaggle
- Images: ~1,500
- Format: Color, lab-controlled
- Quality: Very high quality

---

## Training Configuration (CNN v3)

```python
# Optimizer
optimizer = Adam(learning_rate=0.001)

# Learning Rate Schedule
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# Early Stopping
EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True
)

# Data Augmentation
ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)

# Class Weights (for imbalanced data)
class_weights = {
    0: 1.02,   # angry
    1: 9.41,   # disgust (rarest)
    2: 1.33,   # fear
    3: 0.77,   # happy (most common)
    4: 1.11,   # neutral
    5: 1.10,   # sad
    6: 1.65    # surprise
}
```

---

## Discussion Points for Defense

### Why 68.67% is Good Enough
1. **FER-2013 is inherently difficult**: Even humans disagree on labels
2. **State-of-the-art on FER-2013**: ~75% with massive resources
3. **Our constraints**: No GPU, lightweight model for Pi
4. **Real-world performance**: Smoothing and audio fallback compensate

### Why Custom CNN Over Transfer Learning
1. **FER-2013 is grayscale 48×48**: Pretrained models expect color 224×224
2. **Domain mismatch**: ImageNet (objects) vs FER (faces)
3. **Speed**: Custom CNN is faster for real-time inference
4. **Size**: Custom CNN is smaller (~60MB vs ~100MB+)

### Why MTCNN Over Haar Cascade
1. **Accuracy**: MTCNN detects faces at angles and low light
2. **Confidence scores**: MTCNN provides detection confidence
3. **Modern**: Deep learning-based vs rule-based

### Audio Fallback Strategy
1. **When activated**: Face confidence < 40%
2. **How it works**: Analyzes voice energy and pitch
3. **Why needed**: Compensates for poor lighting/face occlusion

---

## Future Work

1. **Raspberry Pi Deployment**
   - Convert to TFLite
   - Optimize for ARM architecture
   - Test real-world performance

2. **Model Improvements**
   - Collect custom dataset for better accuracy
   - Try EfficientNet with proper grayscale adaptation
   - Implement attention mechanisms

3. **Feature Additions**
   - Age estimation
   - Gender classification
   - Multiple language support
   - Mobile app companion

4. **Hardware Upgrades**
   - Coral USB Accelerator for edge TPU
   - Better camera module
   - External microphone array

---

## References

1. Goodfellow, I.J., et al. "Challenges in Representation Learning: A report on three machine learning contests." *Neural Networks* (2013).
2. Li, S., Deng, W. "Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild." *CVPR* (2017).
3. Lucey, P., et al. "The Extended Cohn-Kanade Dataset (CK+)." *FG* (2010).
4. Schroff, F., Kalenichenko, D., Philbin, J. "FaceNet: A Unified Embedding for Face Recognition and Clustering." *CVPR* (2015).
5. Zhang, K., et al. "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks." *IEEE Signal Processing Letters* (2016).
6. He, K., et al. "Deep Residual Learning for Image Recognition." *CVPR* (2016).
7. Tan, M., Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML* (2019).
8. Woo, S., et al. "CBAM: Convolutional Block Attention Module." *ECCV* (2018).
9. Vosk API Documentation: https://alphacephei.com/vosk/
10. DeepFace Library: https://github.com/serengil/deepface

---

## Timeline Summary

| Date | Milestone |
|------|-----------|
| Apr 16 | Project started, environment setup |
| Apr 17 | First model trained (66.54%) |
| Apr 18 | Smoothing system added, camera fixed |
| Apr 19 | MTCNN implemented, audio fallback added |
| Apr 20 | Logic system designed, GitHub repo created |
| Apr 22 | Face recognition integrated, voice commands added |
| Apr 28 | Full integration complete, testing started |
| Apr 30 | Model loading issues resolved, system stable |
| May 1 | Multi-face support, Vosk offline mode |
| May 2 | Final optimizations, README completed |
| May 5 | Report generation started |
| May 6 | GitHub + Google Drive upload complete |
| May 7 | CNN v3 training started (CPU) |
| May 8 | Google Colab training setup |
| May 9 | CNN v3 completed (69.46% val) |
| May 10 | CNN v6 completed (74.43% TTA) |
| May 11 | FER-2013 test: v3 = 68.67% |
| May 13 | Final model selected, project documented |

---

> **End of Document**  
> Generated: May 15, 2026  
> Total Development Time: ~4 weeks  
> Models Trained: 7 versions  
> Final Accuracy: 68.67% (FER-2013 test)  
> Status: Ready for teammate integration → Raspberry Pi deployment
