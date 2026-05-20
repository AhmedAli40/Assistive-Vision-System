"""
config.py - Unified settings for the integrated system
"""
import os

# ══════════════════════════════════════════
#  Language ("en" = English, "ar" = Arabic)
#  User can switch at runtime via voice command
# ══════════════════════════════════════════
LANGUAGE = "en"   # default: English

# ══════════════════════════════════════════
#  Camera
# ══════════════════════════════════════════
CAMERA_INDEX  = 1
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
TARGET_FPS    = 15

# ══════════════════════════════════════════
#  Emotion model
# ══════════════════════════════════════════
MODEL_PATH        = "models/cnn_v3_final.h5"
IMG_SIZE          = (48, 48)
INFERENCE_EVERY_N = 10       # increased for TTA (5 passes per inference)

EMOTIONS_EN = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

# ══════════════════════════════════════════
#  Dynamic confidence threshold
# ══════════════════════════════════════════
CONFIDENCE_MIN       = 0.45
CONFIDENCE_MIN_LOW   = 0.40  # was 0.35 — less audio fallback
BRIGHTNESS_THRESHOLD = 70    # was 80 — less sensitive to lighting

# ══════════════════════════════════════════
#  Smoothing — faster emotion response
# ══════════════════════════════════════════
SMOOTHING_WINDOW = 12   # كان 8 — نزوده عشان يبقى أثبت
MIN_STABLE_RATIO = 0.60  # كان 0.55 — نزوده عشان يتأكد أكتر

# ══════════════════════════════════════════
#  Audio fallback
# ══════════════════════════════════════════
AUDIO_ENABLED      = True
AUDIO_SAMPLE_RATE  = 16000
AUDIO_RECORD_SEC   = 2.0    # was 2.5 — faster response
AUDIO_DEVICE_INDEX = None

# ══════════════════════════════════════════
#  TTS / STT
# ══════════════════════════════════════════
TTS_COOLDOWN_SEC = 3.0   # 3 seconds between announcements
TTS_RATE         = 150      # was 140 — slightly faster speech

# ══════════════════════════════════════════
#  Display
# ══════════════════════════════════════════
SHOW_WINDOW  = True
WINDOW_TITLE = "Assistive Vision System"

EMOTION_COLORS = {
    "Angry":    (0,   0,   255),
    "Disgust":  (0,   140, 255),
    "Fear":     (128, 0,   128),
    "Happy":    (0,   255, 0  ),
    "Neutral":  (200, 200, 200),
    "Sad":      (255, 100, 0  ),
    "Surprise": (0,   255, 255),
}

# ══════════════════════════════════════════
#  Face Recognition
# ══════════════════════════════════════════
FACE_DB_PATH   = "face_data.pkl"
BLOCKED_PATH   = "blocked.json"
FACE_THRESHOLD = 0.50
LBP_THRESHOLD  = 18.0

# ══════════════════════════════════════════
#  Logic Controller
# ══════════════════════════════════════════
UNKNOWN_REASK_TIMEOUT      = 10.0
UNKNOWN_FRAMES_BEFORE_ASK  = 20

# ══════════════════════════════════════════
#  Logging
# ══════════════════════════════════════════
LOG_ENABLED = True
LOG_DIR     = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ══════════════════════════════════════════
#  STT Settings
# ══════════════════════════════════════════
VOSK_ENABLED = True   # False = Google only, True = Vosk fallback when offline

# ══════════════════════════════════════════
#  Performance & Quality Tuning
# ══════════════════════════════════════════

# TTA — only run 5-pass TTA when confidence is below this threshold
# Above threshold: single fast predict (~50ms)
# Below threshold: full TTA (~250ms) for better accuracy
TTA_CONFIDENCE_THRESHOLD = 0.55  # reduced — less TTA = faster response

# Emotion smoothing — history window per face
EMOTION_HISTORY_SIZE = 3     # minimal smoothing — 3 frames only
EMOTION_STABLE_RATIO = 0.50  # 2 out of 3 frames must agree

# Adaptive inference rate
INFERENCE_STABLE_N  = 8    # reduced from 15 — stay responsive
INFERENCE_ACTIVE_N  = 4    # every 4 frames = ~4 reads/sec at 15fps
EMOTION_STABLE_FRAMES = 20  # frames of same emotion before considered "stable"

# TTS audio cache — cache generated mp3 files for repeated phrases
TTS_CACHE_ENABLED = True
TTS_CACHE_MAX     = 50     # max phrases to cache in memory

# Face recognition adaptive threshold
FACE_THRESHOLD_LOW_LIGHT = 0.55   # more lenient in low light (brightness < 70)
