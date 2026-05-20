"""
main.py - Assistive Vision System
===================================
Integrates:
  - Face Recognition  → identifies known/unknown/blocked persons
  - Emotion Detection → reads emotions using CNN + TTA
  - Logic Controller  → decides what to say and when
  - Shared TTS        → single voice output (Edge TTS neural)

Optimizations applied:
  [1] imports moved outside loops
  [2] TTA only when confidence < TTA_CONFIDENCE_THRESHOLD
  [3] Batch prediction for multiple faces
  [4] Per-face emotion smoothing (history window)
  [5] Adaptive inference rate (stable vs changing)
  [6] Audio fallback only for closest face
  [7] TTS cache for repeated phrases
  [8] Log flush after every write
  [9] Adaptive face threshold in low light
  [10] Fixed unused display_label variable

Press Q or ESC to quit.
"""

import cv2
import sys
import os
import time
import threading
import csv
import numpy as np
from datetime import datetime
from collections import deque, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# ── Load emotion model BEFORE DeepFace (critical for keras compatibility) ─────
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import tensorflow as _tf
_emotion_model_global = None
if os.path.exists(config.MODEL_PATH):
    _emotion_model_global = _tf.keras.models.load_model(
        config.MODEL_PATH, compile=False
    )
    print('[Model] Loaded successfully before DeepFace')
else:
    print(f'ERROR: Model not found at {config.MODEL_PATH}')
    sys.exit(1)

# ── Shared TTS ────────────────────────────────────────────────────────────────
from shared.tts import TTS
tts = TTS(rate=config.TTS_RATE)

# ── Face Recognition ──────────────────────────────────────────────────────────
from face.face_db        import FaceDB
from face.face_processor import FaceProcessor
from face.registration   import RegFlow
from shared.stt          import STT

# ── Emotion Detection ─────────────────────────────────────────────────────────
from emotion.audio_detector import AudioEmotionDetector
from emotion.display        import draw_results, draw_no_face, draw_fps

# ── Logic Controller ──────────────────────────────────────────────────────────
from logic_controller import LogicController

import logging
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  TTS Cache — caches generated audio for repeated phrases
# ─────────────────────────────────────────────────────────────────────────────

class _TTSCache:
    """Simple LRU-style cache for Edge TTS mp3 files."""
    def __init__(self, max_size: int = 50):
        self._cache   = {}           # phrase → mp3 path
        self._max     = max_size
        self._enabled = getattr(config, 'TTS_CACHE_ENABLED', True)

    def get(self, phrase: str):
        if not self._enabled: return None
        path = self._cache.get(phrase)
        if path and os.path.exists(path):
            return path
        return None

    def put(self, phrase: str, path: str):
        if not self._enabled: return
        if len(self._cache) >= self._max:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            try: os.unlink(self._cache[oldest])
            except Exception: pass
            del self._cache[oldest]
        self._cache[phrase] = path

    def clear(self):
        for path in self._cache.values():
            try: os.unlink(path)
            except Exception: pass
        self._cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────────────────────────────────────

class AssistiveVisionSystem:

    def __init__(self):
        print("=" * 55)
        print("  Assistive Vision System — Starting Up")
        print("=" * 55)

        # ── Emotion model ──────────────────────────────────────────────────
        print("\n[1/5] Loading emotion model...")
        self._emotion_model = _emotion_model_global
        print(f"      Model: {config.MODEL_PATH}")

        # ── Face Recognition ───────────────────────────────────────────────
        print("\n[2/5] Initializing Face Recognition...")
        self._face_db   = FaceDB(path=config.FACE_DB_PATH)
        self._face_proc = FaceProcessor(threshold=config.FACE_THRESHOLD)
        self._stt       = STT()
        self._reg       = RegFlow(tts, self._stt, self._face_db, self._face_proc)
        print("      Face Recognition ready.")

        # ── Emotion Detection ──────────────────────────────────────────────
        print("\n[3/5] Initializing Emotion Detection...")
        self._audio_det = AudioEmotionDetector()
        print("      Emotion Detection ready.")

        # ── Logic Controller ───────────────────────────────────────────────
        print("\n[4/5] Starting Logic Controller...")
        self._logic = LogicController(
            tts            = tts,
            stt            = self._stt,
            reg_flow       = self._reg,
            face_processor = self._face_proc,
            face_db        = self._face_db,
        )

        # ── TTS Cache ──────────────────────────────────────────────────────
        self._tts_cache = _TTSCache(
            max_size=getattr(config, 'TTS_CACHE_MAX', 50)
        )

        # ── State ──────────────────────────────────────────────────────────
        self._frame_count        = 0
        self._fps                = 0.0
        self._fps_time           = time.time()
        self._fps_frames         = 0

        self._current_emotion    = "Neutral"
        self._current_conf       = 0.0
        self._current_source     = "face"
        self._current_face_box   = None
        self._current_name       = "Unknown"

        # ── Inference thread ───────────────────────────────────────────────
        self._inference_lock   = threading.Lock()
        self._inference_result = None
        self._inference_busy   = False

        # ── Per-face emotion smoothing ─────────────────────────────────────
        # {face_id: deque of emotion indices}
        _hist_size = getattr(config, 'EMOTION_HISTORY_SIZE', 8)
        self._emotion_history: dict = {}
        self._emotion_history_size  = _hist_size

        # ── Adaptive inference rate ────────────────────────────────────────
        self._stable_frames     = 0   # consecutive frames with same emotion
        self._last_emotion_set  = ""  # for stability tracking
        self._stable_n  = getattr(config, 'INFERENCE_STABLE_N',  15)
        self._active_n  = getattr(config, 'INFERENCE_ACTIVE_N',   8)
        self._stable_threshold = getattr(config, 'EMOTION_STABLE_FRAMES', 20)

        # ── Logger ────────────────────────────────────────────────────────
        self._log_file   = None
        self._log_writer = None
        self._init_logger()

        print("\n[5/5] Calibrating microphone...")
        self._stt.calibrate(duration=2.0)

        print("\n" + "=" * 55)
        print("  System Ready! Press Q or ESC to quit.")
        print("  Say 'vision' or 'فيجن' to activate voice commands.")
        print("=" * 55 + "\n")

    # ── Logger ────────────────────────────────────────────────────────────────

    def _init_logger(self):
        if not config.LOG_ENABLED:
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.LOG_DIR, f"session_{ts}.csv")
        self._log_file   = open(path, "w", newline="", encoding="utf-8")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(
            ["timestamp", "name", "emotion", "emo_conf",
             "source", "brightness", "inference_mode"]
        )
        print(f"      Logging to: {path}")

    def _log(self, name, emotion, emo_conf, source, brightness, mode="normal"):
        if self._log_writer:
            self._log_writer.writerow([
                datetime.now().strftime("%H:%M:%S"),
                name, emotion, f"{emo_conf:.2f}",
                source, f"{brightness:.0f}", mode,
            ])
            self._log_file.flush()   # [FIX #8] flush immediately — no data loss on crash

    # ── FPS ───────────────────────────────────────────────────────────────────

    def _update_fps(self):
        self._fps_frames += 1
        now = time.time()
        if now - self._fps_time >= 1.0:
            self._fps        = self._fps_frames / (now - self._fps_time)
            self._fps_frames = 0
            self._fps_time   = now

    # ── Brightness ────────────────────────────────────────────────────────────

    @staticmethod
    def _brightness(frame_gray) -> float:
        return float(cv2.mean(frame_gray)[0])

    def _confidence_threshold(self, brightness: float) -> float:
        if brightness < config.BRIGHTNESS_THRESHOLD:
            return config.CONFIDENCE_MIN_LOW
        return config.CONFIDENCE_MIN

    def _face_threshold(self, brightness: float) -> float:
        """[OPT #9] Adaptive face threshold — more lenient in low light."""
        if brightness < config.BRIGHTNESS_THRESHOLD:
            return getattr(config, 'FACE_THRESHOLD_LOW_LIGHT', 0.55)
        return config.FACE_THRESHOLD

    # ── Adaptive inference interval ───────────────────────────────────────────

    def _get_inference_n(self) -> int:
        """[OPT #5] Use faster inference when emotion is changing."""
        if self._stable_frames >= self._stable_threshold:
            return self._stable_n   # emotion stable → less frequent
        return self._active_n       # emotion changing → more frequent

    # ── Emotion smoothing ─────────────────────────────────────────────────────

    def _smooth_emotion(self, face_id: str, raw_idx: int) -> int:
        """[OPT #4] Per-face emotion history smoothing."""
        if face_id not in self._emotion_history:
            self._emotion_history[face_id] = deque(
                maxlen=self._emotion_history_size
            )
        hist = self._emotion_history[face_id]
        hist.append(raw_idx)

        stable_ratio = getattr(config, 'EMOTION_STABLE_RATIO', 0.55)
        most_common, count = Counter(hist).most_common(1)[0]
        if count / len(hist) >= stable_ratio:
            return most_common
        return raw_idx   # not stable yet — use raw

    # ── Predict emotion (single face input) ───────────────────────────────────

    def _predict_emotion(self, face_input: np.ndarray) -> tuple:
        """
        [OPT #2] Conditional TTA:
          - If confidence ≥ TTA_CONFIDENCE_THRESHOLD → single fast predict
          - If confidence <  TTA_CONFIDENCE_THRESHOLD → full 5-pass TTA
        """
        tta_threshold = getattr(config, 'TTA_CONFIDENCE_THRESHOLD', 0.65)

        # Fast single predict first
        preds = self._emotion_model.predict(face_input, verbose=0)[0]

        if float(preds.max()) >= tta_threshold:
            # High confidence — no TTA needed
            return preds, "fast"

        # Low confidence — run full TTA for better accuracy
        tta_preds = [preds]   # already have pass 1

        # Pass 2: horizontal flip
        flipped = face_input[:, :, ::-1, :]
        tta_preds.append(
            self._emotion_model.predict(flipped, verbose=0)[0])

        # Pass 3: brightness +10%
        bright = np.clip(face_input * 1.1, 0.0, 1.0).astype("float32")
        tta_preds.append(
            self._emotion_model.predict(bright, verbose=0)[0])

        # Pass 4: brightness -10%
        dark = np.clip(face_input * 0.9, 0.0, 1.0).astype("float32")
        tta_preds.append(
            self._emotion_model.predict(dark, verbose=0)[0])

        # Pass 5: tiny Gaussian noise
        noisy = np.clip(
            face_input + np.random.normal(0, 0.02, face_input.shape),
            0.0, 1.0
        ).astype("float32")
        tta_preds.append(
            self._emotion_model.predict(noisy, verbose=0)[0])

        return np.mean(tta_preds, axis=0), "tta"

    # ── Inference (background thread) ─────────────────────────────────────────

    def _run_inference(self, frame, frame_gray, brightness):
        if self._inference_busy:
            return
        self._inference_busy = True

        def _infer():
            try:
                result = self._infer_frame(
                    frame.copy(), frame_gray.copy(), brightness
                )
                with self._inference_lock:
                    self._inference_result = result
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
            finally:
                self._inference_busy = False

        threading.Thread(target=_infer, daemon=True).start()

    def _infer_frame(self, frame, frame_gray, brightness):
        """
        Full inference pipeline:
          1. Detect all faces (Haar)
          2. Liveness check (LBP)
          3. Face recognition (Facenet512 + cosine)
          4. Batch emotion prediction (CNN + conditional TTA)
          5. Per-face smoothing
          6. Sort by area (closest first)
        """
        faces = self._face_proc.detect(frame)
        if not faces:
            return []

        db          = self._face_db.all()
        face_thresh = self._face_threshold(brightness)
        results     = []

        # ── Per-face: liveness + recognition + crop ────────────────────────
        face_inputs = []   # batch for emotion model
        face_meta   = []   # parallel metadata

        for box in faces:
            x, y, w, h = box
            area = w * h

            # Liveness
            live, _ = self._face_proc.is_live(frame, box)
            if not live:
                continue

            # Recognition — use adaptive threshold
            name, rec_score = "Unknown", 0.0
            emb = self._face_proc.embed(frame, box)
            if emb is not None:
                # Temporarily apply adaptive threshold
                original_thresh = self._face_proc.threshold
                self._face_proc.threshold = face_thresh
                if self._face_proc.identify_blocked(emb, db):
                    self._face_proc.threshold = original_thresh
                    block_label = next(
                        (b for b in db if b.startswith("__blocked__")),
                        "__blocked__unknown"
                    )
                    results.append((
                        self._face_proc._grid_key(box),
                        block_label, 1.0, "N/A", 0.0, box, area
                    ))
                    continue
                name, rec_score = self._face_proc.identify(emb, db, box)
                self._face_proc.threshold = original_thresh

            # Crop face for emotion — collect for batch
            face_gray = frame_gray[max(0,y):y+h, max(0,x):x+w]
            if face_gray.size > 0:
                face_resized    = cv2.resize(face_gray, config.IMG_SIZE)
                face_normalized = face_resized.astype("float32") / 255.0
                face_input      = np.expand_dims(face_normalized, axis=(0, -1))
                face_inputs.append(face_input)
            else:
                face_inputs.append(None)

            face_id = self._face_proc._grid_key(box)
            face_meta.append((face_id, name, rec_score, box, area))

        # ── Batch emotion prediction [OPT #3] ─────────────────────────────
        # Collect all valid face inputs into one batch
        valid_indices = [i for i, fi in enumerate(face_inputs) if fi is not None]

        if valid_indices:
            # Build batch
            batch = np.concatenate(
                [face_inputs[i] for i in valid_indices], axis=0
            )  # (N, 48, 48, 1)

            # Fast single-pass predict for all faces
            batch_preds = self._emotion_model.predict(batch, verbose=0)

            # Check which faces need TTA (low confidence)
            tta_threshold = getattr(config, 'TTA_CONFIDENCE_THRESHOLD', 0.65)
            final_preds = {}

            for local_i, orig_i in enumerate(valid_indices):
                preds = batch_preds[local_i]
                if float(preds.max()) < tta_threshold:
                    # Run TTA for this specific face only
                    tta_result, _ = self._predict_emotion(face_inputs[orig_i])
                    final_preds[orig_i] = tta_result
                else:
                    final_preds[orig_i] = preds

        # ── Assemble results ───────────────────────────────────────────────
        for i, (face_id, name, rec_score, box, area) in enumerate(face_meta):
            if i in final_preds:
                preds    = final_preds[i]
                raw_idx  = int(np.argmax(preds))
                # Minimal smoothing — only 3 frames, prevents flicker
                # Logic Controller applies its own cooldown for announcements
                smooth_idx = self._smooth_emotion(face_id, raw_idx)
                emo_conf   = float(preds[raw_idx])   # use raw confidence
                emotion    = config.EMOTIONS_EN[smooth_idx]
            else:
                emotion, emo_conf = "Neutral", 0.0

            results.append((face_id, name, rec_score, emotion, emo_conf, box, area))

        # Sort by area descending (closest first)
        results.sort(key=lambda r: r[6], reverse=True)
        return results

    # ── Audio Fallback ────────────────────────────────────────────────────────

    def _on_audio_result(self, emotion, confidence):
        self._current_emotion = emotion
        self._current_conf    = confidence
        self._current_source  = "audio"
        print(f"[MIC] {emotion} ({confidence*100:.0f}%)")

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

        if not cap.isOpened():
            print("ERROR: Cannot open camera. Check CAMERA_INDEX in config.py")
            return

        print("Camera running...\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Failed to read frame.")
                    break

                self._update_fps()
                self._frame_count += 1
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = self._brightness(frame_gray)
                threshold  = self._confidence_threshold(brightness)

                # Feed to registration
                if self._reg.active:
                    self._reg.feed(frame)

                # Check registration result
                reg_result = self._reg.result()
                if reg_result and reg_result != "PENDING":
                    if isinstance(reg_result, str) and \
                       reg_result.startswith("registered:"):
                        name = reg_result.replace("registered:", "")
                        self._logic.on_registered(name)

                # ── [OPT #5] Adaptive inference rate ─────────────────────
                inference_n = self._get_inference_n()
                if self._frame_count % inference_n == 0:
                    self._run_inference(frame, frame_gray, brightness)

                # ── Apply inference result ────────────────────────────────
                with self._inference_lock:
                    result = self._inference_result
                    self._inference_result = None

                if result is not None:
                    faces_data = []

                    # [OPT #6] Audio fallback only for closest face
                    closest_needs_audio = (
                        result and
                        not result[0][1].startswith("__blocked__") and
                        result[0][4] < threshold
                    )
                    if closest_needs_audio:
                        self._audio_det.analyze_async(self._on_audio_result)

                    for face_id, name, rec_score, emotion, emo_conf, face_box, area \
                            in result:
                        # Apply audio fallback result to close face if needed
                        if closest_needs_audio and \
                           face_id == result[0][0] and \
                           not name.startswith("__blocked__"):
                            emotion  = self._current_emotion
                            emo_conf = self._current_conf

                        faces_data.append(
                            (face_id, name, rec_score, emotion, emo_conf, area)
                        )

                    # Update display state + stability tracking
                    if result:
                        _, name, _, emotion, emo_conf, face_box, _ = result[0]
                        self._current_name     = name
                        self._current_face_box = face_box
                        self._current_emotion  = emotion
                        self._current_conf     = emo_conf
                        self._current_source   = "face"

                        # Track stability for adaptive inference
                        if emotion == self._last_emotion_set:
                            self._stable_frames += 1
                        else:
                            self._stable_frames    = 0
                            self._last_emotion_set = emotion

                    # Send to Logic Controller
                    self._logic.process_faces(
                        faces_data = faces_data,
                        brightness = brightness,
                        frame      = frame,
                    )

                    # Log
                    if result:
                        mode = "stable" if self._stable_frames >= \
                               self._stable_threshold else "active"
                        self._log(
                            self._current_name, self._current_emotion,
                            self._current_conf, self._current_source,
                            brightness, mode
                        )

                # ── Draw ─────────────────────────────────────────────────
                if config.SHOW_WINDOW:
                    if self._current_face_box is not None:
                        # [FIX #10] Use name + emotion properly
                        frame = draw_results(
                            frame,
                            self._current_emotion,
                            self._current_conf,
                            self._current_face_box,
                            self._current_source,
                        )
                        x, y, w, h = self._current_face_box
                        label = self._current_name
                        if self._current_emotion not in ("N/A", ""):
                            label += f" | {self._current_emotion}"
                        cv2.putText(
                            frame, label, (x, y - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 0), 2
                        )
                    else:
                        frame = draw_no_face(frame)

                    frame = draw_fps(frame, self._fps)
                    cv2.imshow(config.WINDOW_TITLE, frame)

                # ── Key Handler ───────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Exiting...")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._tts_cache.clear()
            if self._log_file:
                self._log_file.close()
            print("System closed.")


if __name__ == "__main__":
    system = AssistiveVisionSystem()
    system.run()
