"""
face_processor.py — v6 Final
==============================
المشكلة في كل النسخ السابقة:
  DeepFace.represent() كانت بتتستدعى كل frame لكل وجه
  = 3 وجوه × 120ms = 360ms تأخير كل frame

الحل في v6:
  1. Detection:  Haar Cascade         (~5ms)
  2. Crop:       نقطع منطقة الوجه
  3. Embedding:  DeepFace مرة واحدة  (~120ms) في thread منفصل
  4. Matching:   Cosine على RAM       (~1ms)  فوري
  
  النتيجة: الكاميرا شغّالة بـ 30fps
           الـ recognition بيحصل كل 500ms تقريباً بدل كل frame
           
Liveness: LBP texture (~2ms)
Voting:   7 frames بالأغلبية لكل وجه منفصل
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# ── DeepFace ──────────────────────────────────────────────────────────────────
try:
    from deepface import DeepFace
    DF_OK = True
    print("[FaceProc] ✅ DeepFace loaded")
except ImportError:
    DF_OK = False
    print("[FaceProc] ⚠️  DeepFace not found!")

# ── LBP ───────────────────────────────────────────────────────────────────────
try:
    from skimage.feature import local_binary_pattern as _lbp
    LBP_OK = True
except ImportError:
    LBP_OK = False

# ── Config ────────────────────────────────────────────────────────────────────
HAAR_XML      = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MODEL         = "Facenet512"
BACKEND       = "opencv"
THRESHOLD     = 0.50    # cosine distance — أقل = أشبه
LBP_THRESHOLD = 18.0   # liveness
LBP_R, LBP_P  = 3, 24


class FaceProcessor:
    def __init__(self, threshold: float = None):
        self._cascade  = cv2.CascadeClassifier(HAAR_XML)
        if self._cascade.empty():
            raise RuntimeError("Haar Cascade not found")

        self.threshold = threshold or THRESHOLD
        self._df_ok    = DF_OK

        if DF_OK:
            self._warmup()

        if LBP_OK:
            print(f"[FaceProc] Liveness ON (LBP>{LBP_THRESHOLD})")

        print(f"[FaceProc] Threshold={self.threshold}")

        # Voting منفصل لكل موقع وجه في الإطار
        self._votes : dict = {}   # { grid_key: [name, ...] }
        self._vsize = 7

        # Pre-computed gamma LUT (1/1.5 gamma for low-light boost)
        # Computed once here instead of every frame
        self._gamma_lut = np.array(
            [(i / 255.0) ** (1.0 / 1.5) * 255 for i in range(256)],
            dtype=np.uint8
        )

    def _warmup(self):
        print("[FaceProc] Loading Facenet512... (first run ~30s)")
        try:
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            DeepFace.represent(img_path=dummy, model_name=MODEL,
                               enforce_detection=False, detector_backend=BACKEND)
            print("[FaceProc] ✅ Model ready")
        except Exception as e:
            logger.warning(f"Warmup: {e}")

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Tuple]:
        """Haar detection — سريع ~5ms"""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gamma correction — uses pre-computed LUT (computed once in __init__)
        g = cv2.LUT(g, self._gamma_lut)
        f = self._cascade.detectMultiScale(
            g, scaleFactor=1.1, minNeighbors=4,
            minSize=(45, 45)   # أصغر حجم للكشف عن الوجوه البعيدة
        )
        return [tuple(x) for x in f] if len(f) else []

    # ── Liveness ──────────────────────────────────────────────────────────────

    def is_live(self, frame: np.ndarray, box: Tuple) -> Tuple[bool, float]:
        if not LBP_OK: return True, 99.0
        x, y, w, h = box
        p   = int(min(w,h)*0.1)
        roi = frame[max(0,y-p):min(frame.shape[0],y+h+p),
                    max(0,x-p):min(frame.shape[1],x+w+p)]
        if roi.size == 0: return True, 0.0
        g   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi
        g   = cv2.resize(g, (64, 64))
        lb  = _lbp(g, LBP_P, LBP_R, method="uniform")
        hist, _ = np.histogram(lb.ravel(), bins=LBP_P+2,
                               range=(0,LBP_P+2), density=True)
        var = float(np.var(hist)*10000)
        return var >= LBP_THRESHOLD, var

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, frame: np.ndarray, box: Tuple) -> Optional[np.ndarray]:
        """
        يستخرج الـ embedding من الوجه.
        بيتستدعى من الـ inference thread — مش من الـ main thread.
        """
        if not self._df_ok: return self._embed_pixel(frame, box)
        return self._embed_df(frame, box)

    def _embed_df(self, frame, box) -> Optional[np.ndarray]:
        x, y, w, h = box
        # margin لتحسين الدقة
        pad = max(10, int(min(w,h)*0.15))
        roi = frame[max(0,y-pad):min(frame.shape[0],y+h+pad),
                    max(0,x-pad):min(frame.shape[1],x+w+pad)]
        if roi.size == 0: return None
        try:
            r = DeepFace.represent(
                img_path          = roi,
                model_name        = MODEL,
                enforce_detection = False,
                detector_backend  = BACKEND,
            )
            if r:
                v = np.array(r[0]["embedding"], dtype=np.float32)
                n = np.linalg.norm(v)
                return v/n if n > 0 else None
        except Exception as e:
            logger.debug(f"embed: {e}")
        return None

    def _embed_pixel(self, frame, box) -> Optional[np.ndarray]:
        """Fallback لو DeepFace مش موجود"""
        x, y, w, h = box
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0: return None
        g   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi
        g   = cv2.createCLAHE(2.0,(8,8)).apply(g)
        v   = cv2.resize(g,(128,128)).astype(np.float32).flatten()
        n   = np.linalg.norm(v)
        return v/n if n > 0 else None

    # ── Identify ──────────────────────────────────────────────────────────────

    def identify(self, emb: np.ndarray, db: dict,
                 box: Tuple = None) -> Tuple[str, float]:
        """
        يطابق الـ embedding مع قاعدة البيانات.
        سريع جداً ~1ms لأنه مجرد cosine على vectors في الـ RAM.
        """
        if not db or emb is None: return "Unknown", 0.0

        best, dist = "Unknown", float("inf")
        for name, rec in db.items():
            if not rec.embeddings or name.startswith("__blocked__"):
                continue
            stored = np.array(rec.embeddings)      # (N, 512)
            dists  = 1.0 - (stored @ emb)          # cosine distance
            top5   = sorted(dists)[:min(5, len(dists))]
            # Weighted average — closer matches have more influence
            weights = [1.0 / (d + 1e-6) for d in top5]
            w_sum   = sum(weights)
            avg     = float(sum(d * w / w_sum for d, w in zip(top5, weights)))
            if avg < dist:
                dist, best = avg, name

        if dist <= self.threshold:
            raw_name = best
        else:
            raw_name = "Unknown"

        score = round(1.0 - dist, 3)

        # Voting منفصل لكل موقع وجه
        key = self._grid_key(box)
        if key not in self._votes:
            self._votes[key] = []
        buf = self._votes[key]
        buf.append(raw_name)
        if len(buf) > self._vsize: buf.pop(0)

        return self._vote(buf), score

    def identify_blocked(self, emb: np.ndarray, db: dict) -> bool:
        """
        هل الوجه ده محظور؟
        بيستخدم threshold أضيق (0.35) عشان يتأكد إن الشخص ده هو هو المبلوك
        ومش أي شخص عنده شبه بعيد بالمبلوك
        """
        BLOCK_THRESHOLD = 0.35   # أضيق من الـ recognition threshold
        if not db or emb is None: return False
        for name, rec in db.items():
            if not name.startswith("__blocked__"): continue
            if not rec.embeddings: continue
            stored = np.array(rec.embeddings)
            dists  = 1.0 - (stored @ emb)
            # نأخذ أحسن 3 نتائج ومعدلهم للدقة
            top3   = sorted(dists)[:min(3, len(dists))]
            avg    = float(np.mean(top3))
            if avg <= BLOCK_THRESHOLD:
                return True
        return False

    # ── Voting ────────────────────────────────────────────────────────────────

    @staticmethod
    def _vote(buf: list) -> str:
        if not buf: return "Unknown"
        c = Counter(buf).most_common(1)[0]
        return c[0] if c[1]/len(buf) >= 0.55 else "Unknown"

    @staticmethod
    def _grid_key(box) -> str:
        if not box: return "0_0"
        x, y, w, h = box
        # Finer grid (80px cells) — avoids merging two nearby faces
        return f"{(x+w//2)//80}_{(y+h//2)//80}"

    def reset(self):
        self._votes.clear()

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, frame, box, label, score, lv=None):
        x, y, w, h = box
        if label == "Blocked":      col = (100,100,100)
        elif label == "Unknown":    col = (0,0,220)
        elif label == "Screen":     col = (0,165,255)
        else:                       col = (0,220,0)
        cv2.rectangle(frame, (x,y), (x+w,y+h), col, 2)
        txt = f"{label} ({score:.2f})"
        if lv is not None: txt += f" L:{lv:.0f}"
        cv2.putText(frame, txt, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    @staticmethod
    def _cos(a, b) -> float:
        d = np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.dot(a,b)/d) if d > 0 else 0.0
