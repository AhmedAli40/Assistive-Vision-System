# Project Audit — Assistive Vision System

## 1. Overview
- Repository name: Assistive Vision System
- Purpose: دمج التعرف على الوجوه مع قراءة العواطف والتحكم الصوتي باللغة العربية والإنجليزية.
- الرئيسية: `main.py`.
- الشيفرة الحالية تعتمد على:
  - `tensorflow`
  - `opencv-python`
  - `deepface`
  - `speech_recognition`
  - `vosk`
  - `edge-tts` / `pygame` / `gtts` / `pyttsx3`

## 2. المستودع وبنيته
### الجذر
- `main.py`
- `logic_controller.py`
- `config.py`
- `README.md`
- `install.bat`, `run.bat`
- `project_complete_log.md`
- `face_data.pkl`
- مجلدات:
  - `shared/`
  - `face/`
  - `emotion/`
  - `models/`
  - `cnn_v3/`, `cnn_v6/`
  - `logs/`

### مجلد `shared`
- `shared/stt.py` — وحدة STT عامة مع Google و Vosk وفحص الكلمات.
- `shared/tts.py` — وحدة TTS مع Edge TTS وبدائل.

### مجلد `face`
- `face/face_db.py` — قاعدة بيانات الوجوه وحفظها/تحميلها.
- `face/face_processor.py` — كشف وجوه، حساب embeddings، مطابقة، تصفية محظورين.
- `face/registration.py` — تدفق تسجيل/حذف/حظر/رفع حظر الأشخاص.

### مجلد `emotion`
- `emotion/face_detector.py` — كشف عواطف الوجه باستخدام MTCNN أو Haar Cascade.
- `emotion/audio_detector.py` — فشل صوتي لتصنيف العاطفة من الصوت.
- `emotion/display.py` — عرض النتائج على الشاشة.

## 3. الموديلات والبيانات
- `models/cnn_v3_final.h5`
- `models/cnn_v6_final.h5`
- `models/emotion_fixed.h5`
- `models/vosk-model/`
- `models/vosk-model-ar/`
- `cnn_v3/cnn_v3.tflite`
- `cnn_v6/cnn_v6.tflite`

## 4. النقاط الحرجة في الكود
### 4.1 `main.py`
- يقوم بتحميل نموذج العاطفة قبل استدعاء DeepFace.
- يستخدم `shared.tts.TTS` و `shared.stt.STT`.
- يستورد `FaceDB`, `FaceProcessor`, `RegFlow` من مجلد `face`.
- يعتمد على `emotion.display` لعرض النتائج.
- يحدد `MODEL_PATH` في `config.py` كنقطة حساسة؛ إذا لم يوجد الملف ينهي التشغيل.

### 4.2 `logic_controller.py`
- هدفه: مراقبة الوجه، بناء النصوص، التعامل مع الأوامر الصوتية، التبديل بين العربية والإنجليزية.
- يوجد استيراد حساس:
  ```python
  try:
      from shared.stt import match_command, detect_switch, is_wake_word, EMOTIONS_AR
  except ImportError:
      from stt import match_command, detect_switch, detect_voice_change, is_wake_word, EMOTIONS_AR
  ```
- الملاحظة المهمة:
  - في هذه البنية يوجد فقط `shared/stt.py`، ولا يوجد ملف `stt.py` في جذر المشروع.
  - لذلك إذا فشل استيراد `shared.stt` لأي سبب، فالفرع الاحتياطي `from stt import ...` سيؤدي إلى نفس خطأ "No module named 'stt'".
  - هذا سلوك غير مستقر وقد يسبب الأخطاء التي ظهرت سابقاً.

### 4.3 `shared/stt.py`
- يحتوي على:
  - تعاريف أوامر صوتية عربية وإنجليزية.
  - `detect_switch(text)` لتحويل اللغة.
  - `detect_voice_change(text)` لتغيير جنس الصوت.
  - `is_wake_word(text)` لكلمة التنبيه.
  - كائن `STT` يستخدم Google وVosk.

### 4.4 `shared/tts.py`
- يدعم عدة محركات TTS:
  - `edge-tts` (الافتراضي)
  - `gTTS`
  - `win32com` SAPI
  - `pyttsx3`
- يتيح تغيير اللغة والجنس.

## 5. الملاحظات الفنية العامة
- لا توجد `requirements.txt` واضحة في الجذر، لكن `README.md` يوجّه إلى `install.bat`.
- `README.md` يذكر تحميل النماذج يدوياً إلى `models/`.
- هناك استخدام واضح للملف `face_data.pkl` كسجل لقاعدة بيانات الوجوه.
- مشروعك يحتوي على بعض ملفات السجل والمجلدات المساعدة (`logs/`, `__pycache__/`).

## 6. نقاط التحسين المقترحة
1. إزالة أو إصلاح فرع الاستيراد الاحتياطي في `logic_controller.py`.
2. إضافة `requirements.txt` أو `pyproject.toml` لتوضيح التبعيات.
3. توثيق الأوامر الصوتية بالكامل في `README.md` إذا كنت تريد دعمها بشكل أكبر.
4. التأكد من وجود `models/cnn_v3_final.h5` و `models/vosk-model/...` قبل التشغيل.
5. وضع فحص `shared` و `sys.path` بشكل صريح في `main.py` إن كان المشروع يُشغل من أماكن مختلفة.

## 7. الحالة الحالية
- قمت بفحص الكود دون تعديل أي ملف من ملفات المشروع.
- هذه الوثيقة مجرد تقييم وتمثيل للحالة الحالية للمشروع.

## 8. توصيات سريعة
- إذا ظهرت مشكلة "No module named 'stt'" مجدداً، فابدأ بفحص السطر الموجود في `logic_controller.py` كما هو موضح أعلاه.
- تأكد من تشغيل `run.bat` من مجلد المشروع نفسه أو من بيئة Python حيث يكون جذر المشروع في `sys.path`.
- استخدم `install.bat` لتثبيت التبعيات قبل التشغيل.
