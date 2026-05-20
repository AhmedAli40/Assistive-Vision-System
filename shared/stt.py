"""
shared/stt.py - Speech to Text
================================
Dual-language STT (Arabic + English).
Every command has a rich set of synonyms in both languages.

Online:  Google Speech (AR first or EN first depending on current language)
Offline: Vosk AR (if downloaded) + Vosk EN fallback
"""
import speech_recognition as sr
import logging
import time
import threading
import os

logger = logging.getLogger(__name__)

# ── Yes / No ─────────────────────────────────────────────────────────────────
YES = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "correct",
    "right", "do it", "go ahead", "confirm", "affirmative", "please",
    "أيوه", "ايوه", "نعم", "اه", "آه", "صح", "تمام", "موافق",
    "بالظبط", "اكيد", "طبعاً", "طبعا", "ماشي", "يلا",
}
NO = {
    "no", "nope", "nah", "skip", "cancel", "stop", "wrong",
    "negative", "ignore", "don't", "dont", "abort", "never mind",
    "لا", "لأ", "لأه", "الغ", "إلغاء", "مش", "بلاش", "غلط",
    "اوقف", "متعملش", "معلش",
}

# ── Offline Vosk commands ─────────────────────────────────────────────────────
OFFLINE_COMMANDS_EN = [
    "vision", "register", "save", "add", "record", "note",
    "block", "ban", "forbid", "restrict",
    "unblock", "allow", "unlock", "permit",
    "delete", "remove", "erase", "forget", "clear", "wipe",
    "list", "show", "names", "all",
    "who", "identify", "name",
    "quiet", "silence", "mute", "hush",
    "speak", "resume", "unmute", "talk",
    "stop", "halt", "pause", "enough",
    "switch", "arabic", "english",
    "yes", "no", "yeah", "nope", "okay", "cancel",
]
OFFLINE_COMMANDS_AR = [
    "فيجن", "فيجين", "بصر", "مساعد",
    "سجل", "احفظ", "اضف", "ضيف",
    "احظر", "حظر", "امنع",
    "الغ الحظر", "ارفع الحظر",
    "امسح", "احذف", "شيل",
    "قائمة", "الاسماء", "اعرض",
    "مين", "من", "عرفني",
    "اسكت", "سكوت", "هدوء",
    "اتكلم", "كمل", "استمر",
    "وقف", "بس",
    "عربي", "انجليزي",
    "ايوه", "اه", "نعم", "تمام",
    "لا", "لأ", "بلاش",
]

VOSK_MODEL_PATH_EN = "models/vosk-model"
VOSK_MODEL_PATH_AR = "models/vosk-model-ar"

# ── Command map — every synonym in AR + EN ────────────────────────────────────
COMMAND_MAP = {
    "register": [
        # English — all synonyms
        "register", "save", "add", "record", "memorize", "note", "store",
        "enroll", "sign up", "capture", "introduce", "include", "enter",
        "new person", "add person", "save person", "save face",
        # Arabic — all synonyms
        "سجل", "احفظ", "ضيف", "اضف", "أضف", "تسجيل", "حفظ",
        "إضافة", "ادخل", "خزن", "دخّل", "ثبت", "عرّف",
        "شخص جديد", "اضف شخص", "احفظ شخص",
    ],
    "unblock": [
        # English
        "unblock", "allow", "unlock", "permit", "approve",
        "whitelist", "unban", "restore", "reinstate",
        "remove block", "lift ban", "clear block",
        # Arabic
        "الغ الحظر", "ارفع الحظر", "افتح", "سماح",
        "إلغاء الحظر", "رفع الحظر", "رفع الحجب",
        "سمحله", "فك الحظر", "اسمح",
    ],
    "block": [
        # English
        "block", "ban", "blacklist", "forbid", "restrict",
        "deny", "reject", "exclude", "bar", "prohibit",
        "add to blacklist", "block person", "ban person",
        # Arabic
        "احظر", "حظر", "امنع", "منع", "بلوك",
        "اضف للقائمة السوداء", "ممنوع", "استبعد",
        "حجب", "اتخلص منه", "ابعده",
    ],
    "who": [
        # English
        "who", "identify", "name", "tell me", "who is", "who is this",
        "who is that", "what is the name", "do you know", "recognize",
        "who are you looking at", "what is his name", "what is her name",
        # Arabic
        "مين", "من", "اعرفني", "قول لي", "من هو", "من هي",
        "عرفني", "هو مين", "هي مين", "ايه اسمه", "ايه اسمها",
        "تعرفه", "تعرفها", "مين ده", "مين دي",
    ],
    "delete": [
        # English
        "delete", "remove", "erase", "forget", "clear", "wipe",
        "unregister", "deregister", "eliminate", "purge",
        "delete person", "remove person", "forget person",
        # Arabic
        "امسح", "احذف", "حذف", "مسح", "اشيل", "شيل",
        "ازيل", "إزالة", "الغ التسجيل", "مسح شخص",
        "احذف شخص", "شيله", "امسحه",
    ],
    "list": [
        # English
        "list", "show", "names", "who do you know", "show all",
        "everyone", "all names", "all people", "show names",
        "who is registered", "who is saved", "what names",
        "give me the list", "tell me the names",
        # Arabic
        "الاسماء", "الأسماء", "قائمة", "اعرض", "كل الناس",
        "مين عندك", "من عندك", "كل الأسماء", "اعرض الأسماء",
        "مين المسجلين", "إيه الأسماء", "ايه الاسماء",
    ],
    "quiet": [
        # English
        "quiet", "silence", "mute", "shut up", "stop talking",
        "be quiet", "hush", "shush", "enough", "no more talking",
        "stop speaking", "don't talk", "be silent",
        # Arabic
        "اسكت", "سكوت", "هدوء", "بلاش كلام", "متكلمش",
        "صمت", "كفاية", "بطل كلام", "مش عايزك تكلم",
        "وقف الكلام", "بلاش صوت",
    ],
    "speak": [
        # English
        "speak", "resume", "unmute", "talk", "continue",
        "start talking", "voice on", "talk again", "keep talking",
        "go on", "carry on", "keep going",
        # Arabic
        "اتكلم", "كمل", "رجع", "ارجع للكلام", "استمر",
        "شغل الكلام", "كمل كلام", "ارجع اتكلم",
        "ابدأ تتكلم", "شغل الصوت",
    ],
    "stop": [
        # English
        "stop", "halt", "pause", "cut it", "cut off",
        "stop now", "hold on", "wait",
        # Arabic
        "وقف", "وقف الكلام", "اوقف", "بس",
        "كفاية كده", "وقف دلوقتي", "استنى",
    ],
}

# ── Wake words ────────────────────────────────────────────────────────────────
WAKE_WORDS = [
    "vision", "hey vision", "ok vision", "visions",
    "فيجن", "فيجين", "بصر", "يا بصر", "مساعد", "يا مساعد",
]

# ── Switch language commands ─────────────────────────────────────────────────
SWITCH_TO_AR = [
    # English triggers
    "arabic", "switch to arabic", "change to arabic", "speak arabic",
    "arabic mode", "ar mode", "go arabic", "use arabic", "arabic please",
    "talk arabic", "speak in arabic", "change language arabic",
    # Arabic triggers
    "عربي", "غير للعربي", "غير لعربي", "تحويل للعربي",
    "كلم عربي", "اتكلم عربي", "غير اللغة للعربي",
    "غير للعربية", "اللغة العربية", "بالعربي",
    "اتكلم بالعربي", "كلمني عربي", "عايزك تكلم عربي",
    "حول للعربي", "غير اللغة عربي",
]
SWITCH_TO_EN = [
    # English triggers
    "english", "switch to english", "change to english", "speak english",
    "english mode", "en mode", "go english", "use english", "english please",
    "talk english", "speak in english", "change language english",
    # Arabic triggers
    "انجليزي", "إنجليزي", "غير للإنجليزي", "غير لانجليزي",
    "تحويل للانجليزي", "كلم إنجليزي", "اتكلم إنجليزي",
    "غير للإنجليزية", "اللغة الإنجليزية", "بالانجليزي",
    "اتكلم بالانجليزي", "كلمني انجليزي", "عايزك تكلم انجليزي",
    "حول للانجليزي", "غير اللغة انجليزي",
]

# ── Voice gender change commands ──────────────────────────────────────────────
VOICE_MALE_AR = [
    # Arabic triggers — change Arabic voice to male
    "صوت رجالي", "صوت ذكر", "صوت رجل", "غير للذكر", "صوت ولاد",
    "صوت راجل", "صوت مذكر", "بدل للذكر", "عايز صوت راجل",
    "غير الصوت رجالي", "صوت عربي رجالي", "صوت عربي ذكر",
    "خلي الصوت رجالي", "صوت دكر",
    # English triggers — change Arabic voice to male
    "male arabic", "arabic male", "man voice arabic",
    "arabic man", "male arab voice", "change arabic male",
    "arabic voice male",
]
VOICE_FEMALE_AR = [
    # Arabic triggers — change Arabic voice to female
    "صوت ستات", "صوت أنثى", "صوت بنت", "غير للأنثى", "صوت نسائي",
    "صوت مؤنث", "بدل للأنثى", "عايز صوت بنت", "صوت سيت",
    "غير الصوت نسائي", "صوت عربي نسائي", "صوت عربي أنثى",
    "خلي الصوت نسائي", "صوت انثى",
    # English triggers — change Arabic voice to female
    "female arabic", "arabic female", "woman voice arabic",
    "arabic woman", "female arab voice", "change arabic female",
    "arabic voice female",
]
VOICE_MALE_EN = [
    # English triggers — change English voice to male
    "male english", "english male", "man voice", "male voice english",
    "guy voice", "male voice", "change to male", "male please",
    "english man", "man english", "male english voice",
    "switch to male", "use male voice",
    # Arabic triggers — change English voice to male
    "صوت رجالي إنجليزي", "صوت ذكر إنجليزي", "صوت راجل إنجليزي",
    "انجليزي رجالي", "صوت انجليزي ذكر",
]
VOICE_FEMALE_EN = [
    # English triggers — change English voice to female
    "female english", "english female", "woman voice", "female voice english",
    "lady voice", "female voice", "change to female", "female please",
    "english woman", "woman english", "female english voice",
    "switch to female", "use female voice",
    # Arabic triggers — change English voice to female
    "صوت أنثى إنجليزي", "صوت ستات إنجليزي", "صوت بنت إنجليزي",
    "انجليزي نسائي", "صوت انجليزي أنثى",
]


def detect_voice_change(text: str):
    """
    Detect voice gender change commands.
    Returns (lang, gender) tuple or None.
    lang: 'ar' or 'en'
    gender: 'male' or 'female'
    """
    t = text.strip().lower()
    for kw in VOICE_MALE_AR:
        if kw in t:
            return ("ar", "male")
    for kw in VOICE_FEMALE_AR:
        if kw in t:
            return ("ar", "female")
    for kw in VOICE_MALE_EN:
        if kw in t:
            return ("en", "male")
    for kw in VOICE_FEMALE_EN:
        if kw in t:
            return ("en", "female")
    return None

# ── Emotion names in Arabic ───────────────────────────────────────────────────
EMOTIONS_AR = {
    "Angry":    "غاضب",
    "Disgust":  "مشمئز",
    "Fear":     "خائف",
    "Happy":    "سعيد",
    "Neutral":  "طبيعي",
    "Sad":      "حزين",
    "Surprise": "متفاجئ",
}


def match_command(text: str) -> str | None:
    """Match spoken text to a command. Checks unblock before block."""
    text = text.strip().lower()
    for cmd in ["unblock", "register", "block", "who", "delete",
                "list", "quiet", "speak", "stop"]:
        for kw in COMMAND_MAP[cmd]:
            if kw in text:
                return cmd
    return None


def detect_switch(text: str) -> str | None:
    """Returns 'ar', 'en', or None."""
    t = text.strip().lower()
    for kw in SWITCH_TO_AR:
        if kw in t:
            return "ar"
    for kw in SWITCH_TO_EN:
        if kw in t:
            return "en"
    return None


def is_wake_word(text: str) -> bool:
    t = text.strip().lower()
    for w in WAKE_WORDS:
        if w in t:
            return True
    return False


class STT:
    def __init__(self):
        self.r = sr.Recognizer()
        self.r.pause_threshold          = 1.5
        self.r.phrase_threshold         = 0.3
        self.r.non_speaking_duration    = 0.8
        self.r.dynamic_energy_threshold = False
        self.r.energy_threshold         = 400
        self._mic  = None
        self._lang = "en"

        self._online            = True
        self._last_online_check = 0

        self._vosk_ar_rec   = None
        self._vosk_en_rec   = None
        self._vosk_ar_ready = False
        self._vosk_en_ready = False
        self._init_vosk()

    def set_language(self, lang: str):
        self._lang = lang
        print(f"[STT] Language → {lang.upper()}")

    def _init_vosk(self):
        def _load():
            try:
                import config
                if not config.VOSK_ENABLED:
                    return
                from vosk import Model, KaldiRecognizer, SetLogLevel
                import json
                SetLogLevel(-1)

                if os.path.exists(VOSK_MODEL_PATH_AR):
                    print("[STT] Loading Arabic Vosk model...")
                    ar_model = Model(VOSK_MODEL_PATH_AR)
                    self._vosk_ar_rec   = KaldiRecognizer(
                        ar_model, 16000, json.dumps(OFFLINE_COMMANDS_AR)
                    )
                    self._vosk_ar_ready = True
                    print("[STT] Arabic Vosk ready")
                else:
                    print(f"[STT] Arabic Vosk not found — download:")
                    print("[STT] https://alphacephei.com/vosk/models/vosk-model-ar-mgb2-0.4.zip")

                if os.path.exists(VOSK_MODEL_PATH_EN):
                    print("[STT] Loading English Vosk model...")
                    en_model = Model(VOSK_MODEL_PATH_EN)
                    self._vosk_en_rec   = KaldiRecognizer(
                        en_model, 16000, json.dumps(OFFLINE_COMMANDS_EN)
                    )
                    self._vosk_en_ready = True
                    print("[STT] English Vosk ready")
            except Exception as e:
                print(f"[STT] Vosk init failed: {e}")
        threading.Thread(target=_load, daemon=True).start()

    def _check_online(self) -> bool:
        now = time.time()
        if now - self._last_online_check < 10.0:
            return self._online
        self._last_online_check = now
        try:
            import socket
            socket.setdefaulttimeout(2)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
            self._online = True
        except Exception:
            self._online = False
        return self._online

    def calibrate(self, duration: float = 2.0):
        print("[STT] Calibrating mic...")
        try:
            with sr.Microphone(device_index=self._mic) as src:
                self.r.adjust_for_ambient_noise(src, duration=duration)
            self.r.energy_threshold = max(300, min(1500, self.r.energy_threshold))
            print(f"[STT] Ready. Threshold = {self.r.energy_threshold:.0f}")
        except Exception as e:
            logger.warning(f"Calibrate: {e}")
            self.r.energy_threshold = 400
            print("[STT] Default threshold = 400")

    def _quick_recal(self):
        try:
            with sr.Microphone(device_index=self._mic) as src:
                self.r.adjust_for_ambient_noise(src, duration=0.5)
            self.r.energy_threshold = max(300, min(1500, self.r.energy_threshold))
        except Exception:
            pass

    def listen(self, timeout: float = 8.0,
               phrase_limit: float = 10.0,
               recal: bool = False) -> str | None:
        if recal:
            self._quick_recal()
        try:
            with sr.Microphone(device_index=self._mic) as src:
                print(f"[STT] Listening... (E={self.r.energy_threshold:.0f})")
                audio = self.r.listen(src, timeout=timeout,
                                      phrase_time_limit=phrase_limit)
        except sr.WaitTimeoutError:
            print("[STT] No speech detected.")
            return None
        except Exception as e:
            logger.warning(f"Mic: {e}")
            return None

        if self._check_online():
            if self._lang == "ar":
                result = self._recognize_google(audio, "ar-EG")
                if result: return result
                result = self._recognize_google(audio, "en-US")
            else:
                result = self._recognize_google(audio, "en-US")
                if result: return result
                result = self._recognize_google(audio, "ar-EG")
            if result: return result
            print("[STT] Google failed — trying Vosk")

        return self._recognize_vosk(audio)

    def _recognize_google(self, audio, lang: str) -> str | None:
        try:
            text = self.r.recognize_google(audio, language=lang)
            print(f"[STT] Google ({lang}): '{text}'")
            return text.strip()
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"[STT] Google error: {e}")
            self._online = False
            return None

    def _recognize_vosk(self, audio) -> str | None:
        import json
        wav = audio.get_wav_data(convert_rate=16000, convert_width=2)

        if self._lang == "ar" and self._vosk_ar_ready:
            try:
                self._vosk_ar_rec.AcceptWaveform(wav)
                text = json.loads(self._vosk_ar_rec.Result()).get("text", "").strip()
                if text:
                    print(f"[STT] Vosk AR: '{text}'")
                    return text
            except Exception as e:
                logger.warning(f"Vosk AR: {e}")

        if self._vosk_en_ready:
            try:
                self._vosk_en_rec.AcceptWaveform(wav)
                text = json.loads(self._vosk_en_rec.Result()).get("text", "").strip()
                if text:
                    print(f"[STT] Vosk EN: '{text}'")
                    return text.lower()
            except Exception as e:
                logger.warning(f"Vosk EN: {e}")

        print("[STT] Vosk: no result")
        return None

    def yes_no(self, tries: int = 4, timeout: float = 8.0) -> bool | None:
        for i in range(1, tries + 1):
            text = self.listen(timeout=timeout, phrase_limit=4.0, recal=(i > 1))
            if text is None: continue
            tl = text.lower()
            if any(w in tl for w in YES):
                print("[STT] YES"); return True
            if any(w in tl for w in NO):
                print("[STT] NO");  return False
            print(f"[STT] '{text}' not yes/no — {i}/{tries}")
        return None

    def get_name(self, tries: int = 3, timeout: float = 9.0) -> str | None:
        for i in range(1, tries + 1):
            text = self.listen(timeout=timeout, phrase_limit=10.0, recal=(i > 1))
            if not text: continue
            for filler in ["اسمه", "اسمها", "اسمي", "اسم", "هو", "هي",
                           "ده", "دي", "is", "his name is", "her name is",
                           "name is", "call him", "call her"]:
                text = text.replace(filler, "").strip()
            words = text.strip().split()
            name  = " ".join(words[:2]).title()
            if len(name) >= 2:
                print(f"[STT] Name: '{name}'")
                return name
            print(f"[STT] Name too short — {i}/{tries}")
        return None
