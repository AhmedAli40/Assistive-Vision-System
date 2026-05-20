"""
logic_controller.py - Brain of the Assistive Vision System
===========================================================
- Multi-face support (closest first)
- Switchable EN/AR language at runtime
- Concise Arabic phrases
- Varied response templates
"""

import time
import random
import threading
import logging

import config
try:
    from shared.stt import (match_command, detect_switch,
                             detect_voice_change, is_wake_word, EMOTIONS_AR)
except ImportError:
    from stt import (match_command, detect_switch,
                     detect_voice_change, is_wake_word, EMOTIONS_AR)

logger = logging.getLogger(__name__)

# ── Emotion groups — big change = crossing group boundary → announce fast ─────
_POSITIVE  = {"Happy", "Surprise"}
_NEGATIVE  = {"Angry", "Disgust", "Fear", "Sad"}
_NEUTRAL   = {"Neutral"}

def _is_big_change(old_emo: str, new_emo: str) -> bool:
    """Returns True if emotion crossed a group boundary (e.g. Neutral→Happy)."""
    def _group(e):
        if e in _POSITIVE: return "pos"
        if e in _NEGATIVE: return "neg"
        return "neu"
    return _group(old_emo) != _group(new_emo)

# ── Name transliteration — English names → Arabic pronunciation ───────────────
# يضاف اسم جديد تلقائياً لو مش موجود في القاموس

NAME_AR = {
    # أسماء شائعة عربية بنطق إنجليزي
    "ahmed":    "أحمد",
    "mohamed":  "محمد",
    "mohammed": "محمد",
    "muhammad": "محمد",
    "ali":      "علي",
    "omar":     "عمر",
    "sara":     "سارة",
    "sarah":    "سارة",
    "mona":     "منى",
    "nour":     "نور",
    "noura":    "نورة",
    "youssef":  "يوسف",
    "yousef":   "يوسف",
    "joseph":   "يوسف",
    "karim":    "كريم",
    "kareem":   "كريم",
    "layla":    "ليلى",
    "leila":    "ليلى",
    "hana":     "هنا",
    "hanna":    "هنا",
    "mariam":   "مريم",
    "maryam":   "مريم",
    "mary":     "ماري",
    "fatima":   "فاطمة",
    "fatime":   "فاطمة",
    "aisha":    "عائشة",
    "aysha":    "عائشة",
    "hassan":   "حسن",
    "hussein":  "حسين",
    "hosein":   "حسين",
    "tarek":    "طارق",
    "tariq":    "طارق",
    "nadia":    "نادية",
    "rania":    "رانيا",
    "rana":     "رنا",
    "dina":     "دينا",
    "heba":     "هبة",
    "amr":      "عمرو",
    "amro":     "عمرو",
    "adel":     "عادل",
    "walid":    "وليد",
    "khaled":   "خالد",
    "khalid":   "خالد",
    "sameh":    "سامح",
    "sami":     "سامي",
    "wael":     "وائل",
    "sherif":   "شريف",
    "sherief":  "شريف",
    "ismail":   "إسماعيل",
    "osama":    "أسامة",
    "usama":    "أسامة",
    "mahmoud":  "محمود",
    "mostafa":  "مصطفى",
    "mustafa":  "مصطفى",
    "yara":     "يارا",
    "yasmin":   "ياسمين",
    "jasmine":  "ياسمين",
    "manar":    "منار",
    "mai":      "ماي",
    "may":      "ماي",
    # أسماء أجنبية شائعة
    "john":     "جون",
    "james":    "جيمس",
    "michael":  "مايكل",
    "david":    "ديفيد",
    "chris":    "كريس",
    "daniel":   "دانيال",
    "adam":     "آدم",
    "peter":    "بيتر",
    "mark":     "مارك",
    "paul":     "بول",
    "george":   "جورج",
    "anna":     "آنا",
    "emma":     "إيما",
    "olivia":   "أوليفيا",
    "sophia":   "صوفيا",
    "lisa":     "ليزا",
    "kate":     "كيت",
    "emily":    "إيميلي",
    "jessica":  "جيسيكا",
    "linda":    "ليندا",
}

def _arabic_name(name: str) -> str:
    """
    Convert English name to Arabic pronunciation.
    If not in dictionary, returns the original name
    so SAPI/EdgeTTS can attempt to pronounce it.
    """
    key = name.strip().lower()
    # Check full name first, then first word only
    if key in NAME_AR:
        return NAME_AR[key]
    first = key.split()[0] if " " in key else key
    if first in NAME_AR:
        rest = name.split(" ", 1)[1] if " " in name else ""
        return NAME_AR[first] + (" " + rest if rest else "")
    return name   # fallback: original name

# ── Emotion phrase builders ───────────────────────────────────────────────────

def _en_emotion(name: str, emotion: str, certain: bool = True) -> str:
    if certain:
        return random.choice([
            f"{name} looks {emotion}",
            f"{name} seems {emotion}",
            f"{name} appears {emotion}",
        ])
    return random.choice([
        f"Might be {name}, looks {emotion}",
        f"I think that's {name}, {emotion}",
    ])

def _en_unknown(emotion: str) -> str:
    return random.choice([
        f"Unknown person, {emotion}",
        f"Don't recognize them, {emotion}",
        f"Unknown face, {emotion}",
    ])

def _ar_emotion(name: str, emotion: str, certain: bool = True) -> str:
    ar  = EMOTIONS_AR.get(emotion, emotion)
    ar_name = _arabic_name(name)
    if certain:
        return random.choice([
            f"{ar_name} يبدو {ar}",
            f"{ar_name} يبدو عليه {ar}",
            f"{ar_name} حالته {ar}",
        ])
    return random.choice([
        f"ممكن يكون {ar_name}، ويبدو {ar}",
        f"أظن إنه {ar_name}، وهو {ar}",
    ])

def _ar_unknown(emotion: str) -> str:
    ar = EMOTIONS_AR.get(emotion, emotion)
    return random.choice([
        f"شخص مجهول يبدو {ar}",
        f"وجه غير معروف، يبدو {ar}",
    ])

# ── Static phrases ────────────────────────────────────────────────────────────

PHRASES = {
    "en": {
        "yes":              "Yes?",
        "no_speech":        "Didn't catch that.",
        "no_person":        "No one detected.",
        "cmd_not_found":    "Unknown command.",
        "no_registered":    "No one registered.",
        "quiet_on":         "Quiet.",
        "quiet_off":        "Resuming.",
        "low_light":        "Low light, using audio.",
        "switched_to_ar":   "Switched to Arabic.",
        "switched_to_en":   "Already in English.",
        "already_reg":      lambda n: f"{n} already registered.",
        "cannot_block_reg": lambda n: f"Can't block {n}, use delete.",
        "who_unknown":      lambda e: f"Unknown, {e}.",
        "who_known":        lambda n, e: f"{n}, {e}.",
        "reg_list":         lambda ns: f"Registered: {', '.join(ns)}.",
        "voice_changed":    lambda g: f"Voice changed to {'male' if g == 'male' else 'female'}.",
        "voice_already":    lambda g: f"Already using {'male' if g == 'male' else 'female'} voice.",
    },
    "ar": {
        "yes":              "نعم؟",
        "no_speech":        "لم أسمع شيئاً",
        "no_person":        "لا يوجد أحد أمامي",
        "cmd_not_found":    "الأمر غير معروف",
        "no_registered":    "لا توجد أسماء مسجلة",
        "quiet_on":         "حسناً، سأصمت",
        "quiet_off":        "حسناً، سأكمل الكلام",
        "low_light":        "الإضاءة ضعيفة، سأعتمد على الصوت",
        "switched_to_ar":   "تم التحويل إلى العربية",
        "switched_to_en":   "تم التحويل إلى الإنجليزية",
        "already_reg":      lambda n: f"{_arabic_name(n)} مسجل بالفعل",
        "cannot_block_reg": lambda n: f"لا يمكن حظر {_arabic_name(n)}، استخدم أمر الحذف",
        "who_unknown":      lambda e: f"شخص غير معروف، يبدو {EMOTIONS_AR.get(e, e)}",
        "who_known":        lambda n, e: f"هذا {_arabic_name(n)}، يبدو {EMOTIONS_AR.get(e, e)}",
        "reg_list":         lambda ns: f"الأسماء المسجلة هي: {', '.join(ns)}",
        "voice_changed":    lambda g: f"تم التغيير إلى صوت {'ذكر' if g == 'male' else 'أنثى'}.",
        "voice_already":    lambda g: f"الصوت الحالي {'ذكر' if g == 'male' else 'أنثى'} بالفعل.",
    },
}


class LogicController:

    def __init__(self, tts, stt, reg_flow, face_processor, face_db):
        self.tts  = tts
        self.stt  = stt
        self.reg  = reg_flow
        self.proc = face_processor
        self.db   = face_db

        self._lang = config.LANGUAGE

        self._last_announced  = {}
        self._last_seen       = {}
        self._last_emotion    = {}

        self._current_name    = None
        self._current_emotion = None

        self._announce_queue  = []
        self._announcing      = False
        self._low_light_warned = False
        self._processing_command = False

        threading.Timer(5.0, self._start_command_listener).start()
        print(f"      Logic Controller ready. Language: {self._lang.upper()}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _p(self, key: str, *args) -> str:
        val = PHRASES[self._lang][key]
        return val(*args) if callable(val) else val

    def _say(self, key: str, *args):
        self.tts.say_wait(self._p(key, *args))

    def _build_emotion_msg(self, name, emotion, certain):
        if self._lang == "ar":
            return _ar_emotion(name, emotion, certain)
        return _en_emotion(name, emotion, certain)

    def _build_unknown_msg(self, emotion):
        if self._lang == "ar":
            return _ar_unknown(emotion)
        return _en_unknown(emotion)

    def _switch_language(self, lang: str):
        old = self._lang
        self._lang = lang
        config.LANGUAGE = lang
        self.tts.set_language(lang)
        self.stt.set_language(lang)
        if lang == "ar":
            self.tts.say_wait(PHRASES["ar"]["switched_to_ar"])
        else:
            self.tts.say_wait(PHRASES["en"]["switched_to_en" if old == "en" else "switched_to_en"])
        print(f"[LANG] → {lang.upper()}")

    def _change_voice(self, lang: str, gender: str):
        """Change voice gender for a language and confirm to user."""
        current_gender = self.tts.get_voice_gender(lang)
        if current_gender == gender:
            self.tts.say_wait(self._p("voice_already", gender))
            return
        self.tts.set_voice(lang, gender)
        # Confirm in the language being changed
        if lang == self._lang:
            self.tts.say_wait(self._p("voice_changed", gender))
        else:
            # Confirm in current language
            self.tts.say_wait(self._p("voice_changed", gender))
        print(f"[VOICE] {lang.upper()} → {gender}")

    # ── Queue ─────────────────────────────────────────────────────────────────

    def _queue_announcement(self, msg: str):
        self._announce_queue.append(msg)
        if not self._announcing:
            threading.Thread(target=self._flush_queue, daemon=True).start()

    def _flush_queue(self):
        self._announcing = True
        while self._announce_queue:
            if self.tts.busy():
                time.sleep(0.2)
                continue
            msg = self._announce_queue.pop(0)
            self.tts.say(msg)
            time.sleep(0.3)
            self.tts.wait(timeout=10.0)
        self._announcing = False

    # ── Command listener ──────────────────────────────────────────────────────

    def _start_command_listener(self):
        def _loop():
            while True:
                try:
                    if self.reg.active or self.tts.busy() or self._processing_command:
                        time.sleep(0.3)
                        continue

                    text = self.stt.listen(timeout=4.0, phrase_limit=3.0)
                    if not text or not is_wake_word(text):
                        continue

                    self._processing_command = True
                    self._announce_queue.clear()
                    self.tts.say_wait(PHRASES[self._lang]["yes"], pause=0.5)

                    command = self.stt.listen(timeout=6.0, phrase_limit=5.0)
                    if not command:
                        self.tts.say_wait(PHRASES[self._lang]["no_speech"])
                    else:
                        self._handle_command(command)

                except Exception as e:
                    logger.debug(f"Listener: {e}")
                    time.sleep(0.5)
                finally:
                    self._processing_command = False

        threading.Thread(target=_loop, daemon=True).start()

    # ── Command handler ───────────────────────────────────────────────────────

    def _handle_command(self, text: str):

        # Language switch — highest priority
        sw = detect_switch(text)
        if sw is not None:
            self._switch_language(sw)
            return

        # Voice gender change — second priority
        vc = detect_voice_change(text)
        if vc is not None:
            self._change_voice(vc[0], vc[1])
            return

        cmd = match_command(text)

        if cmd == "register":
            if self._current_name is None:
                self._say("no_person")
            elif self._current_name != "Unknown":
                self._say("already_reg", self._current_name)
            else:
                self.reg.start_register()
            return

        if cmd == "unblock":
            self.reg.start_unblock()
            return

        if cmd == "block":
            if self._current_name is None:
                self._say("no_person")
            elif self._current_name != "Unknown":
                self._say("cannot_block_reg", self._current_name)
            else:
                self.reg.start_block()
            return

        if cmd == "who":
            e = self._current_emotion or "Neutral"
            if self._current_name in (None, "Unknown"):
                self.tts.say_wait(self._p("who_unknown", e))
            else:
                self.tts.say_wait(self._p("who_known", self._current_name, e))
            return

        if cmd == "delete":
            self.reg.start_delete()
            return

        if cmd == "list":
            names = [n for n in self.db.names()
                     if not n.startswith("__blocked__")]
            if names:
                self.tts.say_wait(self._p("reg_list", names))
            else:
                self._say("no_registered")
            return

        if cmd == "quiet":
            self._announce_queue.clear()
            self.tts.set_quiet(True)
            self.tts.say_wait(PHRASES[self._lang]["quiet_on"])
            return

        if cmd == "speak":
            self.tts.set_quiet(False)
            self.tts.say_wait(PHRASES[self._lang]["quiet_off"])
            return

        if cmd == "stop":
            self._announce_queue.clear()
            self.tts.stop()
            return

        self.tts.say_wait(PHRASES[self._lang]["cmd_not_found"])

    # ── Process faces ─────────────────────────────────────────────────────────

    def process_faces(self, faces_data: list, brightness: float, frame=None) -> str:
        now = time.time()

        if frame is not None and self.reg.active:
            self.reg.feed(frame)

        if brightness < config.BRIGHTNESS_THRESHOLD and not self._low_light_warned:
            self._low_light_warned = True
            self._queue_announcement(PHRASES[self._lang]["low_light"])

        if not faces_data:
            self._current_name    = None
            self._current_emotion = None
            return "no_face"

        self._current_name    = faces_data[0][1]
        self._current_emotion = faces_data[0][3]

        for face_id, name, rec_score, emotion, emo_conf, box_area in faces_data:
            if name.startswith("__blocked__"):
                continue
            if name != "Unknown":
                self._process_known(name, rec_score, emotion, now)
            else:
                self._process_unknown(face_id, emotion, now)

        return "processed"

    def process(self, face_id, name, rec_score, emotion,
                emo_conf, brightness, frame=None):
        return self.process_faces(
            [(face_id, name, rec_score, emotion, emo_conf, 1)],
            brightness, frame
        )

    def _process_known(self, name, score, emotion, now):
        last_seen     = self._last_seen.get(name, 0)
        just_returned = (now - last_seen) > config.UNKNOWN_REASK_TIMEOUT
        self._last_seen[name] = now

        last_ann  = self._last_announced.get(name, 0)
        last_emo  = self._last_emotion.get(name, "")
        emo_changed = (emotion != last_emo)

        if not emo_changed and not just_returned:
            return  # nothing changed — skip

        # Smart cooldown:
        # Big change (Neutral→Happy) → 1s cooldown — announce fast
        # Same group change or repeat → normal cooldown (config)
        if emo_changed and _is_big_change(last_emo, emotion):
            cooldown = 1.0   # fast response for big changes
        else:
            cooldown = config.TTS_COOLDOWN_SEC

        cooldown_ok = (now - last_ann) >= cooldown

        if just_returned or (emo_changed and cooldown_ok):
            self._last_announced[name] = now
            self._last_emotion[name]   = emotion
            self._queue_announcement(
                self._build_emotion_msg(name, emotion, certain=(score >= 0.75))
            )

    def _process_unknown(self, face_id, emotion, now):
        key       = f"unknown_{face_id}"
        last_ann  = self._last_announced.get(key, 0)
        last_emo  = self._last_emotion.get(key, "")
        emo_changed = (emotion != last_emo)

        if not emo_changed:
            return

        # Smart cooldown — same logic as known person
        if _is_big_change(last_emo, emotion):
            cooldown = 1.0
        else:
            cooldown = config.TTS_COOLDOWN_SEC

        if (now - last_ann) >= cooldown:
            self._last_announced[key] = now
            self._last_emotion[key]   = emotion
            self._queue_announcement(self._build_unknown_msg(emotion))

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on_person_left(self, name: str):
        self._last_announced.pop(name, None)
        self._last_emotion.pop(name, None)

    def on_registered(self, name: str):
        emotion = self._current_emotion or "Neutral"
        self._announce_queue.clear()
        self._queue_announcement(
            self._build_emotion_msg(name, emotion, certain=True)
        )
        t = time.time()
        self._last_announced[name] = t
        self._last_seen[name]      = t
        self._last_emotion[name]   = emotion
