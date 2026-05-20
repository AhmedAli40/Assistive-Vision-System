"""
shared/tts.py - Unified TTS engine
====================================
Uses Microsoft Edge TTS (neural voices) — sounds like a real human.
Free, no API key, works online.

Arabic  voice: ar-EG-SalmaNeural   (female, Egyptian Arabic)
English voice: en-US-AriaNeural    (female, natural US English)

Fallback chain:
  Edge TTS → gTTS → SAPI win32com → pyttsx3 → print

Install once:
  pip install edge-tts pygame

Language switch via set_language("ar") / set_language("en").
"""
import threading
import queue
import time
import os
import tempfile
import asyncio
import logging

logger = logging.getLogger(__name__)
_STOP = object()

# ── Voice catalog ────────────────────────────────────────────────────────────
# All available voices per language + gender
VOICE_CATALOG = {
    "ar": {
        "female": [
            "ar-EG-SalmaNeural",      # مصري أنثى
            "ar-SA-ZariyahNeural",    # سعودي أنثى
        ],
        "male": [
            "ar-EG-ShakirNeural",     # مصري ذكر
            "ar-SA-HamedNeural",      # سعودي ذكر
        ],
    },
    "en": {
        "female": [
            "en-US-AriaNeural",       # US English female
            "en-GB-SoniaNeural",      # British English female
        ],
        "male": [
            "en-US-GuyNeural",        # US English male
            "en-GB-RyanNeural",       # British English male
        ],
    },
}

# Current active voices — default: female for both
EDGE_VOICES = {
    "ar": VOICE_CATALOG["ar"]["female"][0],   # ar-EG-SalmaNeural
    "en": VOICE_CATALOG["en"]["female"][0],   # en-US-AriaNeural
}


class TTS:
    def __init__(self, rate: int = 150):
        self._rate  = rate
        self._lang  = "en"
        self._quiet = False

        self._q     = queue.Queue()
        self._done  = threading.Event()
        self._done.set()
        self._ready = threading.Event()

        self._fn_lock  = threading.Lock()
        self._speak_fn = None

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        self._ready.wait(timeout=15)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_language(self, lang: str):
        self._lang = lang
        print(f"[TTS] Language → {lang.upper()}")
        threading.Thread(target=self._rebuild_fn, daemon=True).start()

    def set_voice(self, lang: str, gender: str) -> str:
        """
        Switch voice for a language.
        lang:   'ar' or 'en'
        gender: 'male' or 'female'
        Returns the new voice name for confirmation.
        """
        gender = gender.strip().lower()
        if gender not in ("male", "female"):
            return ""
        if lang not in VOICE_CATALOG:
            return ""

        voices = VOICE_CATALOG[lang][gender]
        new_voice = voices[0]   # pick first option
        EDGE_VOICES[lang] = new_voice
        print(f"[TTS] Voice changed: {lang.upper()} → {new_voice}")

        # Rebuild engine if this is the active language
        if lang == self._lang:
            threading.Thread(target=self._rebuild_fn, daemon=True).start()

        return new_voice

    def get_voice_gender(self, lang: str) -> str:
        """Return current gender for a language: 'female' or 'male'."""
        current = EDGE_VOICES.get(lang, "")
        for gender, voices in VOICE_CATALOG.get(lang, {}).items():
            if current in voices:
                return gender
        return "female"

    def _rebuild_fn(self):
        fn = self._build_speak_fn(self._lang)
        with self._fn_lock:
            self._speak_fn = fn
        print(f"[TTS] Engine ready for {self._lang.upper()}")

    def say(self, text: str):
        if self._quiet:
            print(f"[TTS-QUIET] {text}")
            return
        print(f"[TTS] {text}")
        self._done.clear()
        self._q.put(text)

    def say_wait(self, text: str, pause: float = 1.2):
        print(f"[TTS] {text}")
        self._done.clear()
        self._q.put(text)
        self._done.wait(timeout=30)
        time.sleep(pause)

    def stop(self):
        while not self._q.empty():
            try: self._q.get_nowait()
            except: pass
        self._done.set()

    def set_quiet(self, quiet: bool):
        self._quiet = quiet
        if quiet:
            self.stop()
            print("[TTS] Quiet ON")
        else:
            print("[TTS] Quiet OFF")

    def is_quiet(self) -> bool:
        return self._quiet

    def wait(self, timeout: float = 15.0):
        self._done.wait(timeout=timeout)

    def busy(self) -> bool:
        return not self._done.is_set()

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker(self):
        fn = self._build_speak_fn(self._lang)
        with self._fn_lock:
            self._speak_fn = fn
        self._ready.set()

        while True:
            try:
                text = self._q.get(timeout=0.1)
            except queue.Empty:
                self._done.set()
                continue

            if text is _STOP:
                break

            with self._fn_lock:
                fn = self._speak_fn

            try:
                fn(text)
            except Exception as e:
                logger.warning(f"[TTS] error: {e}")
                try:
                    new_fn = self._build_speak_fn(self._lang)
                    with self._fn_lock:
                        self._speak_fn = new_fn
                    new_fn(text)
                except Exception as e2:
                    logger.error(f"[TTS] retry failed: {e2}")

            if self._q.empty():
                self._done.set()

    # ── Engine builder ────────────────────────────────────────────────────────

    def _build_speak_fn(self, lang: str):
        fn = self._try_edge_tts(lang)
        if fn: return fn
        fn = self._try_gtts(lang)
        if fn: return fn
        fn = self._try_sapi(lang)
        if fn: return fn
        fn = self._try_pyttsx3()
        if fn: return fn
        return self._speak_print

    # ── Method 1: Edge TTS (neural, human-like) ─────────────────────────────────
    # Uses an in-memory mp3 cache — repeated phrases play instantly.
    # Falls back to SAPI immediately if network is slow (>3s timeout).

    def _try_edge_tts(self, lang: str):
        try:
            import edge_tts
            import pygame
            pygame.mixer.init()
            voice = EDGE_VOICES.get(lang, EDGE_VOICES["en"])
            print(f"[TTS] Using Edge TTS neural voice: {voice}")

            # Per-voice mp3 cache {text: mp3_path}
            _cache = {}
            _cache_lock = threading.Lock()

            def _get_cached(text, _voice=voice):
                """Return cached mp3 path or generate and cache it."""
                with _cache_lock:
                    path = _cache.get(text)
                    if path and os.path.exists(path):
                        return path

                # Generate in temp file
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tmp.close()

                async def _synth():
                    comm = edge_tts.Communicate(text, _voice)
                    await comm.save(tmp.name)

                asyncio.run(_synth())

                # Cache up to 60 phrases — evict oldest if full
                with _cache_lock:
                    if len(_cache) >= 60:
                        oldest = next(iter(_cache))
                        try: os.unlink(_cache[oldest])
                        except Exception: pass
                        del _cache[oldest]
                    _cache[text] = tmp.name

                return tmp.name

            def _speak(text, _voice=voice):
                try:
                    # Try Edge TTS with 4s timeout
                    result = [None]
                    error  = [None]

                    def _gen():
                        try:
                            result[0] = _get_cached(text, _voice)
                        except Exception as e:
                            error[0] = e

                    t = threading.Thread(target=_gen, daemon=True)
                    t.start()
                    t.join(timeout=4.0)   # max 4 seconds to generate

                    if result[0] and os.path.exists(result[0]):
                        pygame.mixer.music.load(result[0])
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.05)
                        pygame.mixer.music.unload()
                    else:
                        # Timeout or error — raise to trigger SAPI fallback
                        raise RuntimeError(
                            f"Edge TTS timeout or failed: {error[0]}"
                        )
                except Exception as e:
                    logger.warning(f"[TTS] Edge TTS: {e} — falling back to SAPI")
                    # Immediate SAPI fallback so user hears something now
                    sapi_fn = self._try_sapi(lang)
                    if sapi_fn:
                        sapi_fn(text)
                        # Swap engine to SAPI permanently if Edge TTS keeps failing
                        with self._fn_lock:
                            self._speak_fn = sapi_fn
                        print("[TTS] Switched to SAPI (Edge TTS unavailable)")

            return _speak
        except ImportError:
            print("[TTS] edge-tts/pygame not installed — run: pip install edge-tts pygame")
            return None
        except Exception as e:
            logger.debug(f"[TTS] Edge TTS init: {e}")
            return None

    # ── Method 2: gTTS fallback ───────────────────────────────────────────────

    def _try_gtts(self, lang: str):
        try:
            from gtts import gTTS
            import pygame
            pygame.mixer.init()
            gtts_lang = "ar" if lang == "ar" else "en"
            print(f"[TTS] Using gTTS fallback ({gtts_lang})")

            def _speak(text, _lang=gtts_lang):
                try:
                    tts = gTTS(text=text, lang=_lang, slow=False)
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".mp3", delete=False
                    )
                    tmp.close()
                    tts.save(tmp.name)
                    pygame.mixer.music.load(tmp.name)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
                    pygame.mixer.music.unload()
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"[TTS] gTTS play: {e}")
                    raise

            return _speak
        except ImportError:
            return None
        except Exception as e:
            logger.debug(f"[TTS] gTTS: {e}")
            return None

    # ── Method 3: SAPI win32com ───────────────────────────────────────────────

    def _try_sapi(self, lang: str):
        try:
            import pythoncom
            pythoncom.CoInitialize()
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            voices  = speaker.GetVoices()

            keywords = (
                ["arabic", "naayf", "hoda", "ar-", "ar_"]
                if lang == "ar"
                else ["english", "david", "zira", "mark", "en-us"]
            )
            for i in range(voices.Count):
                v = voices.Item(i)
                d = v.GetDescription().lower()
                if any(k in d for k in keywords):
                    speaker.Voice = v
                    print(f"[TTS] SAPI: {v.GetDescription()}")
                    break

            speaker.Rate = max(-5, min(5, int((self._rate - 150) / 20)))

            def _speak(text, _s=speaker):
                _s.Speak(text)
            return _speak
        except Exception as e:
            logger.debug(f"[TTS] SAPI: {e}")
            return None

    # ── Method 4: pyttsx3 ────────────────────────────────────────────────────

    def _try_pyttsx3(self):
        try:
            import pythoncom
            pythoncom.CoInitialize()
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", self._rate)
            print("[TTS] Using pyttsx3")

            def _speak(text, _e=eng):
                _e.say(text)
                _e.runAndWait()
            return _speak
        except Exception as e:
            logger.debug(f"[TTS] pyttsx3: {e}")
            return None

    # ── Method 5: print only ──────────────────────────────────────────────────

    @staticmethod
    def _speak_print(text):
        print(f"[TTS-PRINT] {text}")
