# -*- coding: utf-8 -*-
import os
import re
import sys
from difflib import SequenceMatcher
import tempfile
import threading
import time
import warnings
import tkinter.font as tkfont

import customtkinter as ctk
import speech_recognition as sr

# Quieter Hugging Face cache on Windows (no symlink support)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings(
    "ignore",
    message=".*symlinks.*",
    category=UserWarning,
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

NIQQUD_RE = re.compile(r"[\u0591-\u05C7]")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def remove_niqqud(text: str) -> str:
    return NIQQUD_RE.sub("", text)


def split_words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


class ReadingCoach(ctk.CTk):
    def __init__(self):
        super().__init__()
        self._font_family = self._pick_hebrew_font()

        self.title("מאמן קריאה חכם - עברית")
        self.geometry("800x600")
        self.minsize(640, 480)

        self.target_text = "הַיֶּלֶד רָץ מַהֵר לַגִּינָה"
        self.is_recording = False
        self.max_word_progress = 0
        self.listen_timeout = ctk.DoubleVar(value=5.0)
        self.text_font_size = ctk.IntVar(value=45)

        self.recognizer = sr.Recognizer()
        self._whisper_model = None
        self._whisper_lock = threading.Lock()
        self.whisper_model_size = "tiny"
        self._whisper_backend_label = "CPU"
        self._audio_lock = threading.Lock()
        self._audio_buffer = None  # numpy float32 mono, set when streaming starts
        self._sample_rate = 16000
        self._mic_level = 0.0  # smoothed 0..1 for UI meter
        # התאמה מעורפלת בין מילה למילה (ילדים: חיתוך עיצורים, בליעת עיצורים)
        self.word_match_ratio_threshold = 0.8

        self.setup_ui()

    def _ensure_whisper(self):
        if self._whisper_model is not None:
            return self._whisper_model
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError("Install: pip install faster-whisper") from e

        # CPU: int8 quantization (~2–3× faster than float); tiny = smallest / fastest.
        import os as _os

        self._whisper_backend_label = "CPU"
        self._whisper_model = WhisperModel(
            self.whisper_model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=min(8, max(2, (_os.cpu_count() or 4))),
        )
        return self._whisper_model

    def _transcribe_numpy(self, samples) -> str:
        """Transcribe mono float32 PCM (e.g. numpy array), 16 kHz."""
        import numpy as np

        if samples is None or len(samples) < int(self._sample_rate * 0.18):
            return ""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return ""
        model = self._ensure_whisper()
        prompt = remove_niqqud(self.target_text).strip()
        transcribe_kw = {
            "language": "he",
            "beam_size": 1,
            "best_of": 1,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "without_timestamps": True,
        }
        if prompt:
            transcribe_kw["initial_prompt"] = prompt
        with self._whisper_lock:
            segments, _info = model.transcribe(x, **transcribe_kw)
        parts = [s.text.strip() for s in segments if s.text.strip()]
        return " ".join(parts)

    def _transcribe_audio(self, audio: sr.AudioData) -> str:
        """המרת AudioData מ־SpeechRecognition ל־WAV זמני ותמלול בעברית."""
        wav_bytes = audio.get_wav_data()
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(wav_bytes)
            model = self._ensure_whisper()
            prompt = remove_niqqud(self.target_text).strip()
            kw = {
                "language": "he",
                "beam_size": 1,
                "best_of": 1,
                "vad_filter": False,
            }
            if prompt:
                kw["initial_prompt"] = prompt
            with self._whisper_lock:
                segments, _info = model.transcribe(path, **kw)
            parts = [s.text.strip() for s in segments if s.text.strip()]
            return " ".join(parts)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _pick_hebrew_font(self) -> str:
        try:
            families = set(tkfont.families(root=self))
        except Exception:
            families = set()
        for name in (
            "Segoe UI",
            "Segoe UI Hebrew",
            "David",
            "Gisha",
            "Miriam",
            "Tahoma",
            "Arial",
        ):
            if name in families:
                return name
        return "Arial"

    def setup_ui(self):
        # 1. כותרת — למעלה
        self.label_title = ctk.CTkLabel(
            self, text="תרגול קריאה", font=(self._font_family, 24, "bold")
        )
        self.label_title.pack(pady=10, side="top")

        # 2. כפתורים וסטטוס — לתחתית (לפני אזור האמצע כדי שיישארו גלויים)
        self.bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.bottom_frame.pack(side="bottom", pady=20)

        self.btn_record = ctk.CTkButton(
            self.bottom_frame,
            text="התחל הקלטה 🎤",
            command=self.toggle_listening,
            width=200,
            height=50,
            font=(self._font_family, 18, "bold"),
            fg_color="#3498DB",
        )
        self.btn_record.grid(row=0, column=0, padx=10)

        self.btn_reset = ctk.CTkButton(
            self.bottom_frame,
            text="איפוס 🔄",
            command=self.reset_practice,
            width=120,
            height=50,
            fg_color="gray",
            font=(self._font_family, 16),
        )
        self.btn_reset.grid(row=0, column=1, padx=10)

        self.btn_settings = ctk.CTkButton(
            self.bottom_frame,
            text="הגדרות ⚙",
            command=self.open_settings,
            width=140,
            height=50,
            fg_color="#555",
            font=(self._font_family, 16),
        )
        self.btn_settings.grid(row=0, column=2, padx=10)

        self.status_label = ctk.CTkLabel(
            self, text="מוכן לתרגול", font=(self._font_family, 16)
        )
        self.status_label.pack(side="bottom", pady=5)

        # 3. תיבת טקסט — ממלאת את השטח שנשאר באמצע
        self.text_frame = ctk.CTkFrame(self, fg_color="#242424", corner_radius=15)
        self.text_frame.pack(pady=10, padx=40, fill="both", expand=True)

        # Bottom strip: thin L→R level line + large transcript (pack bottom first, words above)
        self.hearing_frame = ctk.CTkFrame(self.text_frame, fg_color="#1a1d2e", corner_radius=12)
        self.hearing_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 14))

        ctk.CTkLabel(
            self.hearing_frame,
            text="What the app hears",
            font=(self._font_family, 13, "bold"),
            text_color="#AAB4E0",
        ).pack(anchor="w", padx=14, pady=(10, 4))

        # Single horizontal meter: full width, low height (fills left → right)
        self.input_level_bar = ctk.CTkProgressBar(
            self.hearing_frame,
            height=6,
            progress_color="#2ECC71",
            fg_color="#2a2d3e",
        )
        self.input_level_bar.set(0)
        self.input_level_bar.pack(fill="x", padx=14, pady=(4, 6))

        self.input_level_hint = ctk.CTkLabel(
            self.hearing_frame,
            text="Microphone idle",
            font=(self._font_family, 11),
            text_color="gray",
        )
        self.input_level_hint.pack(anchor="w", padx=14, pady=(0, 6))

        self.live_label = ctk.CTkLabel(
            self.hearing_frame,
            text="Transcript will appear here while you speak.",
            font=(self._font_family, 26, "bold"),
            text_color="#D0DCFF",
            wraplength=620,
            justify="center",
        )
        self.live_label.pack(fill="x", padx=14, pady=(0, 16))

        self.words_inner = ctk.CTkFrame(self.text_frame, fg_color="#242424")
        self.words_inner.pack(
            fill="both", expand=True, padx=20, pady=(20, 8)
        )

        self.update_display()

    def open_settings(self):
        win = ctk.CTkToplevel(self)
        win.title("הגדרות")
        win.geometry("520x420")
        win.transient(self)
        win.grab_set()

        ctk.CTkLabel(
            win, text="משפט לתרגול (עם ניקוד או בלי):", font=(self._font_family, 14)
        ).pack(anchor="w", padx=16, pady=(16, 4))
        box = ctk.CTkTextbox(win, height=100, font=(self._font_family, 16))
        box.pack(fill="x", padx=16, pady=(0, 8))
        box.insert("1.0", self.target_text)

        ctk.CTkLabel(
            win, text="זמן המתנה לדיבור (שניות):", font=(self._font_family, 14)
        ).pack(anchor="w", padx=16, pady=(8, 0))
        ctk.CTkSlider(
            win,
            from_=3,
            to=15,
            number_of_steps=24,
            variable=self.listen_timeout,
        ).pack(fill="x", padx=16, pady=6)

        ctk.CTkLabel(
            win, text="גודל גופן טקסט המשפט:", font=(self._font_family, 14)
        ).pack(anchor="w", padx=16, pady=(8, 0))
        ctk.CTkSlider(
            win,
            from_=24,
            to=56,
            number_of_steps=32,
            variable=self.text_font_size,
            command=lambda _: self._apply_font_size(),
        ).pack(fill="x", padx=16, pady=6)

        ctk.CTkLabel(win, text="מצב תצוגה:", font=(self._font_family, 14)).pack(
            anchor="w", padx=16, pady=(8, 0)
        )
        try:
            _mode = ctk.get_appearance_mode()
        except Exception:
            _mode = "dark"
        mode_var = ctk.StringVar(value=_mode)
        ctk.CTkOptionMenu(
            win,
            values=["dark", "light", "system"],
            variable=mode_var,
            command=lambda m: ctk.set_appearance_mode(m),
            font=(self._font_family, 14),
        ).pack(anchor="w", padx=16, pady=6)

        def apply_and_close():
            raw = box.get("1.0", "end").strip()
            if raw:
                self.target_text = raw
                self.max_word_progress = 0
                self.update_display()
            win.destroy()

        ctk.CTkButton(
            win,
            text="שמור והחל",
            command=apply_and_close,
            font=(self._font_family, 16),
            height=40,
        ).pack(pady=20)

    def _apply_font_size(self):
        self.update_display()

    def update_display(self):
        """מציג כל מילה כ־Label; מילים שנקראו בירוק. סדר: ימין→שמאל (המילה הראשונה בימין)."""
        for child in self.words_inner.winfo_children():
            child.destroy()

        words = split_words(self.target_text)
        if not words:
            ctk.CTkLabel(
                self.words_inner,
                text="(הוסף משפט בהגדרות)",
                text_color="gray",
                font=(self._font_family, 20),
            ).pack(anchor="center", expand=True)
            return

        done = self.max_word_progress
        fs = self.text_font_size.get()
        for i, word in enumerate(words):
            color = "#2ECC71" if i < done else "#CCCCCC"
            ctk.CTkLabel(
                self.words_inner,
                text=word,
                text_color=color,
                font=(self._font_family, fs, "bold"),
            ).pack(side="right", padx=6, pady=8)

    def reset_practice(self):
        self.is_recording = False
        self.max_word_progress = 0
        self._set_record_button_idle()
        self.update_display()
        self.live_label.configure(
            text="Transcript will appear here while you speak."
        )
        self.input_level_bar.set(0)
        self.input_level_hint.configure(text="Microphone idle", text_color="gray")
        self.status_label.configure(
            text="התרגול אופס. מוכן להתחלה", text_color="white"
        )

    def _set_record_button_idle(self):
        self.btn_record.configure(
            text="התחל הקלטה 🎤",
            fg_color="#3498DB",
            state="normal",
        )

    def _set_record_button_recording(self):
        self.btn_record.configure(
            text="עצור הקלטה ⏹",
            fg_color="#C0392B",
            state="normal",
        )

    def toggle_listening(self):
        if not self.is_recording:
            self.is_recording = True
            self.max_word_progress = 0
            self.update_display()
            self._set_record_button_recording()
            self.status_label.configure(
                text="Starting… (first run loads Whisper)", text_color="white"
            )
            threading.Thread(target=self.process_audio_loop, daemon=True).start()
        else:
            self.is_recording = False
            self._set_record_button_idle()
            self.live_label.configure(
                text="Transcript will appear here while you speak."
            )
            self.input_level_bar.set(0)
            self.input_level_hint.configure(
                text="Microphone idle", text_color="gray"
            )
            self.status_label.configure(text="ההקלטה נעצרה", text_color="orange")

    def _apply_stream_result(self, text: str) -> None:
        preview = (text or "").strip()
        if preview:
            self.live_label.configure(text=f"Transcript: {preview[:280]}")
            self.compare_texts(preview)

    def _level_ui_tick(self) -> None:
        """Update mic level bar from smoothed RMS (main thread)."""
        if not self.is_recording:
            self.input_level_bar.set(0)
            self.input_level_hint.configure(text="Microphone idle", text_color="gray")
            return
        with self._audio_lock:
            lv = float(getattr(self, "_mic_level", 0.0))
        lv = max(0.0, min(1.0, lv))
        self.input_level_bar.set(lv)
        if lv < 0.04:
            self.input_level_hint.configure(
                text="Quiet — speak closer to the mic or raise volume",
                text_color="#888888",
            )
        elif lv < 0.12:
            self.input_level_hint.configure(
                text="Low level — keep speaking",
                text_color="#CCCCAA",
            )
        else:
            self.input_level_hint.configure(
                text="Sound detected — good input level",
                text_color="#2ECC71",
            )
        self.after(70, self._level_ui_tick)

    def process_audio_loop(self):
        """Real-time style: mic capture runs continuously; we transcribe overlapping windows often."""
        import numpy as np

        def set_status(msg: str, color: str = "white") -> None:
            self.after(
                0,
                lambda m=msg, c=color: self.status_label.configure(
                    text=m, text_color=c
                ),
            )

        def set_live(msg: str) -> None:
            self.after(0, lambda m=msg: self.live_label.configure(text=m))

        set_status(
            "Loading Whisper (first run may download ~75MB)…",
            "#AED6F1",
        )
        try:
            self._ensure_whisper()
            set_status(
                f"Whisper ready ({getattr(self, '_whisper_backend_label', 'CPU')}) — listening…",
                "#AED6F1",
            )
        except RuntimeError as e:
            set_status(str(e)[:200], "red")
            self.is_recording = False
            self.after(0, self._set_record_button_idle)
            return

        try:
            import pyaudio
        except ImportError:
            set_status("Install: pip install PyAudio numpy", "red")
            self.is_recording = False
            self.after(0, self._set_record_button_idle)
            return

        sr = self._sample_rate
        # מרווח ארוך יותר על CPU — פחות עומס מחזורי transcribe; אפשר להוריד ל-0.45 אם חלק
        step = 0.6
        max_buf_sec = 3.0
        max_samples = int(sr * max_buf_sec)
        frames_per_buffer = 4000  # 0.25 s @ 16 kHz per read()

        with self._audio_lock:
            self._audio_buffer = np.zeros(0, dtype=np.float32)

        def mic_reader():
            pa = None
            stream = None
            try:
                pa = pyaudio.PyAudio()
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=sr,
                    input=True,
                    frames_per_buffer=frames_per_buffer,
                )
                while self.is_recording:
                    data = stream.read(
                        frames_per_buffer,
                        exception_on_overflow=False,
                    )
                    chunk = (
                        np.frombuffer(data, dtype=np.int16)
                        .astype(np.float32)
                        .reshape(-1)
                    )
                    chunk /= 32768.0
                    if chunk.size:
                        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
                        instant = min(1.0, rms * 12.0)
                        with self._audio_lock:
                            b = self._audio_buffer
                            if b is None:
                                continue
                            prev = float(getattr(self, "_mic_level", 0.0))
                            self._mic_level = prev * 0.78 + instant * 0.22
                            self._audio_buffer = np.concatenate([b, chunk])
                            if self._audio_buffer.size > max_samples:
                                self._audio_buffer = self._audio_buffer[-max_samples:]
            except Exception as ex:
                self.after(
                    0,
                    lambda e=str(ex): self.status_label.configure(
                        text=f"Mic stream error: {e[:120]}",
                        text_color="red",
                    ),
                )
                self.is_recording = False
            finally:
                if stream is not None:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
                if pa is not None:
                    try:
                        pa.terminate()
                    except Exception:
                        pass

        reader = threading.Thread(target=mic_reader, daemon=True)
        reader.start()

        self._mic_level = 0.0
        self.after(0, self._level_ui_tick)

        set_status("Listening (live)… speak continuously.", "#AED6F1")
        set_live("Transcript: …")

        while self.is_recording:
            time.sleep(step)
            with self._audio_lock:
                snap = (
                    None
                    if self._audio_buffer is None
                    else self._audio_buffer.copy()
                )
            if snap is None or snap.size < int(sr * 0.18):
                continue

            set_status("Transcribing…", "#AED6F1")
            t0 = time.monotonic()
            try:
                text = self._transcribe_numpy(snap)
            except Exception as ex:
                self.after(
                    0,
                    lambda e=str(ex): self.status_label.configure(
                        text=f"Transcribe error: {e[:100]}",
                        text_color="red",
                    ),
                )
                continue
            dt = time.monotonic() - t0
            if self.is_recording:
                set_status(
                    f"Listening (live)… last pass {dt:.1f}s",
                    "#AED6F1",
                )

            if not (text or "").strip():
                continue

            self.after(0, lambda t=text: self._apply_stream_result(t))

        with self._audio_lock:
            self._audio_buffer = None
            self._mic_level = 0.0

        def _zero_hearing_ui():
            self.input_level_bar.set(0)
            self.input_level_hint.configure(text="Microphone idle", text_color="gray")

        self.after(0, _zero_hearing_ui)
        self.after(0, self._on_record_loop_stopped)

    def _on_record_loop_stopped(self):
        if not self.is_recording:
            self._set_record_button_idle()

    @staticmethod
    def _clean_google_words(text: str) -> str:
        """ניקוי ניקוד ותווים שאינם אותיות/רווח (פיסוק מהתמלול)."""
        no_niq = remove_niqqud(text)
        no_punct = re.sub(r"[^\w\s]", "", no_niq, flags=re.UNICODE)
        return re.sub(r"\s+", " ", no_punct).strip()

    def is_word_match(self, spoken_word: str, target_word: str) -> bool:
        """האם המילה הדומה מספיק ליעד (הגייה / טעויות מודל קלות)."""
        s1 = spoken_word.strip()
        s2 = target_word.strip()
        if s1 == s2:
            return True
        if not s1 or not s2:
            return False
        thr = float(getattr(self, "word_match_ratio_threshold", 0.8))
        return SequenceMatcher(None, s1, s2).ratio() >= thr

    def compare_texts(self, recognized: str):
        target_clean = remove_niqqud(self.target_text).strip()
        rec_clean = self._clean_google_words(recognized)

        target_words = split_words(target_clean)
        rec_words = split_words(rec_clean)

        # רצף מילים מההתחלה בהקלט הנוכחי (התאמה מדויקת או דמיון מילולי)
        current_match = 0
        for i in range(len(target_words)):
            if i < len(rec_words) and self.is_word_match(rec_words[i], target_words[i]):
                current_match = i + 1
            else:
                break

        if current_match > self.max_word_progress:
            self.max_word_progress = current_match

        self.update_display()

        n_target = len(target_words)
        if n_target > 0 and self.max_word_progress >= n_target:
            self.status_label.configure(
                text="אלוף! סיימת את המשפט! 🎉", text_color="green"
            )
            self.is_recording = False
            self._set_record_button_idle()
        elif self.max_word_progress > 0:
            self.status_label.configure(
                text=(
                    f"התקדמות (שיא בסשן): {self.max_word_progress}/{n_target} מילים "
                    f"· בהקלטה האחרונה: {current_match}/{n_target}"
                ),
                text_color="#AED6F1",
            )
        else:
            self.status_label.configure(
                text=f"שמעתי: {recognized} — נסה להתאים למשפט מההתחלה",
                text_color="orange",
            )


if __name__ == "__main__":
    app = ReadingCoach()
    app.mainloop()
