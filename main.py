# -*- coding: utf-8 -*-
import os
import random
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


# (menu label, full passage). Add more stories as extra tuples later.
READING_PASSAGES: list[tuple[str, str]] = [
    (
        "1 · Short practice",
        "הַיֶּלֶד רָץ מַהֵר לַגִּינָה",
    ),
    (
        "2 · Story: The computer in my room (full)",
        " ".join(
            [
                "בַּחֶדֶר שֶׁלִּי יֵשׁ מַחְשֵׁב חָדָשׁ וְחָכָם.",
                "בְּכָל בֹּקֶר אֲנִי מַדְלִיק אוֹתוֹ וְרוֹאֶה אוֹרוֹת צִבְעוֹנִיִּים.",
                "אֲנִי אוֹהֵב לִכְתֹּב קוֹד וּלְצַיֵּר צִיּוּרִים יָפִים עַל הַמָּסָךְ.",
                "הַמַּחְשֵׁב עוֹזֵר לִי לִלְמֹד דְּבָרִים חֲדָשִׁים עַל כּוֹכָבִים רְחוֹקִים וְעַל חַיּוֹת בַּיָּם.",
                "אֲנִי חוֹלֵם שֶׁיּוֹם אֶחָד אֲנִי אֶבְנֶה רוֹבּוֹט אֲמִתִּי שֶׁיּוּכַל לְדַבֵּר אִתִּי.",
                "עַד אָז, אֲנִי מַמְשִיךְ לְהִתְאַמֵּן וְלִקְרֹא בְּכָל יוֹם.",
            ]
        ),
    ),
    (
        "3 · Story: Teacher & Danny (apples)",
        " ".join(
            [
                'הַמּוֹרֶה שָׁאַל אֶת הַתַּלְמִיד: "דָּנִי, אִם יֵשׁ לִי חֲמִשָּׁה תַּפּוּחִים בְּיָד אַחַת וַחֲמִשָּׁה תַּפּוּחִים בַּיָּד הַשְּׁנִיָּה, מָה יֵשׁ לִי?"',
                'דָּנִי חָשַׁב לְרֶגַע וְעָנָה בְּחִיּוּךְ: "יֵשׁ לְךָ יָדַיִם מַמָּשׁ גְּדוֹלוֹת, הַמּוֹרֶה!"',
            ]
        ),
    ),
]

WORD_POINTS = 10
STORY_BONUS_POINTS = 50


class ReadingCoach(ctk.CTk):
    def __init__(self):
        super().__init__()
        self._font_family = self._pick_hebrew_font()

        self.title("מאמן קריאה חכם - עברית")
        self.geometry("920x680")
        self.minsize(720, 520)

        self.target_text = READING_PASSAGES[0][1]
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
        # Only transcribe the last N seconds + single-word prompt — less full-sentence drift
        self.transcribe_focus_tail_sec = 1.5
        self.session_points = 0
        self._word_labels: list = []
        self._feedback_after_id = None

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

    def _target_words_clean(self) -> list[str]:
        return split_words(remove_niqqud(self.target_text).strip())

    def _current_target_word(self) -> str | None:
        words = self._target_words_clean()
        i = self.max_word_progress
        return words[i] if i < len(words) else None

    def _transcribe_numpy(self, samples) -> str:
        """Transcribe mono float32 PCM (e.g. numpy array), 16 kHz — tuned for one word at a time."""
        import numpy as np

        if samples is None or len(samples) < int(self._sample_rate * 0.18):
            return ""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return ""
        tail_sec = float(getattr(self, "transcribe_focus_tail_sec", 1.5))
        max_tail = int(self._sample_rate * tail_sec)
        if x.size > max_tail:
            x = x[-max_tail:]
        model = self._ensure_whisper()
        transcribe_kw = {
            "language": "he",
            "beam_size": 1,
            "best_of": 1,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "without_timestamps": True,
        }
        cw = self._current_target_word()
        if cw:
            transcribe_kw["initial_prompt"] = cw
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
            kw = {
                "language": "he",
                "beam_size": 1,
                "best_of": 1,
                "vad_filter": False,
            }
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

        self.score_row = ctk.CTkFrame(self, fg_color="transparent")
        self.score_row.pack(fill="x", padx=28, pady=(0, 4))
        self.points_label = ctk.CTkLabel(
            self.score_row,
            text="Points: 0",
            font=(self._font_family, 18, "bold"),
            text_color="#F4D03F",
        )
        self.points_label.pack(side="left", padx=(0, 16))
        self.feedback_label = ctk.CTkLabel(
            self.score_row,
            text="",
            font=(self._font_family, 16, "bold"),
            text_color="#2ECC71",
        )
        self.feedback_label.pack(side="left")

        self.passage_bar = ctk.CTkFrame(self, fg_color="transparent")
        self.passage_bar.pack(fill="x", padx=28, pady=(0, 6))
        ctk.CTkLabel(
            self.passage_bar,
            text="Practice text:",
            font=(self._font_family, 15, "bold"),
            text_color="#AAB4E0",
        ).pack(side="left", padx=(0, 12))
        self._passage_labels = [p[0] for p in READING_PASSAGES]
        self.passage_menu = ctk.CTkOptionMenu(
            self.passage_bar,
            values=self._passage_labels,
            command=self.on_passage_selected,
            font=(self._font_family, 15),
            width=520,
            anchor="w",
        )
        self.passage_menu.pack(side="left", fill="x", expand=True)
        self.passage_menu.set(self._passage_labels[0])

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

        self.words_scroll = ctk.CTkScrollableFrame(
            self.text_frame,
            orientation="horizontal",
            fg_color="#242424",
            height=150,
            scrollbar_button_color="#3d4159",
            scrollbar_button_hover_color="#5c6bc0",
        )
        self.words_scroll.pack(
            fill="both", expand=True, padx=12, pady=(12, 8)
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

    def _scroll_focus_word_to_center(self) -> None:
        """Keep the current word in the middle of the horizontal scroll view."""
        labels = getattr(self, "_word_labels", None)
        if not labels:
            return
        sc = self.words_scroll
        try:
            canvas = sc._parent_canvas
        except AttributeError:
            return
        self.update_idletasks()
        bbox = canvas.bbox("all")
        if not bbox:
            return
        total_w = max(bbox[2] - bbox[0], 1)
        vw = max(canvas.winfo_width(), 1)
        idx = self.max_word_progress
        if idx >= len(labels):
            idx = len(labels) - 1
        if idx < 0:
            return
        lbl = labels[idx]
        cx = lbl.winfo_x() + lbl.winfo_width() / 2
        target_left = cx - vw / 2
        target_left = max(0.0, min(target_left, max(total_w - vw, 0)))
        frac = target_left / total_w if total_w > 0 else 0.0
        canvas.xview_moveto(frac)

    def update_display(self):
        """Word colors: done=green, current=bright (full niqqud visible), upcoming=dim gray. RTL pack."""
        for child in self.words_scroll.winfo_children():
            child.destroy()
        self._word_labels = []

        words = split_words(self.target_text)
        if not words:
            ctk.CTkLabel(
                self.words_scroll,
                text="(הוסף משפט בהגדרות)",
                text_color="gray",
                font=(self._font_family, 20),
            ).pack(anchor="center", expand=True)
            return

        done = self.max_word_progress
        fs = self.text_font_size.get()
        n = len(words)
        bold_font = ctk.CTkFont(
            family=self._font_family, size=fs, weight="bold", underline=False
        )
        # Dark scroll area: "black" reads best as high-contrast light text; no underline (keeps niqqud clear)
        color_done = "#2ECC71"
        color_current = "#F5F5F5"
        color_upcoming = "#5C5C5C"
        for i, word in enumerate(words):
            if i < done:
                color = color_done
            elif i == done and done < n:
                color = color_current
            else:
                color = color_upcoming
            font_arg = bold_font
            wlab = ctk.CTkLabel(
                self.words_scroll,
                text=word,
                text_color=color,
                font=font_arg,
            )
            wlab.pack(side="right", padx=6, pady=8)
            self._word_labels.append(wlab)
        self.after(20, self._scroll_focus_word_to_center)

    def _update_points_display(self) -> None:
        self.points_label.configure(text=f"Points: {self.session_points}")

    def _clear_feedback(self) -> None:
        self.feedback_label.configure(text="")
        self._feedback_after_id = None

    def _show_word_feedback(self) -> None:
        cheers = [
            "Nice!",
            "Great job!",
            "You got it!",
            "Super!",
            "Well done!",
            "Awesome!",
            "Perfect!",
            "Keep going!",
        ]
        self.feedback_label.configure(
            text=f"+{WORD_POINTS}  {random.choice(cheers)}",
            text_color="#2ECC71",
        )
        if self._feedback_after_id is not None:
            try:
                self.after_cancel(self._feedback_after_id)
            except Exception:
                pass
        self._feedback_after_id = self.after(2200, self._clear_feedback)

    def _show_story_complete_feedback(self) -> None:
        if self._feedback_after_id is not None:
            try:
                self.after_cancel(self._feedback_after_id)
            except Exception:
                pass
        self.feedback_label.configure(
            text=f"+{STORY_BONUS_POINTS}  Story finished — you're a star!",
            text_color="#F39C12",
        )
        self._feedback_after_id = self.after(4500, self._clear_feedback)

    def sync_passage_menu_state(self) -> None:
        self.passage_menu.configure(state="disabled" if self.is_recording else "normal")

    def on_passage_selected(self, choice: str) -> None:
        for label, text in READING_PASSAGES:
            if label == choice:
                self.target_text = text
                break
        self.max_word_progress = 0
        self.session_points = 0
        self._update_points_display()
        self._clear_feedback()
        self.update_display()

    def reset_practice(self):
        self.is_recording = False
        self.max_word_progress = 0
        self.session_points = 0
        self._update_points_display()
        self._clear_feedback()
        self._set_record_button_idle()
        self.update_display()
        self.sync_passage_menu_state()
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
            self.after(0, lambda: self._apply_stream_result(""))
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
        self.sync_passage_menu_state()

    def _apply_stream_result(self, text: str) -> None:
        preview = (text or "").strip()
        cw = self._current_target_word()
        if cw:
            heard = preview if preview else "(listening…)"
            self.live_label.configure(
                text=f"Say this word: {cw}\nHeard: {heard[:120]}",
            )
        else:
            self.live_label.configure(text="Sentence complete.")
        if preview:
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
            self.after(0, self.sync_passage_menu_state)
            return

        try:
            import pyaudio
        except ImportError:
            set_status("Install: pip install PyAudio numpy", "red")
            self.is_recording = False
            self.after(0, self._set_record_button_idle)
            self.after(0, self.sync_passage_menu_state)
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
                self.after(0, self.sync_passage_menu_state)
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

            self.after(0, lambda t=text or "": self._apply_stream_result(t))

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
            self.sync_passage_menu_state()

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

        if not target_words:
            return

        cur = self.max_word_progress
        if cur >= len(target_words):
            return

        tw = target_words[cur]
        # Any token in this short tail transcript can match the one word we care about now
        matched = any(self.is_word_match(rw, tw) for rw in rec_words)

        prev = self.max_word_progress
        candidate = cur + 1 if matched else cur
        candidate = min(candidate, len(target_words))
        new_total = max(prev, min(candidate, prev + 1))

        if new_total > prev:
            self.session_points += WORD_POINTS
            self._update_points_display()
            self._show_word_feedback()

        self.max_word_progress = new_total

        current_match = new_total

        self.update_display()

        n_target = len(target_words)
        if n_target > 0 and self.max_word_progress >= n_target:
            self.session_points += STORY_BONUS_POINTS
            self._update_points_display()
            self._show_story_complete_feedback()
            self.status_label.configure(
                text="אלוף! סיימת את המשפט! 🎉", text_color="green"
            )
            self.is_recording = False
            self._set_record_button_idle()
            self.sync_passage_menu_state()
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
