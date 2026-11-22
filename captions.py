"""Live Taglish subtitles for Windows.

Captures system audio (WASAPI loopback) or the microphone, transcribes it with
faster-whisper (large-v3-turbo on CUDA), and shows it in an always-on-top
overlay. Words go grey while still uncertain and turn white once two
consecutive passes agree on them (LocalAgreement-2).

Overlay: drag to move, drag the left/right edge to resize, Ctrl+scroll for text
size, Shift+scroll for opacity, right-click (or the ☰ shown on hover) for the menu.
Position and settings are remembered in settings.json.

Hotkeys (global):
  Ctrl+Alt+L  lock / unlock the overlay (locked = clicks pass through)
  Ctrl+Alt+H  hide / show the overlay
  Ctrl+Alt+Q  quit
"""
import argparse
import ctypes
import importlib.util
import json
import os
import queue
import re
import threading
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SETTINGS = HERE / "settings.json"
SR = 16000

# ctranslate2 needs cuDNN/cuBLAS DLLs; reuse the ones shipped with torch if present.
_spec = importlib.util.find_spec("torch")
if _spec and _spec.origin:
    _lib = Path(_spec.origin).parent / "lib"
    if _lib.is_dir():
        os.add_dll_directory(str(_lib))
        os.environ["PATH"] = str(_lib) + os.pathsep + os.environ.get("PATH", "")


# --------------------------------------------------------------------------- audio
class AudioSource:
    """Pushes mono float32 16 kHz chunks into self.q."""

    def __init__(self, source: str, device_hint: str | None = None):
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.source = source
        self.device_hint = device_hint
        self.name = source
        self._stream = None
        self._pa = None

    def start(self):
        if Path(self.source).is_file():
            self.name = Path(self.source).name
            threading.Thread(target=self._file_feeder, daemon=True).start()
            return self
        import pyaudiowpatch as pa
        from scipy.signal import resample_poly

        # A fresh PyAudio only sees newly plugged devices if no other instance is
        # alive, so callers must stop() the previous source before start()ing this one.
        self._pa = p = pa.PyAudio()
        wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
        kind, _, want = self.source.partition(":")

        def find(cands, name):
            for d in cands:
                n = d["name"].replace(" [Loopback]", "")
                if n == name or n.startswith(name) or name.startswith(n):
                    return d
            self.stop()
            raise RuntimeError(f"'{name}' is not connected")

        if self.source == "mic":
            dev = p.get_device_info_by_index(wasapi["defaultInputDevice"])
        elif self.source == "system":
            dev = p.get_default_wasapi_loopback()
        elif kind == "out":
            dev = find(p.get_loopback_device_info_generator(), want)
        elif kind == "in":
            devs = (p.get_device_info_by_index(i) for i in range(p.get_device_count()))
            dev = find((d for d in devs if d["hostApi"] == wasapi["index"] and d["maxInputChannels"] > 0
                        and not d.get("isLoopbackDevice")), want)
        else:
            self.stop()
            raise ValueError(f"unknown source {self.source!r}")
        if self.device_hint:
            for i in range(p.get_device_count()):
                d = p.get_device_info_by_index(i)
                if self.device_hint.lower() in d["name"].lower() and d["maxInputChannels"] > 0:
                    dev = d
                    break
        rate = int(dev["defaultSampleRate"])
        ch = max(1, int(dev["maxInputChannels"]))
        g = np.gcd(rate, SR)
        up, down = SR // g, rate // g
        self.name = dev["name"].replace(" [Loopback]", "")
        print(f"[audio] {dev['name']} @ {rate} Hz x{ch}", flush=True)

        def cb(data, frames, t, status):
            x = np.frombuffer(data, np.float32).reshape(-1, ch).mean(axis=1)
            self.q.put(resample_poly(x, up, down).astype(np.float32))
            return (None, pa.paContinue)

        self._stream = p.open(format=pa.paFloat32, channels=ch, rate=rate, input=True,
                              input_device_index=dev["index"], frames_per_buffer=rate // 10,
                              stream_callback=cb)
        self._stream.start_stream()
        return self

    def _file_feeder(self):
        """Test mode: stream a WAV file in real time."""
        import wave
        from scipy.signal import resample_poly
        with wave.open(self.source) as wf:
            rate, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        x = np.frombuffer(raw, {2: np.int16, 4: np.int32}[sw]).astype(np.float32)
        x = x.reshape(-1, ch).mean(axis=1) / (32768.0 if sw == 2 else 2147483648.0)
        if rate != SR:
            g = np.gcd(rate, SR)
            x = resample_poly(x, SR // g, rate // g).astype(np.float32)
        step = SR // 10
        t0 = time.perf_counter()
        for i in range(0, len(x), step):
            self.q.put(x[i:i + step])
            time.sleep(max(0.0, t0 + (i + step) / SR - time.perf_counter()))
        self.q.put(np.zeros(SR * 3, np.float32))  # trailing silence to flush
        time.sleep(3)
        self.q.put(None)

    def stop(self):
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
            if self._pa:
                self._pa.terminate()
        except Exception as e:
            print(f"[audio] stop: {e}", flush=True)
        self._stream = self._pa = None


class DeviceWatcher(threading.Thread):
    """Polls Windows' audio endpoints (Core Audio via pycaw) every couple of seconds.

    Keeps a snapshot of active outputs/inputs and the current defaults, and calls
    on_change(kind) with "out", "in" or "list" when something changes.
    """

    def __init__(self, on_change, interval=2.0):
        super().__init__(daemon=True)
        self.on_change = on_change
        self.interval = interval
        self.lock = threading.Lock()
        self.snap = {"outs": [], "ins": [], "out_default": None, "in_default": None}
        self.ok = True
        self._devnames: dict[str, str] = {}

    def snapshot(self):
        with self.lock:
            return dict(self.snap)

    def _devname(self, dev):
        from pycaw.pycaw import AudioUtilities
        i = dev.GetId()
        if i not in self._devnames:
            self._devnames[i] = AudioUtilities.CreateDevice(dev).FriendlyName
        return i, self._devnames[i]

    def run(self):
        try:
            import comtypes
            from pycaw.pycaw import IMMDeviceEnumerator
            from pycaw.constants import CLSID_MMDeviceEnumerator
            comtypes.CoInitialize()
            en = comtypes.CoCreateInstance(CLSID_MMDeviceEnumerator, IMMDeviceEnumerator,
                                           comtypes.CLSCTX_INPROC_SERVER)
        except Exception as e:
            print(f"[devices] watcher unavailable ({e}); install pycaw for device switching", flush=True)
            self.ok = False
            return
        prev = None
        while True:
            try:
                snap = {}
                for flow, key in ((0, "out"), (1, "in")):
                    col = en.EnumAudioEndpoints(flow, 1)  # DEVICE_STATE_ACTIVE
                    snap[key + "s"] = [self._devname(col.Item(i))[1] for i in range(col.GetCount())]
                    try:
                        snap[key + "_default"] = self._devname(en.GetDefaultAudioEndpoint(flow, 1))[1]
                    except Exception:
                        snap[key + "_default"] = None
                with self.lock:
                    self.snap = snap
                if prev is not None:
                    if snap["out_default"] != prev["out_default"]:
                        self.on_change("out")
                    if snap["in_default"] != prev["in_default"]:
                        self.on_change("in")
                    if snap["outs"] != prev["outs"] or snap["ins"] != prev["ins"]:
                        self.on_change("list")
                prev = snap
            except Exception as e:
                print(f"[devices] poll failed: {e}", flush=True)
            time.sleep(self.interval)


# --------------------------------------------------------------------------- transcriber
HALLUCINATIONS = re.compile(
    r"(thank(s| you) for watching|subscribe|like and share|salamat sa panonood|"
    r"please like|amara\.org|subtitles by|transcribed by)", re.I)


def norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


class Streamer:
    """Rolling-buffer Whisper with LocalAgreement-2 commits.

    on_update(tokens) receives [(word, final, starts_new_line), ...] for the
    recent committed words followed by the tentative ones; on_update(None)
    signals a pause.
    """

    def __init__(self, args, on_update):
        self.args = args
        self.on_update = on_update
        self.status = "loading"      # loading | listening | speech | error
        self.model = None
        self.src: AudioSource | None = None
        self.prompt = args.prompt
        self.lat: list[float] = []
        self.shown_lag: list[float] = []
        self.log = open(args.log, "a", encoding="utf-8") if args.log else None
        self.committed: list[tuple[float, float, str]] = []
        self.breaks: set[int] = set()  # indices in committed that start a new line
        self.t_start = None
        self._reset_needed = False
        self.reset()

    def load(self):
        from faster_whisper import WhisperModel
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        self._vad = lambda a: get_speech_timestamps(a, VadOptions(min_silence_duration_ms=300))
        t = time.perf_counter()
        try:
            self.model = WhisperModel(self.args.model, device=self.args.device,
                                      compute_type=self.args.compute_type)
        except Exception as e:  # no CUDA -> CPU fallback
            print(f"[model] {self.args.device} failed ({e}); falling back to CPU small", flush=True)
            self.model = WhisperModel("small", device="cpu", compute_type="int8")
        print(f"[model] {self.args.model} loaded in {time.perf_counter() - t:.1f}s", flush=True)

    def reset(self):
        self.buf = np.zeros(0, np.float32)
        self.buf_t0 = 0.0          # absolute time (s) of buf[0]
        self.total = 0.0           # absolute seconds of audio received
        self.commit_t = 0.0
        self.prev: list[tuple[float, float, str]] = []
        self.silence = 0.0
        if self.committed:
            self.breaks.add(len(self.committed))

    def switch_source(self, src: "AudioSource"):
        """Called from the UI thread after the old source was stopped."""
        self.src = src
        self._reset_needed = True

    def clear(self):
        self.breaks.add(len(self.committed))
        self.prev = []
        self.on_update([])

    # -- whisper pass over the current buffer, returns words with absolute times
    def _hyp(self):
        # context = committed text whose audio is no longer in the buffer
        tail = " ".join(w for _, b, w in self.committed[-40:] if b <= self.buf_t0 + 0.05)
        prompt = (self.prompt + " " + tail[-250:]).strip() or None
        segs, _ = self.model.transcribe(
            self.buf, language=self.args.lang, task=self.args.task, beam_size=self.args.beam,
            word_timestamps=True, condition_on_previous_text=False, initial_prompt=prompt,
            vad_filter=False, temperature=0.0, max_new_tokens=96)
        words = []
        for s in segs:
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                continue
            if HALLUCINATIONS.search(s.text):
                continue
            for w in s.words or []:
                if w.word.strip():
                    words.append((self.buf_t0 + w.start, self.buf_t0 + w.end, w.word.strip()))
        if self.args.verbose > 1:
            print("   raw:", " ".join(f"{w}@{a - self.buf_t0:.1f}" for a, _, w in words), flush=True)
        if not self.committed:
            return words
        # Words near the commit point may already be committed (timestamps drift
        # between passes). Align on the committed tail by text, then by time.
        cand = [w for w in words if w[0] >= self.commit_t - 1.5]
        ct = [norm(w) for _, _, w in self.committed[-10:]]
        cn = [norm(w) for _, _, w in cand]
        for n in range(min(8, len(ct)), 1, -1):
            for i in range(0, 3):
                if cn[i:i + n] == ct[-n:]:
                    return cand[i + n:]
        out = [w for w in cand if w[0] >= self.commit_t - 0.1]
        if out and ct and norm(out[0][2]) == ct[-1] and out[0][0] < self.commit_t + 0.3:
            out = out[1:]
        return out

    def _commit(self, words):
        if not words:
            return
        self.committed.extend(words)
        self.commit_t = words[-1][1]
        if self.t_start is not None:
            now = time.perf_counter() - self.t_start
            self.lat.extend(now - b for _, b, _ in words)
        if self.log:
            self.log.write(time.strftime("%H:%M:%S ") + " ".join(w for _, _, w in words) + "\n")
            self.log.flush()

    def _trim(self):
        dur = len(self.buf) / SR
        if dur < self.args.max_buffer * 0.6:
            return
        cut = None
        for a, b, w in reversed(self.committed):
            if b <= self.buf_t0:
                break
            if w[-1:] in ".?!":
                cut = b
                break
        if cut is None and dur > self.args.max_buffer and self.commit_t > self.buf_t0:
            cut = self.commit_t
        if cut is None and dur > self.args.max_buffer * 1.5:
            self._commit(self.prev)
            self.prev = []
            cut = self.commit_t if self.commit_t > self.buf_t0 else self.buf_t0 + dur - 1.0
        if cut is not None:
            k = int((cut - self.buf_t0) * SR)
            self.buf = self.buf[k:]
            self.buf_t0 = cut

    def _publish(self):
        start = max(0, len(self.committed) - 120)
        toks = [(w, True, i in self.breaks) for i, (_, _, w) in enumerate(self.committed[start:], start)]
        nb = len(self.committed) in self.breaks
        toks += [(w, False, nb and j == 0) for j, (_, _, w) in enumerate(self.prev)]
        self.on_update(toks)

    def _drop_silence(self):
        self.buf = self.buf[-SR // 2:]
        self.buf_t0 = self.total - len(self.buf) / SR

    def run(self, stop: threading.Event):
        if self.model is None:
            self.load()
        self.status = "listening"
        self._publish()
        self.t_start = time.perf_counter()
        pending = 0.0
        while not stop.is_set():
            src = self.src
            if self._reset_needed:
                self._reset_needed = False
                self.reset()
                self.t_start = time.perf_counter()
                pending = 0.0
            try:
                x = src.q.get(timeout=0.2)
            except queue.Empty:
                if self.status == "speech":
                    self.status = "listening"
                continue
            if x is None:
                break
            chunks = [x]
            while not src.q.empty():
                y = src.q.get_nowait()
                if y is None:
                    stop.set()
                    break
                chunks.append(y)
            x = np.concatenate(chunks)
            self.buf = np.concatenate([self.buf, x])
            self.total += len(x) / SR
            pending += len(x) / SR
            if pending < self.args.step:
                continue
            new = self.buf[-int(pending * SR):]
            pending = 0.0

            has_speech = bool(self._vad(np.concatenate([np.zeros(SR // 4, np.float32), new])))
            if not has_speech:
                self.status = "listening"
                self.silence += len(new) / SR
                if self.silence >= self.args.flush_silence and (self.prev or len(self.buf) > SR):
                    # speaker paused: finalise whatever is pending, start a new line next time
                    if self.prev:
                        hyp = self._hyp()
                        self._commit(hyp or self.prev)
                        self.prev = []
                    if self.committed:
                        self.breaks.add(len(self.committed))
                    self._drop_silence()
                    self._publish()
                    self.on_update(None)
                elif not self.prev and self.commit_t <= self.buf_t0:
                    self._drop_silence()
                continue
            self.silence = 0.0
            self.status = "speech"

            t = time.perf_counter()
            hyp = self._hyp()
            dt = time.perf_counter() - t
            n = 0
            while n < min(len(hyp), len(self.prev)) and norm(hyp[n][2]) == norm(self.prev[n][2]):
                n += 1
            self._commit(hyp[:n])
            self.prev = hyp[n:]
            if hyp:
                self.shown_lag.append(time.perf_counter() - self.t_start - hyp[-1][1])
            self._trim()
            self._publish()
            if self.args.verbose:
                print(f"[asr] {dt * 1000:4.0f} ms  buf {len(self.buf) / SR:4.1f}s  "
                      f"+{n} committed  | {' '.join(w for *_, w in self.prev)}", flush=True)
        if self.prev:
            self._commit(self.prev)
            self.prev = []
            self._publish()


# --------------------------------------------------------------------------- settings
SAVED_KEYS = ("x", "y", "win_w", "font_size", "lines", "opacity", "lang", "task", "source", "clear_after")


def load_settings() -> dict:
    try:
        return {k: v for k, v in json.loads(SETTINGS.read_text(encoding="utf-8")).items() if k in SAVED_KEYS}
    except Exception:
        return {}


def save_settings(d: dict):
    try:
        SETTINGS.write_text(json.dumps({k: d[k] for k in SAVED_KEYS if d.get(k) is not None}, indent=1),
                            encoding="utf-8")
    except Exception as e:
        print(f"[settings] save failed: {e}", flush=True)


# --------------------------------------------------------------------------- overlay
BG = "#141414"
FG_DONE = "#ffffff"
FG_TENT = "#9b9b9b"
FG_DIM = "#7a7a7a"
LANG_LABEL = {"tl": "Taglish", "en": "English", None: "Auto"}
EDGE = 10  # px from the left/right edge that resizes instead of moves


class Overlay:
    def __init__(self, args, streamer: Streamer):
        import tkinter as tk
        import tkinter.font as tkfont
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        self.tk = tk
        self.args = args
        self.st = streamer
        self.root = r = tk.Tk()
        r.title("Taglish Captions")
        r.overrideredirect(True)
        r.attributes("-topmost", True)
        r.attributes("-alpha", args.opacity)
        r.configure(bg=BG)
        self.font = tkfont.Font(family="Segoe UI", size=args.font_size)
        self.small = tkfont.Font(family="Segoe UI", size=10)
        self.padx, self.pady = 20, 10
        self.top = 12  # extra room above the text for the hover status line

        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        w = args.win_w or int(sw * args.width)
        h = self._height()
        x = args.x if args.x is not None else (sw - w) // 2
        y = args.y if args.y is not None else sh - h - int(sh * 0.08)
        x, y = max(0, min(x, sw - 100)), max(0, min(y, sh - 60))  # keep on screen
        r.geometry(f"{w}x{h}+{x}+{y}")

        self.text = tk.Text(r, bg=BG, fg=FG_DONE, bd=0, highlightthickness=0, wrap="none",
                            font=self.font, padx=self.padx, pady=self.pady, cursor="fleur",
                            selectbackground=BG, inactiveselectbackground=BG, takefocus=0)
        self.text.tag_configure("done", foreground=FG_DONE)
        self.text.tag_configure("tent", foreground=FG_TENT)
        self.text.tag_configure("dim", foreground=FG_DIM)
        self.text.pack(fill="both", expand=True, pady=(self.top, 0))
        self.text.configure(state="disabled")

        # hover bar: status + menu button
        self.status = tk.Label(r, bg=BG, fg=FG_DIM, font=self.small, cursor="arrow")
        self.menu_btn = tk.Label(r, text="☰", bg=BG, fg="#cfcfcf", font=self.small, cursor="hand2", padx=6)
        self.menu_btn.bind("<Button-1>", lambda e: self._show_menu(e.x_root, e.y_root) or "break")

        for wdg in (r, self.text, self.status):
            wdg.bind("<ButtonPress-1>", self._press)
            wdg.bind("<B1-Motion>", self._drag)
            wdg.bind("<ButtonRelease-1>", lambda e: self._save() or "break")
            wdg.bind("<Motion>", self._hover_cursor)
            wdg.bind("<Button-3>", lambda e: self._show_menu(e.x_root, e.y_root) or "break")
            wdg.bind("<Double-Button-1>", lambda e: "break")
            wdg.bind("<Control-MouseWheel>", self._wheel_font)
            wdg.bind("<Shift-MouseWheel>", self._wheel_alpha)
        r.bind("<Enter>", lambda e: self._hover(True))
        r.bind("<Leave>", lambda e: self._hover(False))

        self._build_menu()
        self.locked = False
        self.hidden = False
        self.tokens: list | None = []
        self.paused_at = None
        self.idle = False
        self._alpha = args.opacity
        self._shown = None
        self._save_job = None
        self._toast_until = 0.0
        self._toast_msg = ""
        self.on_quit = None
        self.watch = DeviceWatcher(lambda kind: r.after(0, self.on_devices_changed, kind))
        if not Path(args.source).is_file():
            self.watch.start()
        r.after(30, self._round_corners)
        r.after(50, self._tick)

    # ---- geometry helpers
    def _height(self):
        return self.font.metrics("linespace") * self.args.lines + 2 * self.pady + self.top + 4

    def _hwnd(self):
        return int(self.root.wm_frame(), 16)

    def _round_corners(self):
        try:  # Windows 11 native rounded corners (DWMWA_WINDOW_CORNER_PREFERENCE = 33, ROUND = 2)
            pref = ctypes.c_int(2)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(self._hwnd(), 33, ctypes.byref(pref), 4)
        except Exception:
            pass

    def _apply_size(self, w=None):
        """Resize to fit the current font/lines, keeping the bottom edge in place."""
        r = self.root
        w = w or r.winfo_width()
        h = self._height()
        bottom = r.winfo_y() + r.winfo_height()
        y = max(0, bottom - h)
        r.geometry(f"{w}x{h}+{r.winfo_x()}+{y}")
        self._shown = None
        self._save()

    # ---- public (thread-safe: only sets state; _tick renders)
    def update(self, tokens):
        if tokens is None:
            self.paused_at = time.time()
            return
        self.paused_at = None
        self.tokens = tokens

    # ---- render loop
    def _tick(self):
        toks = self.tokens
        if self.paused_at and time.time() - self.paused_at > self.args.clear_after:
            toks = []
        if not toks:
            msg = {"loading": "Loading model…", "error": "Error — see console"}.get(self.st.status, "")
            want_idle = not msg
            self._render(None, msg)
        else:
            want_idle = False
            self._render(toks)
        if want_idle != self.idle:
            self.idle = want_idle
            self._fade_to(self.args.opacity * (0.35 if want_idle else 1.0))
        self._update_status()
        self.root.after(50, self._tick)

    def _wrap(self, toks):
        """Greedy wrap by real pixel width; returns lines of (text, tag)."""
        maxw = self.root.winfo_width() - 2 * self.padx - 8
        sp = self.font.measure(" ")
        lines, cur, curw = [], [], 0
        for word, final, brk in toks:
            ww = self.font.measure(word)
            if cur and (brk or curw + sp + ww > maxw):
                lines.append(cur)
                cur, curw = [], 0
            cur.append((word, "done" if final else "tent"))
            curw += ww + (sp if len(cur) > 1 else 0)
        if cur:
            lines.append(cur)
        return lines[-self.args.lines:]

    def _render(self, toks, msg=""):
        lines = self._wrap(toks) if toks else [[(msg, "dim")]] if msg else []
        key = (tuple(tuple(l) for l in lines), self.root.winfo_width())
        if key == self._shown:
            return
        self._shown = key
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        for i, line in enumerate(lines):
            if i:
                t.insert("end", "\n")
            for j, (word, tag) in enumerate(line):
                t.insert("end", (" " if j else "") + word, tag)
        t.configure(state="disabled")

    def _fade_to(self, target, steps=6):
        start = self._alpha
        def step(i=1):
            a = start + (target - start) * i / steps
            self.root.attributes("-alpha", a)
            self._alpha = a
            if i < steps:
                self.root.after(25, step, i + 1)
        step()

    # ---- hover status bar
    def _hover(self, on):
        if on:
            self.status.place(relx=0, x=self.padx - 4, y=2, anchor="nw")
            self.menu_btn.place(relx=1, x=-6, y=2, anchor="ne")
            if self.idle:
                self._fade_to(self.args.opacity)
        else:
            x, y = self.root.winfo_pointerxy()
            if self.root.winfo_containing(x, y) is None:
                self.status.place_forget()
                self.menu_btn.place_forget()
                if self.idle:
                    self._fade_to(self.args.opacity * 0.35)

    def toast(self, msg, secs=2.5):
        self._toast_msg, self._toast_until = msg, time.time() + secs
        self.status.place(relx=0, x=self.padx - 4, y=2, anchor="nw")
        self.root.after(int(secs * 1000) + 50, lambda: self._hover(False))

    def _update_status(self):
        if time.time() < self._toast_until:
            self.status.configure(text=self._toast_msg, fg="#e8c66a")
            return
        dot = {"speech": "●", "listening": "○", "loading": "…", "error": "!"}.get(self.st.status, "")
        src = self.source_label()
        lang = LANG_LABEL.get(self.args.lang, self.args.lang)
        if self.args.task == "translate":
            lang += " → English"
        self.status.configure(text=f"{dot}  {src} · {lang}",
                              fg="#6fcf7f" if self.st.status == "speech" else FG_DIM)

    # ---- mouse
    def _edge(self, x):
        w = self.root.winfo_width()
        return "l" if x < EDGE else "r" if x > w - EDGE else None

    def _hover_cursor(self, e):
        x = e.x_root - self.root.winfo_x()
        e.widget.configure(cursor="sb_h_double_arrow" if self._edge(x) else "fleur") \
            if e.widget is not self.menu_btn else None

    def _press(self, e):
        r = self.root
        self._mode = self._edge(e.x_root - r.winfo_x())
        self._ox, self._oy = e.x_root - r.winfo_x(), e.y_root - r.winfo_y()
        self._gx, self._gw = r.winfo_x(), r.winfo_width()
        self._px = e.x_root
        return "break"

    def _drag(self, e):
        r = self.root
        h = r.winfo_height()
        if self._mode == "r":
            w = max(300, self._gw + e.x_root - self._px)
            r.geometry(f"{w}x{h}+{self._gx}+{r.winfo_y()}")
        elif self._mode == "l":
            w = max(300, self._gw - (e.x_root - self._px))
            r.geometry(f"{w}x{h}+{self._gx + self._gw - w}+{r.winfo_y()}")
        else:
            r.geometry(f"+{e.x_root - self._ox}+{e.y_root - self._oy}")
        self._shown = None
        return "break"

    def _wheel_font(self, e):
        self.set_font(self.args.font_size + (1 if e.delta > 0 else -1))
        return "break"

    def _wheel_alpha(self, e):
        self.set_opacity(self.args.opacity + (0.05 if e.delta > 0 else -0.05))
        return "break"

    # ---- settings actions
    def set_font(self, size):
        self.args.font_size = max(10, min(72, size))
        self.font.configure(size=self.args.font_size)
        self._apply_size()
        self.toast(f"Text size {self.args.font_size}", 1.2)

    def set_lines(self, n):
        self.args.lines = n
        self._apply_size()

    def set_opacity(self, a):
        self.args.opacity = round(max(0.3, min(1.0, a)), 2)
        self._alpha = self.args.opacity * (0.35 if self.idle else 1.0)
        self.root.attributes("-alpha", self._alpha)
        self.toast(f"Opacity {int(self.args.opacity * 100)}%", 1.2)
        self._save()

    def source_label(self):
        s = self.args.source
        cur = self.st.src.name if self.st.src else ""
        if s == "system":
            return f"Default output · {cur}" if cur else "Default output"
        if s == "mic":
            return f"Default mic · {cur}" if cur else "Default mic"
        if s.startswith(("out:", "in:")):
            return s.split(":", 1)[1]
        return Path(s).name

    def set_source(self, source, force=False, quiet=False):
        """Switch audio input. source: system | mic | out:<name> | in:<name>."""
        if source == self.args.source and not force:
            return
        old = self.st.src
        if old:
            old.stop()  # must go first so PortAudio rescans devices
        try:
            src = AudioSource(source, None).start()
        except Exception as ex:
            self.toast(f"Couldn't open it: {ex}", 4)
            self.v_source.set(self.args.source)
            if old and not Path(old.source).is_file():
                try:
                    self.st.switch_source(AudioSource(old.source, None).start())
                except Exception as ex2:
                    print(f"[audio] could not reopen previous source: {ex2}", flush=True)
            return
        self.args.source = source
        self.args.device_name = None
        self.v_source.set(source)
        self.st.switch_source(src)
        if not quiet:
            self.toast(f"Listening to {src.name}")
        self._save()

    def on_devices_changed(self, kind):
        """From DeviceWatcher (via root.after): follow Windows defaults, handle unplugs."""
        s = self.args.source
        snap = self.watch.snapshot()
        if kind == "out" and s == "system":
            self.set_source("system", force=True)
        elif kind == "in" and s == "mic":
            self.set_source("mic", force=True)
        elif kind == "list" and s.startswith(("out:", "in:")):
            name = s.split(":", 1)[1]
            if name not in snap["outs" if s.startswith("out:") else "ins"]:
                fallback = "system" if s.startswith("out:") else "mic"
                self.set_source(fallback, force=True, quiet=True)
                self.toast(f"{name} disconnected · now {self.st.src.name if self.st.src else fallback}", 4)

    def set_lang(self, lang):
        self.args.lang = None if lang == "auto" else lang
        self._save()

    def set_translate(self):
        self.args.task = "translate" if self.v_translate.get() else "transcribe"
        self._save()

    def _save(self):
        if self._save_job:
            self.root.after_cancel(self._save_job)
        self._save_job = self.root.after(400, self._save_now)

    def _save_now(self):
        self._save_job = None
        r = self.root
        d = {k: getattr(self.args, k, None) for k in SAVED_KEYS}
        d.update(x=r.winfo_x(), y=r.winfo_y(), win_w=r.winfo_width(),
                 lang=self.args.lang or "auto",
                 source=self.args.source if self.args.source in ("system", "mic")
                 or self.args.source.startswith(("out:", "in:")) else None)
        save_settings(d)

    # ---- menu
    def _build_menu(self):
        tk = self.tk
        m = self.menu = tk.Menu(self.root, tearoff=0)
        self.v_source = tk.StringVar(value=self.args.source)
        self.v_lang = tk.StringVar(value=self.args.lang or "auto")
        self.v_translate = tk.BooleanVar(value=self.args.task == "translate")
        self.v_lines = tk.IntVar(value=self.args.lines)
        self.v_lock = tk.BooleanVar(value=False)

        self.src_menu = tk.Menu(m, tearoff=0)
        m.add_cascade(label="Listen to", menu=self.src_menu)

        lang = tk.Menu(m, tearoff=0)
        for val, lab in (("tl", "Taglish / Tagalog"), ("en", "English only"), ("auto", "Auto-detect")):
            lang.add_radiobutton(label=lab, variable=self.v_lang, value=val, command=lambda v=val: self.set_lang(v))
        lang.add_separator()
        lang.add_checkbutton(label="Translate to English", variable=self.v_translate, command=self.set_translate)
        m.add_cascade(label="Language", menu=lang)

        look = tk.Menu(m, tearoff=0)
        look.add_command(label="Larger text", accelerator="Ctrl+Scroll",
                         command=lambda: self.set_font(self.args.font_size + 2))
        look.add_command(label="Smaller text", command=lambda: self.set_font(self.args.font_size - 2))
        look.add_separator()
        for n in (1, 2, 3, 4):
            look.add_radiobutton(label=f"{n} line{'s' if n > 1 else ''}", variable=self.v_lines, value=n,
                                 command=lambda n=n: self.set_lines(n))
        look.add_separator()
        for a in (0.6, 0.75, 0.85, 1.0):
            look.add_command(label=f"Opacity {int(a * 100)}%", command=lambda a=a: self.set_opacity(a))
        look.add_separator()
        look.add_command(label="Reset position and size", command=self.reset_geometry)
        m.add_cascade(label="Appearance", menu=look)

        m.add_separator()
        m.add_command(label="Clear captions", command=self.st.clear)
        m.add_checkbutton(label="Lock (click-through)", accelerator="Ctrl+Alt+L",
                          variable=self.v_lock, command=self.toggle_lock)
        m.add_command(label="Hide", accelerator="Ctrl+Alt+H", command=self.toggle_hide)
        m.add_separator()
        m.add_command(label="Quit", accelerator="Ctrl+Alt+Q", command=self.quit)

    def _fill_sources(self):
        """Rebuild 'Listen to' from the current device list (devices come and go)."""
        m = self.src_menu
        m.delete(0, "end")
        snap = self.watch.snapshot()
        v = self.v_source
        v.set(self.args.source)
        m.add_command(label="What you hear (outputs)", state="disabled")
        od = snap["out_default"]
        m.add_radiobutton(label=f"Windows default{f'  ({od})' if od else ''}", variable=v, value="system",
                          command=lambda: self.set_source("system"))
        for n in snap["outs"]:
            m.add_radiobutton(label=n, variable=v, value=f"out:{n}", command=lambda n=n: self.set_source(f"out:{n}"))
        m.add_separator()
        m.add_command(label="Microphones", state="disabled")
        idf = snap["in_default"]
        m.add_radiobutton(label=f"Windows default{f'  ({idf})' if idf else ''}", variable=v, value="mic",
                          command=lambda: self.set_source("mic"))
        for n in snap["ins"]:
            m.add_radiobutton(label=n, variable=v, value=f"in:{n}", command=lambda n=n: self.set_source(f"in:{n}"))
        if not self.watch.ok:
            m.add_separator()
            m.add_command(label="(install pycaw to list devices)", state="disabled")

    def _show_menu(self, x, y):
        self.v_lang.set(self.args.lang or "auto")
        self._fill_sources()
        try:
            self.menu.tk_popup(x, y)
        finally:
            self.menu.grab_release()

    def reset_geometry(self):
        r = self.root
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        w, h = int(sw * 0.6), self._height()
        r.geometry(f"{w}x{h}+{(sw - w) // 2}+{sh - h - int(sh * 0.08)}")
        self._shown = None
        self._save()

    # ---- hotkey actions
    def toggle_lock(self):
        GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT = -20, 0x80000, 0x20
        u = ctypes.windll.user32
        h = self._hwnd()
        style = u.GetWindowLongW(h, GWL_EXSTYLE)
        self.locked = not self.locked
        self.v_lock.set(self.locked)
        style = (style | WS_EX_LAYERED | WS_EX_TRANSPARENT) if self.locked else (style & ~WS_EX_TRANSPARENT)
        u.SetWindowLongW(h, GWL_EXSTYLE, style)
        self.toast("Locked · Ctrl+Alt+L to unlock" if self.locked else "Unlocked · drag to move")

    def toggle_hide(self):
        self.hidden = not self.hidden
        (self.root.withdraw if self.hidden else self.root.deiconify)()
        if not self.hidden:
            self.root.after(30, self._round_corners)

    def quit(self):
        if self._save_job:
            self.root.after_cancel(self._save_job)
            self._save_now()
        if self.on_quit:
            self.on_quit()
        self.root.after(0, self.root.destroy)


def hotkey_thread(overlay: Overlay):
    u = ctypes.windll.user32
    MOD = 0x0001 | 0x0002 | 0x4000  # ALT | CTRL | NOREPEAT
    keys = {1: ("L", overlay.toggle_lock), 2: ("H", overlay.toggle_hide), 3: ("Q", overlay.quit)}
    for i, (k, _) in keys.items():
        if not u.RegisterHotKey(None, i, MOD, ord(k)):
            print(f"[ui] could not register Ctrl+Alt+{k} (in use?)", flush=True)
    msg = wintypes.MSG()
    while u.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
        if msg.message == 0x0312 and msg.wParam in keys:  # WM_HOTKEY
            overlay.root.after(0, keys[msg.wParam][1])


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="system",
                    help="system (Windows default output) | mic (default mic) | out:<device> | in:<device> | "
                         "path\\to\\file.wav (test, real-time). See --list-devices")
    ap.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    ap.add_argument("--device-name", help="substring of an audio device name to use instead of the default")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--lang", default="tl", help="tl works best for Taglish; 'en' for pure English; 'auto' to detect")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--step", type=float, default=0.4,
                    help="seconds of new audio between passes (lower = snappier, more GPU)")
    ap.add_argument("--max-buffer", type=float, default=12.0)
    ap.add_argument("--flush-silence", type=float, default=0.8, help="pause length that finalises a line")
    ap.add_argument("--clear-after", type=float, default=6.0, help="clear captions after this much silence")
    ap.add_argument("--words", default=str(HERE / "words.txt"), help="prompt file: example Taglish + custom words")
    ap.add_argument("--font-size", type=int, default=22)
    ap.add_argument("--lines", type=int, default=2)
    ap.add_argument("--width", type=float, default=0.6, help="overlay width as a fraction of screen width")
    ap.add_argument("--opacity", type=float, default=0.85)
    ap.add_argument("--log", help="append finalised captions to this file")
    ap.add_argument("--no-overlay", action="store_true", help="print to console only")
    ap.add_argument("--fresh", action="store_true", help="ignore saved settings.json")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    ap.set_defaults(x=None, y=None, win_w=None)
    if "--fresh" not in os.sys.argv:
        ap.set_defaults(**load_settings())   # saved settings < command-line flags
    args = ap.parse_args()
    if args.list_devices:
        w = DeviceWatcher(lambda k: None)
        w.start()
        for _ in range(30):
            time.sleep(0.1)
            if w.snapshot()["outs"] or not w.ok:
                break
        s = w.snapshot()
        print(f"Outputs (default: {s['out_default']}):")
        print("\n".join(f'  --source "out:{n}"' for n in s["outs"]))
        print(f"Microphones (default: {s['in_default']}):")
        print("\n".join(f'  --source "in:{n}"' for n in s["ins"]))
        return
    if args.lang == "auto":
        args.lang = None
    p = Path(args.words)
    args.prompt = " ".join(l.strip() for l in p.read_text(encoding="utf-8").splitlines()
                           if l.strip() and not l.startswith("#")) if p.is_file() else ""

    stop = threading.Event()
    printed = [0]
    holder: dict = {}

    def on_update(tokens):
        if "ov" in holder:
            holder["ov"].update(tokens)
        elif tokens and len(streamer.committed) > printed[0]:
            print("»", " ".join(w for *_, w in streamer.committed[printed[0]:]), flush=True)
            printed[0] = len(streamer.committed)

    streamer = Streamer(args, on_update)
    overlay = None if args.no_overlay else Overlay(args, streamer)
    if overlay:
        holder["ov"] = overlay

    def worker():
        try:
            streamer.src = AudioSource(args.source, args.device_name)
            streamer.load()
            streamer.src.start()
            streamer.run(stop)
        except Exception as e:
            streamer.status = "error"
            print(f"[error] {e!r}", flush=True)
            raise
        finally:
            if streamer.lat:
                l = np.array(streamer.lat)
                print(f"[latency] word end -> committed: median {np.median(l):.2f}s, "
                      f"p90 {np.percentile(l, 90):.2f}s over {len(l)} words", flush=True)
            if streamer.shown_lag:
                l = np.array(streamer.shown_lag)
                print(f"[latency] newest word on screen (grey): median {np.median(l):.2f}s, "
                      f"p90 {np.percentile(l, 90):.2f}s", flush=True)
            if overlay and Path(args.source).is_file():
                overlay.root.after(4000, overlay.quit)

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    if overlay:
        overlay.on_quit = stop.set
        threading.Thread(target=hotkey_thread, args=(overlay,), daemon=True).start()
        print("[ui] drag to move · edges to resize · right-click for menu · Ctrl+Alt+L lock · "
              "Ctrl+Alt+H hide · Ctrl+Alt+Q quit", flush=True)
        overlay.root.mainloop()
        stop.set()
    else:
        try:
            while th.is_alive():
                th.join(0.5)
        except KeyboardInterrupt:
            stop.set()
    th.join(timeout=5)
    if streamer.src:
        streamer.src.stop()


if __name__ == "__main__":
    main()
