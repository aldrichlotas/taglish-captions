# Taglish Captions

Live subtitles for anything playing on your Windows PC, like Windows Live Captions but with Tagalog/English code-switching.
Runs Whisper `large-v3-turbo` on the GPU through faster-whisper.

## Install

Windows 10/11 and Python 3.10 or newer. An NVIDIA GPU is recommended; without CUDA it falls back to the `small` model on the CPU, which is slower and less accurate.

```
py -m pip install -r requirements.txt
```

On the GPU, faster-whisper needs NVIDIA's CUDA 12 cuBLAS and cuDNN 9 libraries. If a CUDA build of PyTorch is installed, the script reuses the DLLs that come with it. Otherwise follow the [faster-whisper GPU notes](https://github.com/SYSTRAN/faster-whisper#gpu). The model (about 1.6 GB) downloads on the first run.

## Run

Double-click `run.bat`, or:

```
py -3.13 -X utf8 captions.py
```

A caption bar appears near the bottom of the screen. White words are final; grey words may still change.
A pause in speech starts a new line. When nobody is talking, the bar fades out.

| Action | How |
|---|---|
| Move the bar | Drag it |
| Make it wider or narrower | Drag its left or right edge |
| Text size | Ctrl+scroll over the bar |
| Opacity | Shift+scroll over the bar |
| Menu (audio source, language, translate, lines, opacity, clear, reset position) | Right-click, or the ☰ that appears on hover |
| Lock it (clicks pass through) | Ctrl+Alt+L (press again to unlock) |
| Hide / show | Ctrl+Alt+H |
| Quit | Ctrl+Alt+Q, or Quit in the menu |

Hovering also shows what it is listening to and the language (● green while it hears speech).
Position, size and menu choices are saved to `settings.json` and restored next time
(flags on the command line still win; `--fresh` ignores the saved file).

## Choosing the audio device

Right-click → **Listen to** lists every output (speakers, headphones, virtual cables) and every microphone.

- **Windows default** (the default choice) follows Windows: plug in headphones or switch the output in the
  taskbar and the captions move with it within about 2 seconds.
- **A specific device** stays on that device even if Windows switches. If it gets unplugged, the app falls
  back to the Windows default and tells you.

From the command line: `py -3.13 captions.py --list-devices`, then for example
`run.bat --source "out:Speakers (Realtek(R) Audio)"` or `--source "in:Microphone Array (Realtek(R) Audio)"`.

Needs `pycaw` (`py -3.13 -m pip install pycaw`) to list devices and notice changes.

## Useful options

| Option | Default | What it does |
|---|---|---|
| `--source` | `system` | `system` (Windows default output), `mic`, `out:<device>`, `in:<device>`, or a `.wav` path (test mode) |
| `--step` | `0.4` | Seconds between passes. `0.6` uses less GPU (good while gaming) but lags ~1 s more |
| `--lang` | `tl` | `tl` works best for Taglish. `en` for English-only content, `auto` to detect |
| `--task translate` | off | Output English instead of the original words |
| `--font-size`, `--lines`, `--width`, `--opacity` | 22, 2, 0.6, 0.85 | Look of the bar |
| `--clear-after` | `6` | Seconds of silence before the captions clear |
| `--log captions.txt` | off | Saves the finished lines with timestamps |
| `--no-overlay -v` | off | Prints to the console instead, with timing for each pass |

## Custom words

Edit `words.txt`. Its text goes to Whisper as a prompt before every pass, so put a sample Taglish
sentence and any names, slang or game terms you want spelled right there. Keep it short.

## Measured on this laptop (RTX 4050, 45 s English speech clip)

- Grey (newest) text on screen: about 1.3 s behind speech (p90 1.6 s)
- White (final) text: about 1.8 s behind (p90 2.1 s)
- Each Whisper pass: about 0.3 s
