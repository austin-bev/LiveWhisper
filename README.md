# LiveWhisperGUI - Whisper based transcription and translation

Produces live sentence-to-sentence transcription and translation to a DearPyGUI GUI. Powered by [faster-whisper](https://github.com/guillaumekln/faster-whisper)

Using [OpenAI's Whisper](https://github.com/openai/whisper) model, and sounddevice library to listen to microphone.
Audio from mic is stored if it hits a volume & frequency threshold, then when silence is detected, it saves the audio to a temp file and sends it to Whisper.

*Dependencies:* Whisper, faster-whisper, numpy, scipy, sounddevice, dearpygui

This implementation is forked from [Nikorasu/LiveWhisper](https://github.com/Nikorasu/LiveWhisper), and the code for recording audio remains largely the same.

Also this will not work on Python 3.11. I have tested it working on 3.9 (3.10 should work), but Whisper is not compatible with 3.11.

---

## Usage
First run `whisperserver.py`, which will be responsible for interacting with the model.
After, run `whisperclient.py` and begin speaking. If the frequecy on the GUI is not changing, it is likely the sounddevice library hasn't picked up your microphone.

