"""Speech-to-text (offline-first) helpers.

Approach:
- Record audio from microphone (sounddevice) into a temporary WAV.
- Transcribe using whisper.cpp binary (`whisper-cli` or `main`) against a ggml model.

We do not vendor whisper.cpp here; users can install/build it separately.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass
class STTConfig:
    backend: str = "whispercpp"  # whispercpp
    whisper_bin: str = "whisper-cli"  # common name; some builds use `main`
    model_path: str = "models/whisper/ggml-tiny.en.bin"
    language: str = "en"
    sample_rate: int = 16000
    record_seconds: float = 6.0


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def record_wav(sample_rate: int, seconds: float) -> str:
    """Record from default microphone and return path to WAV file."""

    frames = int(sample_rate * seconds)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    audio = np.squeeze(audio)

    # Convert float32 [-1,1] to int16 for wave
    int16 = np.clip(audio, -1.0, 1.0)
    int16 = (int16 * 32767).astype(np.int16)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, int16, sample_rate, subtype="PCM_16")
    return path


def transcribe(cfg: STTConfig, wav_path: str) -> str:
    """Transcribe wav_path into text using whisper.cpp."""

    backend = (cfg.backend or "whispercpp").lower()
    if backend != "whispercpp":
        raise ValueError(f"Unsupported STT backend: {backend}")

    whisper_bin = cfg.whisper_bin
    if not os.path.isabs(whisper_bin):
        resolved = _which(whisper_bin) or _which("main")
        if not resolved:
            raise RuntimeError(
                "whisper.cpp binary not found. Install/build whisper.cpp and ensure `whisper-cli` (or `main`) is on PATH."
            )
        whisper_bin = resolved

    model_path = cfg.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        raise RuntimeError(f"Whisper model not found: {model_path}")

    # Use a temp output txt to avoid parsing noisy logs.
    out_txt = tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name

    try:
        # whisper.cpp CLI flags differ slightly across builds; these work for common `main` + `whisper-cli`.
        cmd = [
            whisper_bin,
            "-m",
            model_path,
            "-f",
            wav_path,
            "-l",
            cfg.language,
            "-otxt",
            "-of",
            out_txt[:-4],
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        if os.path.exists(out_txt):
            with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()

        # Some builds output `${-of}.txt`
        alt = out_txt
        if os.path.exists(alt):
            with open(alt, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()

        return ""
    finally:
        try:
            os.remove(out_txt)
        except OSError:
            pass
        try:
            os.remove(out_txt[:-4] + ".txt")
        except OSError:
            pass


def listen_and_transcribe(cfg: STTConfig) -> str:
    """Convenience: record audio then transcribe."""

    wav = record_wav(cfg.sample_rate, cfg.record_seconds)
    try:
        return transcribe(cfg, wav)
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass
