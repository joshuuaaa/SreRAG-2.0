"""Text-to-speech (offline-first) helpers.

Design goals:
- Work offline.
- Prefer Piper if available (local binary + .onnx voice).
- Fall back gracefully if Piper isn't installed.

We intentionally execute TTS via subprocess so users can install piper as a standalone binary
without forcing heavyweight Python deps.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class TTSConfig:
    backend: str = "piper"  # piper | none
    piper_bin: str = "piper"  # resolved via PATH if not absolute
    voice_model: str = ""  # path to piper .onnx
    speaker_id: Optional[int] = None
    length_scale: Optional[float] = None
    noise_scale: Optional[float] = None
    noise_w: Optional[float] = None


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def speak(text: str, cfg: TTSConfig) -> None:
    """Speak text out loud.

    If TTS isn't available (or disabled), this is a no-op.
    """

    if not text or not text.strip():
        return

    backend = (cfg.backend or "none").lower()
    if backend in ("off", "none", "disabled"):
        return

    if backend != "piper":
        # Future: add other backends.
        return

    piper_bin = cfg.piper_bin
    if not os.path.isabs(piper_bin):
        resolved = _which(piper_bin)
        if not resolved:
            # Piper isn't installed; silently skip.
            return
        piper_bin = resolved

    if not cfg.voice_model:
        # No voice model configured; skip.
        return

    voice_model_path = cfg.voice_model
    if not os.path.isabs(voice_model_path):
        # Resolve relative to repo root (cwd for main.py is repo root)
        voice_model_path = os.path.abspath(voice_model_path)

    if not os.path.exists(voice_model_path):
        return

    # Piper reads stdin and outputs wav to a file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wavf:
        wav_path = wavf.name

    try:
        cmd = [
            piper_bin,
            "--model",
            voice_model_path,
            "--output_file",
            wav_path,
        ]

        if cfg.speaker_id is not None:
            cmd += ["--speaker", str(cfg.speaker_id)]
        if cfg.length_scale is not None:
            cmd += ["--length_scale", str(cfg.length_scale)]
        if cfg.noise_scale is not None:
            cmd += ["--noise_scale", str(cfg.noise_scale)]
        if cfg.noise_w is not None:
            cmd += ["--noise_w", str(cfg.noise_w)]

        # Run Piper to synthesize
        proc = subprocess.run(
            cmd,
            input=text.strip().encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            # Keep this non-fatal for the main app, but make it discoverable in logs.
            raise RuntimeError(
                "Piper TTS failed (exit={code}). stderr:\n{stderr}\n\n"
                "Common fix: ensure the model config exists at <voice>.onnx.json, "
                "or pass --config <path>.".format(code=proc.returncode, stderr=stderr)
            )

        # Play wav. Prefer aplay/paplay if present.
        player = _which("paplay") or _which("aplay")
        if not player:
            return

        subprocess.run(
            [player, wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
