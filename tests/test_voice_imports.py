def test_voice_modules_importable():
    # Smoke test: modules should import (even if external binaries aren't present).
    import src.audio.stt  # noqa: F401
    import src.audio.tts  # noqa: F401
