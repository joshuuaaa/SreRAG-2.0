import pytest

from src.llm.flexible_engine import FlexibleLLMEngine


def test_fallback_to_mock_when_primary_backend_fails():
    # Use a guaranteed-invalid URL so Ollama can't possibly be reached.
    cfg = {
        "backend": "ollama",
        "fallback_to_mock": True,
        "max_tokens": 10,
        "ollama_config": {
            "base_url": "http://127.0.0.1:1",
            "model": "llama3.2:3b",
            "connect_timeout": 0.2,
            "read_timeout": 0.2,
        },
    }

    engine = FlexibleLLMEngine(cfg)
    # Should have fallen back to MockLLMBackend
    assert engine.backend is not None
    assert engine.backend.__class__.__name__ == "MockLLMBackend"


def test_no_fallback_raises_when_primary_backend_fails():
    cfg = {
        "backend": "ollama",
        "fallback_to_mock": False,
        "ollama_config": {
            "base_url": "http://127.0.0.1:1",
            "connect_timeout": 0.2,
            "read_timeout": 0.2,
        },
    }

    with pytest.raises(RuntimeError):
        FlexibleLLMEngine(cfg)
