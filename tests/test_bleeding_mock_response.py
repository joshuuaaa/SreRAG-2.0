from src.llm.backends import MockLLMBackend


def test_mock_bleeding_query_mentions_pressure():
    backend = MockLLMBackend({})
    out = backend.generate("User query: My hand is bleeding")
    assert "pressure" in out.lower() or "bleeding" in out.lower()
    assert "splint" not in out.lower()
