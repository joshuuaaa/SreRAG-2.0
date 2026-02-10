from src.llm.backends import MockLLMBackend


def test_mock_fracture_query_mentions_splint():
    backend = MockLLMBackend({})
    out = backend.generate("User query: I think my friend broke their wrist. What should I do?")
    assert "splint" in out.lower() or "immobil" in out.lower()
    assert "direct pressure" not in out.lower()
