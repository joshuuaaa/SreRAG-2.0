from src.llm.backends import MockLLMBackend


def test_mock_burn_query_mentions_cool_water():
    backend = MockLLMBackend({})
    out = backend.generate("User query: I burned my hand with boiling water")
    assert "cool" in out.lower() or "running water" in out.lower()
    assert "direct pressure" not in out.lower()
