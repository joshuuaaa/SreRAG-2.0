from src.llm.backends import LlamaCppBackend


def test_llama_cpp_postprocess_strips_leading_response_label():
    assert (
        LlamaCppBackend._postprocess("Response: Hello there")
        == "Hello there"
    )


def test_llama_cpp_postprocess_truncates_repeated_response_sections():
    text = "Response: first part. Response: second part that should be removed"
    out = LlamaCppBackend._postprocess(text)
    assert "second part" not in out
    assert out.startswith("first part")
