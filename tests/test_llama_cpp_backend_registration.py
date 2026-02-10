from src.llm.backends import create_llm_backend


def test_llama_cpp_backend_is_registered():
    import os

    # Ensure we can create the backend without importing llama-cpp-python.
    os.environ["CRISIS_ASSISTANT_SKIP_LLAMA_CPP"] = "1"
    backend = create_llm_backend({"backend": "llama_cpp", "config": {}})
    assert backend.__class__.__name__ == "LlamaCppBackend"
