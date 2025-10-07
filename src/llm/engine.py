"""LLM Engine using llama-cpp-python for offline inference"""
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Local LLM engine backed by llama-cpp-python.
    Loads a GGUF model and provides simple generate() and chat() helpers.
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: dict with keys like:
                - model_path: str, path to GGUF model (relative to repo root or absolute)
                - n_ctx: int, context length (e.g., 2048 or 4096 for Phi-3)
                - n_threads: int, CPU threads to use
                - n_gpu_layers: int, offload layers to GPU (0 on Raspberry Pi)
                - temperature: float
                - max_tokens: int
                - top_p: float
                - repeat_penalty: float
        """
        self.config = config or {}
        self.model: Optional[Llama] = None

        # Load config with sane defaults
        self.model_path = self.config.get(
            "model_path",
            "models/llm/Phi-3-mini-4k-instruct-q4.gguf",
        )
        self.n_ctx = int(self.config.get("n_ctx", 2048))
        self.n_threads = int(self.config.get("n_threads", max(1, (os.cpu_count() or 4) // 2)))
        self.n_gpu_layers = int(self.config.get("n_gpu_layers", 0))
        self.temperature = float(self.config.get("temperature", 0.3))
        self.max_tokens = int(self.config.get("max_tokens", 300))
        self.top_p = float(self.config.get("top_p", 0.9))
        self.repeat_penalty = float(self.config.get("repeat_penalty", 1.1))

        # Reasonable default stop tokens for Phi-3 style chat
        self.stop_tokens: List[str] = self.config.get(
            "stop_tokens",
            ["<|end|>", "<|user|>", "</s>"],
        )

        # Normalize path: if relative, make absolute from repo root
        if not os.path.isabs(self.model_path):
            project_root = Path(__file__).resolve().parents[2]  # .../project
            self.model_path = str((project_root / self.model_path).resolve())

        self._load_model()

    def _load_model(self) -> None:
        """Load the quantized GGUF model with llama-cpp."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            size_gb = os.path.getsize(self.model_path) / (1024**3)
            print(f"🔄 Loading LLM from {self.model_path}...")
            print(f"   File size: {size_gb:.2f} GB")

            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )

            # Helpful note if user is running with less than train context
            if self.n_ctx < 4096:
                # llama.cpp may also print a warning itself; this is just a friendly hint.
                logger.debug(
                    "Configured n_ctx (%d) is less than typical train context (4096). "
                    "You can raise n_ctx to 4096 if memory allows for better performance.",
                    self.n_ctx,
                )

            print("✅ LLM loaded successfully")

        except Exception as e:
            raise RuntimeError(
                "Failed to load LLM model: {err}\nModel path: {path}\n"
                "Make sure this is a valid GGUF model and your llama-cpp-python build matches it."
                .format(err=str(e), path=self.model_path)
            ) from e

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text given a raw prompt.

        Args:
            prompt: input text
            max_tokens: optional override for max new tokens
            temperature: optional override for temperature
            stop: optional additional stop tokens

        Returns:
            Cleaned string response
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        max_new = max_tokens if max_tokens is not None else self.max_tokens
        temp = temperature if temperature is not None else self.temperature
        stop_list = list(self.stop_tokens)
        if stop:
            # append user-provided stop tokens
            stop_list.extend(s for s in stop if s not in stop_list)

        try:
            resp = self.model(
                prompt,
                max_tokens=max_new,
                temperature=temp,
                top_p=self.top_p,
                repeat_penalty=self.repeat_penalty,
                stop=stop_list,
                echo=False,
            )
            raw = resp["choices"][0]["text"]

            return self._clean_output(raw)

        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}") from e

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Chat-style generation using Phi-3 style tags.

        Args:
            messages: list of {"role": "system" | "user" | "assistant", "content": str}
            system_prompt: optional system override
            max_tokens: optional max new tokens
            temperature: optional temperature

        Returns:
            Assistant response as string
        """
        prompt = self._build_chat_prompt(messages, system_prompt)
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def _build_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build a Phi-3 style prompt:
        <|system|> ... <|end|>
        <|user|> ... <|end|>
        <|assistant|> ...
        """
        # Default system prompt if not provided in messages
        sys_text = system_prompt
        if sys_text is None:
            # Check if provided within messages as a system role
            for m in messages:
                if m.get("role") == "system":
                    sys_text = m.get("content", "")
                    break
        if sys_text is None:
            sys_text = "You are a helpful medical crisis assistant. Be concise and safe."

        parts = [f"<|system|>\n{sys_text}<|end|>"]

        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                parts.append(f"<|user|>\n{content}<|end|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}<|end|>")
            # Skip system since we already added one above

        # Model will complete the assistant turn
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    @staticmethod
    def _clean_output(text: str) -> str:
        """Remove special tags and trim to the first end marker if present."""
        if not text:
            return ""

        # Remove common special tags
        for tag in ("<|assistant|>", "<|user|>", "<|system|>"):
            text = text.replace(tag, " ")

        # Stop at first end token if present
        for end_tok in ("<|end|>", "</s>"):
            if end_tok in text:
                text = text.split(end_tok, 1)[0]

        # Normalize whitespace
        return " ".join(text.strip().split())

    def __del__(self):
        # Explicitly free model on teardown
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
        except Exception:
            pass