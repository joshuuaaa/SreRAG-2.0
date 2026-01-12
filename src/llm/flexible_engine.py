"""Flexible LLM Engine with multiple backend support"""

import os
from typing import Dict, Any, Optional
from .backends import create_llm_backend, LLMBackend

class FlexibleLLMEngine:
    """
    Flexible LLM engine that can use multiple backends:
    - Mock LLM (for testing)
    - Ollama (local)
    - OpenAI (cloud)
    - Anthropic (cloud)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend: Optional[LLMBackend] = None
        self.max_tokens = config.get("max_tokens", 150)
        
        # Try to initialize the configured backend
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the LLM backend with fallback options"""
        
        # Get primary backend from config
        primary_backend = self.config.get("backend", "mock")
        
        # Define fallback order
        fallback_order = [primary_backend]
        if primary_backend != "mock":
            fallback_order.append("mock")
        
        # Try each backend in order
        for backend_type in fallback_order:
            try:
                print(f"🔄 Trying {backend_type} backend...")
                
                backend_config = {
                    "backend": backend_type,
                    "config": self.config.get(f"{backend_type}_config", {})
                }
                
                self.backend = create_llm_backend(backend_config)
                print(f"✅ Successfully initialized {backend_type} backend")
                return
                
            except Exception as e:
                print(f"❌ {backend_type} backend failed: {e}")
                continue
        
        raise RuntimeError("All LLM backends failed to initialize")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text using the active backend"""
        
        if not self.backend:
            raise RuntimeError("No LLM backend available")
        
        # Use provided max_tokens or default
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            return self.backend.generate(prompt, max_tokens=tokens, **kwargs)
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")
    
    def chat(self, messages, **kwargs) -> str:
        """Chat interface - converts to simple prompt"""
        
        # Convert chat messages to simple prompt
        if isinstance(messages, list):
            # Extract last user message
            user_messages = [msg["content"] for msg in messages if msg.get("role") == "user"]
            prompt = user_messages[-1] if user_messages else ""
        else:
            prompt = str(messages)
        
        return self.generate(prompt, **kwargs)