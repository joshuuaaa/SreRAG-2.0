"""Alternative LLM backends for the crisis assistant"""

import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 150, **kwargs) -> str:
        """Generate text from prompt"""
        pass

class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend"""
    
    def __init__(self, config: Dict[str, Any]):
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY")
            )
            self.model = config.get("model", "gpt-3.5-turbo")
            self.temperature = config.get("temperature", 0.3)
            print(f"✅ OpenAI backend loaded (model: {self.model})")
        except ImportError:
            raise RuntimeError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI initialization failed: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 150, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")

class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend"""
    
    def __init__(self, config: Dict[str, Any]):
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            )
            self.model = config.get("model", "claude-3-haiku-20240307")
            self.temperature = config.get("temperature", 0.3)
            print(f"✅ Anthropic backend loaded (model: {self.model})")
        except ImportError:
            raise RuntimeError("Anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Anthropic initialization failed: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 150, **kwargs) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"Anthropic generation failed: {e}")

class OllamaBackend(LLMBackend):
    """Ollama local backend"""
    
    def __init__(self, config: Dict[str, Any]):
        try:
            import requests
            self.base_url = config.get("base_url", "http://localhost:11434")
            self.model = config.get("model", "llama3.2:3b")
            self.temperature = config.get("temperature", 0.3)
            self.connect_timeout = float(config.get("connect_timeout", 5))
            self.read_timeout = float(config.get("read_timeout", 60))
            
            # Test connection
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=(self.connect_timeout, self.connect_timeout),
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama server not responding (status {response.status_code})."
                )
            
            models = response.json().get("models", [])
            available_models = [m["name"] for m in models]
            
            if self.model not in available_models:
                print(f"⚠️  Model {self.model} not found. Available: {available_models}")
                if available_models:
                    self.model = available_models[0]
                    print(f"🔄 Using {self.model} instead")
            
            print(f"✅ Ollama backend loaded (model: {self.model})")
            
        except ImportError:
            raise RuntimeError("Requests package required for Ollama backend")
        except Exception as e:
            raise RuntimeError(
                "Ollama initialization failed. "
                f"Base URL: {config.get('base_url', 'http://localhost:11434')}\n"
                f"Reason: {e}\n"
                "Tips: install/start Ollama, verify it’s running, and pull a model."
            )
    
    def generate(self, prompt: str, max_tokens: int = 150, **kwargs) -> str:
        try:
            import requests
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(self.connect_timeout, self.read_timeout),
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")


class LlamaCppBackend(LLMBackend):
    """Fully-offline GGUF backend via llama-cpp-python.

    This wraps the existing `src.llm.engine.LLMEngine` which loads a local GGUF model
    from disk. No network calls.
    """

    def __init__(self, config: Dict[str, Any]):
        try:
            # Unit tests (and some environments) may not have llama-cpp-python available.
            # Allow skipping initialization so we can still test factory wiring.
            if os.getenv("CRISIS_ASSISTANT_SKIP_LLAMA_CPP", "").strip() in {"1", "true", "yes"}:
                self.engine = None
                print("✅ llama-cpp backend registered (initialization skipped)")
                return

            from .engine import LLMEngine

            # LLMEngine already supports model_path/n_ctx/n_threads/n_gpu_layers/etc.
            self.engine = LLMEngine(config or {})
            print("✅ llama-cpp backend loaded (GGUF offline)")
        except Exception as e:
            raise RuntimeError(f"llama-cpp initialization failed: {e}")

    def generate(self, prompt: str, max_tokens: int = 150, **kwargs) -> str:
        try:
            if self.engine is None:
                raise RuntimeError(
                    "llama-cpp backend initialization was skipped (CRISIS_ASSISTANT_SKIP_LLAMA_CPP=1)."
                )
            temperature = kwargs.get("temperature")
            text = self.engine.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            return self._postprocess(text)
        except Exception as e:
            raise RuntimeError(f"llama-cpp generation failed: {e}")

    @staticmethod
    def _postprocess(text: str) -> str:
        """Clean up common instruct-model artifacts.

        Some models echo labels like 'Response:' or repeat them. We keep this conservative
        so we don't accidentally remove legitimate content.
        """
        if not text:
            return ""

        s = text.strip()

        # If the model outputs multiple 'Response:' segments, keep only the first.
        lower = s.lower()
        marker = "response:"
        if lower.count(marker) >= 2:
            first = lower.find(marker)
            second = lower.find(marker, first + len(marker))
            if second != -1:
                s = s[:second].strip()
                lower = s.lower()

        # If the output starts with 'Response:' strip leading label.
        if lower.startswith(marker):
            s = s[len(marker):].strip()

        return s

class MockLLMBackend(LLMBackend):
    """Mock LLM for testing with humanoid conversational responses"""
    
    def __init__(self, config: Dict[str, Any]):
        print("✅ Mock LLM backend loaded (humanoid conversation mode)")
        self.responses = {
            "burn": (
                "I’ve got you — burns are scary, but we can do the right first steps right now. "
                "First, stop the burning source and move them away from heat. If clothing is hot or wet, remove it only if it’s not stuck to the skin. "
                "Then cool the burn with cool running water for 10–20 minutes — not ice. "
                "Take off rings or tight items near the burn before swelling starts. "
                "After cooling, cover the area with a clean, non-stick dressing (or loosely with clean plastic wrap). Don’t pop any blisters. "
                "If the burn is large, they have trouble breathing — call emergency services now. "
                "Tell me where the burn is and about how big it is (roughly palm-sized or bigger?) and I’ll guide you next."
            ),
            "fracture": (
                "Okay — I’ve got you. A suspected wrist fracture can be really painful, but we can stabilize it safely right now. "
                "First, have them stop using the hand and keep it as still as possible. "
                "If there’s swelling, remove rings/watches right away. "
                "Next, splint it in the position you found it: pad around the wrist/forearm (a towel or clothing works), "
                "then use something rigid like cardboard or a folded magazine along the forearm and secure it with cloth strips/bandage — snug, not tight. "
                "After you tie it, check fingers: are they warm/pink, can they wiggle them, and do they feel normal? If fingers turn cold/blue or go numb, loosen the wrap and get urgent help. "
                "Use an ice pack wrapped in cloth for 15–20 minutes to help with swelling, and keep the hand elevated if it doesn’t increase pain. "
                "If you see bone, a deep wound, severe deformity, or there’s numbness/weakness, call emergency services now."
            ),
            "splint": (
                "I can help you splint this safely. The goal is to STOP movement and protect circulation. "
                "Support the limb, pad around the injury, place a rigid support (cardboard/stick/magazine) so it spans the joint above and below if possible, "
                "then tie it in place above and below the injury — not over the most painful spot. "
                "Re-check fingers/toes for color, warmth, feeling, and movement after splinting. If any of those get worse, loosen the ties and get urgent help."
            ),
            "bleeding": (
                "I understand this is really scary right now, but I'm here to help you through this step by step. "
                "First thing we need to do is apply direct pressure - grab a clean cloth or towel and press it firmly over the wound. "
                "You're doing great. Now, if possible, let's elevate that arm above their heart level - this helps slow the bleeding. "
                "Keep that pressure steady for about 10-15 minutes without peeking. I know it feels like forever, but you've got this. "
                "If the bleeding doesn't slow down after that time, we need to call emergency services right away. "
                "You're handling this perfectly - stay calm and keep me updated."
            ),
            "heart": (
                "Okay, I need you to stay calm - I know this is frightening, but we're going to get through this together. "
                "First, call emergency services right now if you haven't already. While you're doing that, help them sit down and try to keep them calm. "
                "Loosen any tight clothing around their neck or chest so they can breathe easier. "
                "If they're conscious and not allergic, an aspirin can help - have them chew it slowly. "
                "Keep talking to them, let them know help is coming. You're doing everything right. "
                "Monitor their breathing and stay with them until help arrives."
            ),
            "choking": (
                "I understand someone's choking - this is scary but we can handle this together. "
                "If they can still cough or make sounds, encourage them to keep coughing - that's their body trying to clear it. "
                "If they can't cough or speak, we need to act fast. Stand behind them and give 5 firm back blows between their shoulder blades with the heel of your hand. "
                "If that doesn't work, wrap your arms around them and give 5 quick upward thrusts just above their belly button. "
                "Keep alternating between back blows and abdominal thrusts. You're doing great - don't give up. "
                "If the object doesn't come out soon, call emergency services immediately."
            ),
            "unconscious": (
                "I know finding someone unconscious is terrifying, but let's work through this step by step. "
                "First, gently shake their shoulders and shout 'Are you okay?' to see if they respond. "
                "Check if they're breathing normally - look for their chest rising and falling. "
                "If they're breathing, carefully roll them onto their side in the recovery position to keep their airway clear. "
                "If they're not breathing normally, we need to start CPR right away and call emergency services. "
                "You're being so helpful right now. Stay with them and keep checking their breathing. "
                "Help is coming - you're doing everything right."
            ),
            "stroke": (
                "I understand you're worried about stroke symptoms - time is really important here, so let's act quickly together. "
                "Use the FAST test with me: Face - ask them to smile, does one side droop? Arms - have them raise both arms, does one drift down? "
                "Speech - ask them to repeat a simple phrase, is it slurred or strange? Time - if any of these are present, it's time to call emergency services immediately. "
                "Note what time the symptoms started - doctors will need to know this. "
                "Keep them calm and comfortable, don't give them anything to eat or drink in case they have trouble swallowing. "
                "You're doing exactly what you should be doing. Stay with them until help arrives."
            ),
            "seizure": (
                "I know seeing someone have a seizure is really frightening, but you can help them safely. "
                "First, stay calm and time the seizure if you can. Clear the area around them of anything they could hit. "
                "Don't try to hold them down or put anything in their mouth - that's an old myth and can actually hurt them. "
                "Just stay with them and keep them safe. Most seizures stop on their own within a few minutes. "
                "Once it stops, they might be confused - that's normal. Help them sit up slowly and speak calmly to them. "
                "If it lasts more than 5 minutes, or if they have trouble breathing afterwards, call emergency services right away. "
                "You're being such a good helper - they're lucky to have you there."
            )
        }
    
    def generate(self, prompt: str, max_tokens: int = 150, **kwargs) -> str:
        prompt_lower = prompt.lower()

        # Extract the user's actual query from the prompt so routing doesn't get confused by
        # retrieved context (RAG snippets can contain unrelated keywords).
        user_text = ""
        for marker in ("user query:", "user:", "you:"):
            if marker in prompt_lower:
                user_text = prompt_lower.split(marker, 1)[1]
                break
        # Keep just the first line / short span to avoid swallowing context blocks.
        user_text = user_text.strip().splitlines()[0] if user_text else ""

        def has_any(text: str, terms: list[str]) -> bool:
            return any(t in text for t in terms)

        bleeding_terms = ["bleeding", "bleed", "blood", "wound", "hemorrhage", "cut", "laceration"]
        burn_terms = ["burn", "burned", "burnt", "scald", "hot water", "steam", "chemical burn", "electrical burn"]
        fracture_terms = [
            "fracture",
            "broken bone",
            "broke my",
            "broke their",
            "broken",
            "splint",
            "immobil",
            "wrist",
            "ankle",
            "forearm",
            "hip fracture",
            "femur",
        ]

        # Primary routing: what the user actually said.
        if has_any(user_text, burn_terms) and not has_any(user_text, fracture_terms):
            return self.responses["burn"]
        if has_any(user_text, bleeding_terms) and not has_any(user_text, fracture_terms):
            return self.responses["bleeding"]
        if has_any(user_text, fracture_terms) and not has_any(user_text, bleeding_terms):
            return self.responses["fracture"]
        # If user mentions BOTH (e.g., open fracture with bleeding), bleeding control comes first.
        if has_any(user_text, bleeding_terms) and has_any(user_text, fracture_terms):
            return self.responses["bleeding"]

        # Secondary routing: fall back to prompt-wide keyword matching.
        if has_any(prompt_lower, burn_terms) and not has_any(prompt_lower, bleeding_terms + fracture_terms):
            return self.responses["burn"]
        if has_any(prompt_lower, fracture_terms) and not has_any(prompt_lower, bleeding_terms):
            return self.responses["fracture"]
        
        # Find best matching response based on keywords
        for keyword, response in self.responses.items():
            if keyword in prompt_lower:
                return response
        
        # Check for other common emergency terms
        if any(word in prompt_lower for word in ['help', 'emergency', 'hurt', 'pain', 'accident']):
            return (
                "I can hear that you're dealing with an emergency situation, and I want you to know I'm here to help. "
                "First, take a deep breath - you're going to get through this. "
                "Make sure the scene is safe for both you and the person who needs help. "
                "Check if they're responsive by gently tapping their shoulders and asking if they're okay. "
                "If this is a serious emergency, don't hesitate to call emergency services - they're there to help. "
                "You're being incredibly brave and helpful right now. Tell me more about what's happening and I'll guide you through the next steps."
            )
        
        # Default conversational response
        return (
            "I'm here to help you with any emergency situation you're facing. "
            "Whatever's happening right now, we can work through it together step by step. "
            "Can you tell me a bit more about what's going on? I'll guide you through exactly what to do. "
            "You're not alone in this - I'm right here with you."
        )

def create_llm_backend(config: Dict[str, Any]) -> LLMBackend:
    """Factory function to create appropriate LLM backend"""
    
    backend_type = config.get("backend", "mock").lower()
    backend_config = config.get("config", {})
    
    if backend_type == "openai":
        return OpenAIBackend(backend_config)
    elif backend_type == "anthropic":
        return AnthropicBackend(backend_config)
    elif backend_type == "ollama":
        return OllamaBackend(backend_config)
    elif backend_type in {"llama_cpp", "llamacpp", "gguf"}:
        return LlamaCppBackend(backend_config)
    elif backend_type == "mock":
        return MockLLMBackend(backend_config)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")