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
            
            # Test connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise RuntimeError("Ollama server not responding")
            
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
            raise RuntimeError(f"Ollama initialization failed: {e}")
    
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
                timeout=30
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

class MockLLMBackend(LLMBackend):
    """Mock LLM for testing with humanoid conversational responses"""
    
    def __init__(self, config: Dict[str, Any]):
        print("✅ Mock LLM backend loaded (humanoid conversation mode)")
        self.responses = {
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
    elif backend_type == "mock":
        return MockLLMBackend(backend_config)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")