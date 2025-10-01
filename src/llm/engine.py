"""LLM engine using llama.cpp"""
from llama_cpp import Llama

class LLMEngine:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the quantized LLM"""
        print(f"⏳ Loading LLM from {self.config['model_path']}...")
        self.model = Llama(
            model_path=self.config['model_path'],
            n_ctx=self.config['n_ctx'],
            n_threads=self.config['n_threads'],
            n_gpu_layers=self.config['n_gpu_layers'],
        )
        print("✅ LLM loaded successfully")
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        response = self.model(
            prompt,
            max_tokens=self.config['max_tokens'],
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            repeat_penalty=self.config['repeat_penalty'],
            stop=["</s>", "User:", "\n\n\n"]
        )
        
        return response['choices'][0]['text'].strip()
    
    def format_prompt(self, decision_step: str, rag_context: List[str], user_query: str) -> str:
        """Format medical prompt with decision tree + RAG context"""
        context = "\n\n".join([f"Reference {i+1}: {doc}" for i, doc in enumerate(rag_context)])
        
        prompt = f"""You are a calm, empathetic emergency first aid assistant.

USER QUESTION: {user_query}

MEDICAL PROTOCOL (follow exactly):
{decision_step}

AUTHORITATIVE REFERENCES:
{context}

INSTRUCTIONS:
1. Use ONLY information from the protocol and references above
2. Rephrase into clear, numbered steps
3. Use calm, reassuring language
4. Be concise (max 3-4 sentences)
5. Do NOT add medical advice not in the references

RESPONSE:"""
        
        return prompt