#!/usr/bin/env python3
"""Simple LLM test to isolate issues"""

from src.utils import load_config
from src.llm.engine import LLMEngine

def test_llm():
    """Test LLM with a simple prompt"""
    
    # Load config
    config = load_config()
    
    # Initialize LLM
    print("🔄 Loading LLM...")
    llm = LLMEngine(config["llm"])
    print("✅ LLM loaded!")
    
    # Simple test prompt
    prompt = """You are a helpful medical assistant. 

User query: severe bleeding from arm

Please provide 3-4 immediate first aid steps to help with severe bleeding from an arm."""
    
    print("\n🧠 Testing LLM generation...")
    print("Prompt:", repr(prompt[:100] + "..."))
    
    try:
        response = llm.generate(prompt, max_tokens=200)
        print(f"\n✅ LLM Response ({len(response)} chars):")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm()