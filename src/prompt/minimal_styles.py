"""Minimal prompt styles for resource-constrained models"""

def build_simple_prompt(
    user_query: str,
    rag_context: str = "",
    decision_text: str = "",
    style: str = "warm",
) -> str:
    """Build a minimal prompt that fits in small context windows"""
    
    # Very brief context
    context = ""
    if rag_context and rag_context.strip():
        # Take only first 300 chars of RAG context
        context = f"References: {rag_context[:300]}...\n"
    
    if decision_text and decision_text.strip():
        context += f"Protocol: {decision_text[:200]}...\n"
    
    # Minimal prompt
    prompt = f"""You are a first aid assistant.

{context}
Question: {user_query}

Provide 3-4 clear numbered steps to help with this emergency. Be calm and direct."""
    
    return prompt