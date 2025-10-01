"""Main orchestrator - connects decision trees, RAG, and LLM"""
import sys
from typing import Optional
from src.utils import load_config
from src.decision.engine import DecisionTreeEngine
from src.rag.retriever import RAGRetriever
from src.llm.engine import LLMEngine

class Orchestrator:
    def __init__(self):
        self.config = load_config()
        
        # Initialize components
        self.decision_engine = DecisionTreeEngine(self.config['decision_trees']['path'])
        self.retriever = RAGRetriever(
            self.config['rag']['index_path'],
            self.config['rag']['embedding_model'],
            self.config['rag']['top_k']
        )
        self.llm = LLMEngine(self.config['llm'])
        
        # Safety settings
        self.safety = self.config['safety']
        self.conversation_count = 0
    
    def process_query(self, user_input: str) -> str:
        """Main query processing pipeline"""
        # Step 1: Match decision tree
        match = self.decision_engine.match_query(user_input)
        
        if not match:
            return self._fallback_response(user_input)
        
        # Step 2: Get current decision node
        current_node = match['current_node']
        
        # Step 3: Retrieve relevant documents from RAG
        rag_results = []
        if 'rag_tags' in current_node:
            query = f"{user_input} {current_node.get('prompt', '')}"
            rag_results = self.retriever.search(
                query,
                filters=current_node.get('rag_filter')
            )
        
        # Step 4: Format context
        rag_context = [r['text'] for r in rag_results[:self.config['rag']['top_k']]]
        decision_step = current_node.get('prompt', '')
        
        # Step 5: Generate response via LLM
        prompt = self.llm.format_prompt(decision_step, rag_context, user_input)
        response = self.llm.generate(prompt)
        
        # Step 6: Add safety disclaimer
        if self.safety['always_suggest_emergency_services']:
            response += "\n\n⚠️  If this is life-threatening, call emergency services immediately."
        
        return response
    
    def _fallback_response(self, user_input: str) -> str:
        """Handle queries that don't match decision trees"""
        return (
            "I'm here to help with emergency first aid guidance. "
            "I can assist with:\n"
            "- Severe bleeding\n"
            "- CPR and breathing emergencies\n"
            "- Burns\n"
            "- Choking\n\n"
            "Could you describe the emergency situation?"
        )
    
    def run_text_loop(self):
        """Text-based interaction loop"""
        print("\n🚑 Crisis Assistant - Text Mode")
        print("=" * 50)
        
        if self.safety['disclaimer_on_start']:
            print("\n⚠️  DISCLAIMER: This is an educational tool, not a substitute for professional medical care.")
            print("Always call emergency services when available.\n")
        
        print("Type your emergency question (or 'quit' to exit)\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Stay safe. Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Check conversation limit
                self.conversation_count += 1
                if self.conversation_count > self.safety['max_conversation_length']:
                    print("\n⚠️  Conversation limit reached. Please call emergency services.")
                    break
                
                # Process query
                response = self.process_query(user_input)
                print(f"\n🚑 Assistant: {response}\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Stay safe!")
                sys.exit(0)
            except Exception as e:
                print(f"\n❌ Error: {e}")