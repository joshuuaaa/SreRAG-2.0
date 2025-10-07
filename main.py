#!/usr/bin/env python3
import sys
import traceback
from typing import List, Dict

from src.utils import load_config
from src.llm.engine import LLMEngine
from src.prompt.styles import build_prompt

# Optional engines (import if present)
try:
    from src.rag.engine import RAGEngine
except Exception:
    RAGEngine = None

try:
    from src.decision.engine import DecisionEngine as _DecisionEngine
    DecisionEngine = _DecisionEngine
except Exception:
    DecisionEngine = None

DISCLAIMER = (
    "DISCLAIMER: This assistant is for educational guidance only and is NOT a "
    "substitute for professional medical care. If someone is in immediate danger, "
    "call your local emergency number now."
)

def maybe_get_decision_text(decision_engine, query: str) -> str:
    if not decision_engine:
        return ""
    for candidate in ("recommend", "get_steps", "match_protocol", "infer"):
        fn = getattr(decision_engine, candidate, None)
        if callable(fn):
            try:
                result = fn(query)
                if isinstance(result, dict) and "text" in result:
                    return result["text"]
                if isinstance(result, str):
                    return result
            except Exception:
                pass
    return ""

def main() -> int:
    print("🚑 Crisis Assistant (CLI)")
    print("=" * 60)
    print(DISCLAIMER)
    print("=" * 60)

    try:
        config = load_config()
    except Exception as e:
        print("Failed to load configuration:", e)
        return 1

    # Tone/style from config (defaults to warm)
    app_cfg = config.get("app", {})
    prompt_style = (app_cfg.get("prompt_style") or "warm").lower()

    # LLM
    try:
        llm = LLMEngine(config.get("llm", {}))
    except Exception as e:
        print("Failed to initialize LLM:", e)
        traceback.print_exc()
        return 1

    # RAG (optional)
    rag = None
    if RAGEngine is not None and "rag" in config:
        try:
            rag = RAGEngine(config["rag"])
            stats = getattr(rag, "get_stats", lambda: {})()
            if stats:
                print(f"RAG index: {stats.get('total_documents', 0)} docs | "
                      f"dim={stats.get('embedding_dimension', '?')} | "
                      f"model={stats.get('model', '?')}")
        except Exception as e:
            print("RAG engine could not be initialized. Continuing without RAG.")
            print("Reason:", e)

    # Decision (optional)
    decision_engine = None
    if DecisionEngine is not None and "decision" in config:
        try:
            decision_engine = DecisionEngine(config["decision"])
            print("Decision engine loaded.")
        except Exception as e:
            print("Decision engine could not be initialized. Continuing without decision tree.")
            print("Reason:", e)

    print("\nType your emergency question (or 'quit' to exit).")
    while True:
        try:
            user_query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Stay safe.")
            return 0

        if not user_query:
            continue
        if user_query.lower() in {"quit", "exit", "q"}:
            print("Goodbye. Stay safe.")
            return 0

        # Decision protocol text (optional)
        decision_text = ""
        try:
            decision_text = maybe_get_decision_text(decision_engine, user_query)
        except Exception:
            pass

        # RAG retrieval (optional)
        rag_context = ""
        retrieved = []
        if rag:
            try:
                retrieved = rag.retrieve(user_query)
                if hasattr(rag, "format_context"):
                    rag_context = rag.format_context(retrieved)
                else:
                    chunks = []
                    for i, d in enumerate(retrieved, 1):
                        src = d.get("metadata", {}).get("source", f"Doc {i}")
                        chunks.append(f"[{src}]\n{d.get('text','')}")
                    rag_context = "\n\n".join(chunks)
            except Exception as e:
                print("RAG retrieval failed; proceeding without context.")
                print("Reason:", e)

        # Build a warmer, more natural prompt
        prompt = build_prompt(
            user_query=user_query,
            rag_context=rag_context,
            decision_text=decision_text,
            style=prompt_style,
        )

        try:
            answer = llm.generate(
                prompt,
                max_tokens=config.get("llm", {}).get("max_tokens", 300),
            )
        except Exception as e:
            print("LLM generation failed:", e)
            continue

        print("\nAssistant:")
        print("-" * 60)
        print(answer)
        print("-" * 60)

        if retrieved:
            print("Sources:")
            for i, d in enumerate(retrieved, 1):
                src = d.get("metadata", {}).get("source", f"Doc {i}")
                score = d.get("score", None)
                if score is not None:
                    print(f"  {i}. {src} (score: {score:.3f})")
                else:
                    print(f"  {i}. {src}")

    return 0

if __name__ == "__main__":
    sys.exit(main())