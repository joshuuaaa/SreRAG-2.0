#!/usr/bin/env python3
import argparse
import sys
import traceback
from typing import List, Dict

from src.utils import load_config
from src.llm.flexible_engine import FlexibleLLMEngine as LLMEngine

# Import enhanced prompt system with fallback
try:
    from src.prompt.enhanced_styles import build_enhanced_prompt as build_enhanced_prompt_func
    from src.prompt.humanoid_styles import build_simple_conversational_prompt as build_prompt
    ENHANCED_PROMPTS = False  # Use humanoid conversational style
except ImportError:
    from src.prompt.styles import build_prompt
    ENHANCED_PROMPTS = False

# Import response validator
try:
    from src.llm.response_validator import validate_medical_response
    RESPONSE_VALIDATION = True  # Re-enable now that LLM works
except ImportError:
    RESPONSE_VALIDATION = False

# Import adaptive config for performance monitoring
try:
    from src.utils.adaptive_config import adaptive_config
    PERFORMANCE_MONITORING = False  # Disable for now
except ImportError:
    PERFORMANCE_MONITORING = False

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
    parser = argparse.ArgumentParser(description="Crisis Assistant")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable push-to-talk voice mode (offline STT/TTS).",
    )
    parser.add_argument(
        "--say",
        type=str,
        default=None,
        help="(Voice mode) Bypass microphone and use the provided text as input.",
    )
    args = parser.parse_args()

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
            print(f"RAG loaded: {rag}")
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

    # Voice mode
    if args.voice:
        try:
            from src.audio.stt import STTConfig, listen_and_transcribe
            from src.audio.tts import TTSConfig, speak
        except Exception as e:
            print("Voice mode dependencies not available:", e)
            print("Tip: ensure numpy, sounddevice, soundfile are installed and whisper.cpp + piper binaries are available.")
            return 1

        voice_cfg = config.get("voice", {})
        stt_cfg_d = voice_cfg.get("stt", {})
        tts_cfg_d = voice_cfg.get("tts", {})

        stt_cfg = STTConfig(
            backend=stt_cfg_d.get("backend", "whispercpp"),
            whisper_bin=stt_cfg_d.get("whisper_bin", "whisper-cli"),
            model_path=stt_cfg_d.get("model_path", "models/whisper/ggml-tiny.en.bin"),
            language=stt_cfg_d.get("language", "en"),
            sample_rate=int(stt_cfg_d.get("sample_rate", 16000)),
            record_seconds=float(stt_cfg_d.get("record_seconds", 6.0)),
        )

        tts_cfg = TTSConfig(
            backend=tts_cfg_d.get("backend", "piper"),
            piper_bin=tts_cfg_d.get("piper_bin", "piper"),
            voice_model=tts_cfg_d.get("voice_model", ""),
        )

        print("\nVoice mode (push-to-talk).")
        print("- Press Enter to record.")
        print("- Say 'quit' to exit.")

        # Fast path: bypass mic for debugging / laptop testing.
        if args.say:
            user_query = args.say.strip()
            if user_query.lower() in {"quit", "exit", "q", "stop"}:
                print("Goodbye. Stay safe.")
                return 0

            decision_text = ""
            try:
                decision_text = maybe_get_decision_text(decision_engine, user_query)
            except Exception:
                pass

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
                            text = d.get("text", "")
                            if len(text) > 200:
                                text = text[:200] + "..."
                            chunks.append(f"[{src}]\n{text}")
                        rag_context = "\n\n".join(chunks)
                    if len(rag_context) > 800:
                        rag_context = rag_context[:800] + "\n[...context truncated...]"
                except Exception:
                    rag_context = ""

            prompt = build_prompt(
                user_query=user_query,
                rag_context=rag_context,
                decision_text=decision_text,
                style=prompt_style,
            )

            try:
                answer = llm.generate(
                    prompt,
                    max_tokens=config.get("llm", {}).get("max_tokens", 150),
                )
            except Exception as e:
                print("LLM generation failed:", e)
                return 1

            print("\nAssistant:")
            print("-" * 60)
            print(answer)
            print("-" * 60)

            try:
                speak(answer, tts_cfg)
            except Exception as e:
                print("TTS failed:", e)

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

        while True:
            try:
                input("\nPress Enter to talk (Ctrl+C to exit)...")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting. Stay safe.")
                return 0

            try:
                print("Listening...")
                user_query = listen_and_transcribe(stt_cfg).strip()
                if not user_query:
                    print("(Didn't catch that — try again.)")
                    continue
                print(f"You (STT): {user_query}")
            except Exception as e:
                print("STT failed:", e)
                continue

            if user_query.lower() in {"quit", "exit", "q", "stop"}:
                print("Goodbye. Stay safe.")
                return 0

            # Reuse existing text pipeline by injecting query into a single-turn flow.
            # We implement the same steps as the text loop below.

            decision_text = ""
            try:
                decision_text = maybe_get_decision_text(decision_engine, user_query)
            except Exception:
                pass

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
                            text = d.get("text", "")
                            if len(text) > 200:
                                text = text[:200] + "..."
                            chunks.append(f"[{src}]\n{text}")
                        rag_context = "\n\n".join(chunks)
                    if len(rag_context) > 800:
                        rag_context = rag_context[:800] + "\n[...context truncated...]"
                except Exception:
                    rag_context = ""

            emergency_context = None
            prompt = build_prompt(
                user_query=user_query,
                rag_context=rag_context,
                decision_text=decision_text,
                style=prompt_style,
            )

            try:
                answer = llm.generate(
                    prompt,
                    max_tokens=config.get("llm", {}).get("max_tokens", 150),
                )
            except Exception as e:
                print("LLM generation failed:", e)
                continue

            print("\nAssistant:")
            print("-" * 60)
            print(answer)
            print("-" * 60)

            # Speak answer
            try:
                speak(answer, tts_cfg)
            except Exception:
                pass

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

    print("\nType your emergency question (or 'quit' to exit).")
    if PERFORMANCE_MONITORING:
        print("📊 Performance monitoring enabled")
        
    while True:
        try:
            user_query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Show performance summary before exit
            if PERFORMANCE_MONITORING:
                try:
                    summary = adaptive_config.get_performance_summary()
                    print(f"\n📊 Session Summary: {summary['session_metrics']['query_count']} queries, "
                          f"{summary['session_metrics']['avg_response_time']:.2f}s avg response time")
                    health = summary['system_health']
                    print(f"🏥 System Health: {health['overall_health']} (safety: {health['safety_score']:.2f})")
                except Exception:
                    pass
            print("\nExiting. Stay safe.")
            return 0

        if not user_query:
            continue
        if user_query.lower() in {"quit", "exit", "q"}:
            # Show performance summary before exit
            if PERFORMANCE_MONITORING:
                try:
                    summary = adaptive_config.get_performance_summary()
                    print(f"\n📊 Session Summary: {summary['session_metrics']['query_count']} queries")
                    if summary['session_metrics']['query_count'] > 0:
                        print(f"   Average response time: {summary['session_metrics']['avg_response_time']:.2f}s")
                        health = summary['system_health']
                        print(f"   System health: {health['overall_health']} (safety score: {health['safety_score']:.2f})")
                        
                        # Show emergency type distribution
                        if summary['emergency_distribution']:
                            print("   Emergency types:", 
                                  ", ".join(f"{k}:{v}" for k, v in summary['emergency_distribution'].items()))
                except Exception:
                    pass
            print("Goodbye. Stay safe.")
            return 0

        # Start performance monitoring
        timer = None
        if PERFORMANCE_MONITORING:
            timer = adaptive_config.start_query_timer()

        # Decision protocol text (optional)
        decision_text = ""
        try:
            decision_text = maybe_get_decision_text(decision_engine, user_query)
            if not decision_text:
                print("No decision protocol matched. LLM will answer directly.")
        except Exception:
            print("ERROR: Exception occurred while fetching decision protocol.")
            traceback.print_exc()

        # RAG retrieval (optional)
        rag_context = ""
        retrieved = []
        if rag:
            if PERFORMANCE_MONITORING and timer:
                adaptive_config.mark_rag_start(timer)
                
            try:
                retrieved = rag.retrieve(user_query)
                if hasattr(rag, "format_context"):
                    rag_context = rag.format_context(retrieved)
                else:
                    chunks = []
                    for i, d in enumerate(retrieved, 1):
                        src = d.get("metadata", {}).get("source", f"Doc {i}")
                        text = d.get('text', '')
                        # Truncate text to keep prompts manageable
                        if len(text) > 200:
                            text = text[:200] + "..."
                        chunks.append(f"[{src}]\n{text}")
                    rag_context = "\n\n".join(chunks)
                
                # Truncate entire RAG context if too long
                if len(rag_context) > 800:
                    rag_context = rag_context[:800] + "\n[...context truncated...]"
                    
            except Exception as e:
                print("RAG retrieval failed; proceeding without context.")
                print("Reason:", e)
            
            if PERFORMANCE_MONITORING and timer:
                adaptive_config.mark_rag_end(timer)

        # Build prompt - force simple prompts for now
        emergency_context = None  # Simplified for now
        prompt = build_prompt(
            user_query=user_query,
            rag_context=rag_context,
            decision_text=decision_text,
            style=prompt_style,
        )

        try:
            if PERFORMANCE_MONITORING and timer:
                adaptive_config.mark_llm_start(timer)
                
            answer = llm.generate(
                prompt,
                max_tokens=config.get("llm", {}).get("max_tokens", 150),
            )
            
            if PERFORMANCE_MONITORING and timer:
                adaptive_config.mark_llm_end(timer)
            
            # Validate response for medical safety
            safety_result = None
            if RESPONSE_VALIDATION:
                if PERFORMANCE_MONITORING and timer:
                    adaptive_config.mark_validation_start(timer)
                    
                validation_result = validate_medical_response(
                    response=answer,
                    query=user_query,
                    emergency_context=emergency_context
                )
                
                if PERFORMANCE_MONITORING and timer:
                    adaptive_config.mark_validation_end(timer)
                
                safety_result = validation_result.safety_level.value
                
                # Handle validation results
                if validation_result.safety_level.value == "blocked":
                    print("\n🚫 SAFETY BLOCK: Response contained dangerous medical advice.")
                    print("Reason:", "; ".join(validation_result.issues))
                    
                    # Record metrics and continue
                    if PERFORMANCE_MONITORING and timer:
                        emergency_type = emergency_context.get('emergency_type', 'general') if emergency_context else 'general'
                        adaptive_config.end_query_timer(timer, success=False, emergency_type=emergency_type, safety_result=safety_result)
                    continue
                    
                elif validation_result.safety_level.value == "dangerous":
                    print(f"\n⚠️  SAFETY WARNING: Response has significant safety issues (score: {validation_result.score:.2f})")
                    if validation_result.modified_response:
                        print("Using enhanced safety version:")
                        answer = validation_result.modified_response
                        
                elif validation_result.safety_level.value == "warning":
                    print(f"\n💡 Safety note: Minor issues detected (score: {validation_result.score:.2f})")
                    if validation_result.suggestions:
                        print("Suggestions:", "; ".join(validation_result.suggestions[:2]))
            
            # Record successful query metrics
            if PERFORMANCE_MONITORING and timer:
                emergency_type = emergency_context.get('emergency_type', 'general') if emergency_context else 'general'
                adaptive_config.end_query_timer(timer, success=True, emergency_type=emergency_type, safety_result=safety_result)
                
        except Exception as e:
            print("LLM generation failed:", e)
            
            # Record failed query metrics
            if PERFORMANCE_MONITORING and timer:
                emergency_type = emergency_context.get('emergency_type', 'general') if emergency_context else 'general'
                adaptive_config.end_query_timer(timer, success=False, emergency_type=emergency_type)
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