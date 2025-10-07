from typing import List, Optional

DEFAULT_SYSTEM = (
    "You are a calm, reassuring first-aid assistant. You speak naturally and briefly, "
    "avoid medical jargon unless necessary, and give practical, safe guidance."
)

def _section(title: str, text: str) -> str:
    return f"\n{title}:\n{text.strip()}\n"

def warm_style(user_query: str, rag_block: str, protocol_block: str) -> str:
    intro = (
        "Start with one short, empathetic sentence acknowledging the situation. "
        "Then give clear, simple steps. Keep sentences short. "
        "If there is any chance of life-threatening danger, say to call emergency services. "
        "End with one brief check question to confirm the person’s status."
    )
    parts: List[str] = [
        f"<|system|>\n{DEFAULT_SYSTEM}<|end|>",
        _section("INSTRUCTIONS", intro),
    ]
    if protocol_block:
        parts.append(_section("MEDICAL PROTOCOL (follow exactly)", protocol_block))
    if rag_block:
        parts.append(_section("REFERENCES (use as facts)", rag_block))
    parts.append(_section("QUESTION", user_query))
    parts.append("\nAnswer in this structure:\n"
                 "- 1 short empathetic sentence\n"
                 "- 3–6 concise steps (numbered)\n"
                 "- 1 brief reassurance + when to call emergency services\n"
                 "- 1 short follow‑up question\n")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)

def coach_style(user_query: str, rag_block: str, protocol_block: str) -> str:
    intro = (
        "Use a motivating coaching tone. Keep it friendly and direct. "
        "Use bullets instead of numbers if steps are fewer than four."
    )
    parts: List[str] = [
        f"<|system|>\n{DEFAULT_SYSTEM}<|end|>",
        _section("INSTRUCTIONS", intro),
    ]
    if protocol_block:
        parts.append(_section("MEDICAL PROTOCOL (follow exactly)", protocol_block))
    if rag_block:
        parts.append(_section("REFERENCES (use as facts)", rag_block))
    parts.append(_section("QUESTION", user_query))
    parts.append("\nAnswer with: brief friendly line, 3–6 steps (bulleted if short), reassurance, one follow‑up question.\n")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)

def plain_style(user_query: str, rag_block: str, protocol_block: str) -> str:
    intro = "Be concise and neutral. Keep steps short. No extra filler."
    parts: List[str] = [
        f"<|system|>\n{DEFAULT_SYSTEM}<|end|>",
        _section("INSTRUCTIONS", intro),
    ]
    if protocol_block:
        parts.append(_section("MEDICAL PROTOCOL (follow exactly)", protocol_block))
    if rag_block:
        parts.append(_section("REFERENCES (use as facts)", rag_block))
    parts.append(_section("QUESTION", user_query))
    parts.append("\nAnswer with 3–6 steps and a brief closing.\n")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)

def build_prompt(user_query: str,
                 rag_context: str = "",
                 decision_text: str = "",
                 style: str = "warm") -> str:
    """Return a Phi-3 compatible chat prompt with the selected tone."""
    rag_block = rag_context.strip()
    protocol_block = decision_text.strip()
    style = (style or "warm").lower()
    if style == "coach":
        return coach_style(user_query, rag_block, protocol_block)
    if style == "plain":
        return plain_style(user_query, rag_block, protocol_block)
    return warm_style(user_query, rag_block, protocol_block)