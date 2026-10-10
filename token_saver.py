"""
Token saver - keeps the context small, which matters most in long chats.

CrewAI manages the agents' own prompts, so the biggest lever we control is the
GROUNDING we add: on a long session, feeding every earlier turn back verbatim
balloons the prompt each round. This module condenses that history into a few
lines with a single cheap LLM call, and only when it is actually long. Short
history is passed through untouched (summarizing it would cost more than it
saves).
"""
from __future__ import annotations

from llm import build_llm

# How many of the most recent turns to keep VERBATIM (never summarised). A
# follow-up almost always refers to the last turn, so it stays in full.
RECENT_TURNS_VERBATIM = 2
# Only summarise the older turns once they get long; below this, keep them
# verbatim too (no point spending an LLM call to shrink a little text).
SUMMARISE_OLDER_OVER = 1500


def _summarise_older(text: str) -> str:
    """Make a /compact-style, understanding-based summary of the OLDER turns.

    Not a truncation - the LLM reads the whole earlier conversation and writes a
    faithful recap: what was asked, the decisions made, the files changed, the
    current state, and anything still open. If the gateway fails we keep the full
    text rather than lose any of it."""
    llm = build_llm(temperature=0.0)
    prompt = (
        "You are writing a handoff summary of the EARLIER part of a coding session, "
        "so the work can continue without losing context. Read all of it and write a "
        "faithful, understanding-based summary (like a /compact): what the user asked "
        "across these turns, the decisions and approach taken, the files created or "
        "changed, the current state, and anything unresolved. Preserve every meaningful "
        "detail, requirement, constraint, dependency, and open issue; drop only "
        "repetition and filler. Use terse bullets. No preamble.\n\n"
        + text
    )
    try:
        return llm.call(prompt).strip()
    except Exception:
        return text   # never chop - keep the full older text if summarising fails


def build_history(session, recent: int = RECENT_TURNS_VERBATIM) -> str:
    """Build the grounding for a continued session: the most recent turn(s) kept
    in FULL, plus a summary of everything older (not a truncation).

    This replaces the old 'chop every turn to 200 chars' behaviour - older turns
    are summarised with understanding, so no content is simply lost."""
    from sessions import format_turns

    turns = getattr(session, "turns", None) or []
    if not turns:
        return ""

    recent_turns = turns[-recent:]
    older_turns = turns[:-recent]
    parts = []

    if older_turns:
        older_text = format_turns(older_turns)
        if len(older_text) > SUMMARISE_OLDER_OVER:
            body = _summarise_older(older_text)
            parts.append("SUMMARY OF EARLIER TURNS (understanding-based recap):\n" + body)
        else:
            parts.append("EARLIER TURNS:\n" + older_text)

    parts.append("MOST RECENT TURN(S), IN FULL:\n" + format_turns(recent_turns))
    return "\n\n".join(parts)


# Backwards-compatible shim: older callers passed the pre-built brief string.
def condense_history(history_brief: str) -> str:
    if not history_brief or len(history_brief) < SUMMARISE_OLDER_OVER:
        return history_brief
    return "EARLIER IN THIS SESSION (condensed):\n" + _summarise_older(history_brief)
