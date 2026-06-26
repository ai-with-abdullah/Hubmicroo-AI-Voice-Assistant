"""Message router — decides whether to skip the LLM (simple lookup) or use it.

Simple lookups: product search, price check, stock query, category browse.
Conversational: multi-turn, complaint, comparison, nuanced policy question.

Uses a keyword classifier — no LLM call for routing.
"""
from __future__ import annotations

import re

_LOOKUP_PATTERNS = re.compile(
    r"\b("
    r"price|cost|kitna|qeemat|سعر|ثمن"
    r"|stock|available|availability|hai kya|موجود"
    r"|show|list|find|dhundho|دکھاؤ|ابحث"
    r"|buy|purchase|order|kharidna|خریدنا|اشتري"
    r"|how much|کتنا|کم یستغرق"
    r"|do you have|kya aap ke paas|هل لديك"
    r"|shipping|delivery|dilevery|شحن|توصيل"
    r"|return|refund|واپسی|إرجاع"
    r")\b",
    re.IGNORECASE,
)

_CONVO_PATTERNS = re.compile(
    r"\b("
    r"compare|difference|which is better|recommend|suggest"
    r"|explain|why|how does|what is the difference"
    r"|complaint|problem|issue|broken|not working"
    r"|hello|hi|salam|مرحبا|السلام|اسلام"
    r")\b",
    re.IGNORECASE,
)

# Greetings — ultra-short path, no retrieval needed
_GREETING = re.compile(
    r"^\s*(hi|hello|hey|salam|salaam|السلام|مرحبا|اسلام|"
    r"aoa|assalam|walaikum|good\s*(morning|evening|afternoon)|"
    r"kaise hain|kya haal|theek hain?|aap kaise)\W*$",
    re.IGNORECASE,
)


def classify(message: str) -> str:
    """Return 'greeting' | 'lookup' | 'conversational'."""
    msg = message.strip()
    if _GREETING.match(msg):
        return "greeting"
    if _LOOKUP_PATTERNS.search(msg):
        return "lookup"
    if _CONVO_PATTERNS.search(msg):
        return "conversational"
    # Default: short messages lean lookup, long messages lean conversational
    return "lookup" if len(msg.split()) <= 8 else "conversational"
