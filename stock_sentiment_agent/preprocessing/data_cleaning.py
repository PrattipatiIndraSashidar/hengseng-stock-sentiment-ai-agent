"""Text cleaning, deduplication, and explicit trading position extraction."""

from __future__ import annotations

import hashlib
import html
import re
from typing import Iterable, List, TypeVar

T = TypeVar("T")

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# Only extract trading positions when explicitly stated in English, Chinese, or Korean
_LONG_PATTERNS = [
    re.compile(r"\b(going long|long position|buy and hold|entered long|bought calls|strong buy|accumulating)\b", re.I),
    re.compile(r"(做多|買入|加倉|建倉|長線持有|看漲買進)"),
    re.compile(r"(매수|롱 포지션|추가 매수|비중 확대|분할 매수)"),
]
_SHORT_PATTERNS = [
    re.compile(r"\b(going short|short position|shorting|bought puts|sell position|trimming exposure)\b", re.I),
    re.compile(r"(做空|沽空|減倉|賣出|止損離場)"),
    re.compile(r"(매도|공매도|숏 포지션|비중 축소|손절)"),
]
_HOLD_PATTERNS = [
    re.compile(r"\b(holding position|on hold|wait and see|staying flat|hold rating)\b", re.I),
    re.compile(r"(觀望|繼續持有|暫不操作|持倉待變)"),
    re.compile(r"(관망|보유 유지|홀딩)"),
]


def clean_text(raw: str) -> str:
    """Normalize multilingual social/news text without stripping CJK or Hangul characters."""
    if not raw:
        return ""
    text = html.unescape(raw)
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def text_fingerprint(text: str) -> str:
    normalized = clean_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def deduplicate_records(records: Iterable[T], key_fn) -> List[T]:
    seen = set()
    output: List[T] = []
    for item in records:
        key = key_fn(item)
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def extract_explicit_trading_position(texts: Iterable[str]) -> str:
    """Extract Long/Short/Hold ONLY when explicitly stated; otherwise return 'Not mentioned'."""
    long_hits = 0
    short_hits = 0
    hold_hits = 0

    for raw in texts:
        t = clean_text(raw)
        if not t:
            continue
        if any(p.search(t) for p in _LONG_PATTERNS):
            long_hits += 1
        if any(p.search(t) for p in _SHORT_PATTERNS):
            short_hits += 1
        if any(p.search(t) for p in _HOLD_PATTERNS):
            hold_hits += 1

    if long_hits == 0 and short_hits == 0 and hold_hits == 0:
        return "Not mentioned"
    if long_hits > short_hits and long_hits >= hold_hits:
        return "Long"
    if short_hits > long_hits and short_hits >= hold_hits:
        return "Short"
    return "Hold"
