"""Verifiable math reward for HRM DAPO rollouts.

extract_last_boxed: returns the LAST \\boxed{...} (brace-matched) — DeepSeek-style
self-corrections leave intermediate boxes; only the final one is the answer.
"""
import re
from math_verify import parse, verify

_BOXED = re.compile(r'\\boxed\{')


def extract_last_boxed(text: str):
    starts = [m.start() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    s = starts[-1]
    i = s + len(r'\boxed{')
    depth = 1
    while i < len(text) and depth:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[s:i] if depth == 0 else text[s:]  # tolerate truncated close


def is_correct(response: str, ground_truth: str) -> bool:
    lb = extract_last_boxed(response)
    if lb is None:
        return False
    try:
        return bool(verify(parse(ground_truth), parse(lb)))
    except Exception:
        return False


def reward_fn(response: str, ground_truth: str, format_bonus: float = 0.0) -> float:
    """1.0 correct, else 0.0 (+optional format_bonus if a \\boxed{} is present)."""
    lb = extract_last_boxed(response)
    if lb is None:
        return 0.0
    base = 0.0
    try:
        if verify(parse(ground_truth), parse(lb)):
            base = 1.0
    except Exception:
        base = 0.0
    return base + (format_bonus if base == 0.0 else 0.0)
