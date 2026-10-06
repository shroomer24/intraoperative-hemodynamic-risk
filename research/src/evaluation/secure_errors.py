"""Sanitize exception chains before saving; frame locals are never included."""

import os
import re
import traceback


def sanitized_traceback(exc: Exception) -> str:
    text = "".join(
        traceback.TracebackException.from_exception(exc, capture_locals=False).format(chain=True)
    )
    for name, value in os.environ.items():
        if value and re.search(r"token|secret|password|authorization|api_?key", name, re.I):
            text = text.replace(value, "[REDACTED_CREDENTIAL]")
    text = re.sub(r"(?i)\bBearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", text)
    text = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b", "[REDACTED_JWT]", text)
    text = re.sub(
        r"(?i)((?:authorization|x-api-key|api_key|access_token|refresh_token|password)"
        r"[\"']?\s*[:=]\s*)[^\r\n,}]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)(https?://[^\s?'\"]+)\?[^\s'\"]+", r"\1?[REDACTED_QUERY]", text)
    text = re.sub(
        r"\[(?:\s*(?:[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?|nan|inf)\s*,?){3,}\]",
        "[REDACTED_NUMERIC_PAYLOAD]",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"(?i)((?:subject_id|case_id|window_id|anchor_time_seconds)[\"']?\s*[:=]\s*)"
        r"(?:[\"'][^\"']*[\"']|[^\s,}\]]+)",
        r"\1[REDACTED_PATIENT_FIELD]",
        text,
    )
    text = re.sub(
        r"(?is)((?:\"|')(?:X|y|data|features|labels|train_set)(?:\"|')\s*:\s*)\[.*?\]",
        r"\1[REDACTED_PATIENT_PAYLOAD]",
        text,
    )
    return text
