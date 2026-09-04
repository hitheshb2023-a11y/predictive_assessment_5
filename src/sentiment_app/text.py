import re
from typing import Any

URL_RE = re.compile(r"(https?://\\S+|www\\.\\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\\w)@\\w+")
WHITESPACE_RE = re.compile(r"\\s+")


def normalize_tweet(value: Any) -> str:
    text = "" if value is None else str(value)
    text = URL_RE.sub("<URL>", text)
    text = MENTION_RE.sub("<USER>", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
