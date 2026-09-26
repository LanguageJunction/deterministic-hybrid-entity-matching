import re
from collections import Counter

WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "the", "and", "of", "for", "at", "inc", "incorporated", "corp",
    "corporation", "co", "company", "ltd", "limited", "llc", "plc",
    "pvt", "private", "public", "holdings", "holding", "group"
}

def word_tokens(text: str):
    return WORD_RE.findall(text.lower())

def content_tokens(text: str):
    return [t for t in word_tokens(text) if t not in STOPWORDS]

def char_ngrams(text: str, n: int = 3):
    s = f" {text.strip()} "
    if len(s) <= n:
        return {s}
    return {s[i:i+n] for i in range(len(s)-n+1)}

def acronym(text: str):
    toks = content_tokens(text)
    return "".join(t[0] for t in toks if t)

def token_counter(text: str):
    return Counter(content_tokens(text))
