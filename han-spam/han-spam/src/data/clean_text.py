"""
src/data/clean_text.py
Text-cleaning utilities applied to raw email subject + body text,
following the preprocessing steps described in Zavrak & Yilmaz (2023), Sec 4.3.
"""
import re
import unicodedata

import nltk

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s\.\!\?]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_MULTI_PUNCT_RE = re.compile(r"([\.\!\?]){2,}")


def strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def remove_urls_and_emails(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    return text


def remove_punctuation(text: str, keep_sentence_boundaries: bool = True) -> str:
    """Strip punctuation but optionally keep . ! ? so sentence tokenisation still works."""
    if keep_sentence_boundaries:
        text = _NON_ALPHA_RE.sub(" ", text)
        text = _MULTI_PUNCT_RE.sub(r"\1", text)
    else:
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text


def collapse_whitespace(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def clean_email_text(subject: str, body: str) -> str:
    """
    Full cleaning pipeline for one email:
    1. Concatenate subject + body
    2. Strip HTML tags
    3. Normalise unicode → ascii
    4. Remove URLs / email addresses
    5. Lowercase
    6. Remove punctuation (keep sentence boundaries)
    7. Collapse whitespace
    """
    text = f"{subject or ''} . {body or ''}"
    text = strip_html(text)
    text = normalize_unicode(text)
    text = remove_urls_and_emails(text)
    text = text.lower()
    text = remove_punctuation(text, keep_sentence_boundaries=True)
    text = collapse_whitespace(text)
    return text


def sentence_tokenize(text: str) -> list[str]:
    """Split cleaned text into sentences using NLTK punkt."""
    sentences = nltk.sent_tokenize(text)
    # Re-strip any leftover punctuation per-sentence for the word tokeniser
    return [s.strip() for s in sentences if s.strip()]


def word_tokenize(sentence: str) -> list[str]:
    """Split a sentence into words using NLTK."""
    return [w for w in nltk.word_tokenize(sentence) if w.isalpha()]
