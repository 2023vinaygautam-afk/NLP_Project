"""Reusable lexical feature extraction for the adaptive healthcare corpus."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer


_WORD_TOKEN = re.compile(r"^[a-z0-9]+(?:[-'][a-z0-9]+)*$")
_TOKENIZER = TreebankWordTokenizer()
_PORTER = PorterStemmer()


def tokenize_documents(
    documents: Sequence[tuple[str, str]],
) -> dict[str, list[str]]:
    """Tokenize document text with NLTK and retain words and hyphenated terms."""
    tokenized: dict[str, list[str]] = {}
    for document_id, text in documents:
        tokenized[document_id] = [
            token.lower()
            for token in _TOKENIZER.tokenize(text)
            if _WORD_TOKEN.fullmatch(token.lower())
        ]
    return tokenized


def english_stop_words() -> set[str]:
    """Return the standard NLTK English stop-word list."""
    try:
        return set(stopwords.words("english"))
    except LookupError as error:
        raise LookupError(
            "NLTK's English stop-word corpus is unavailable. Install it with "
            "`python -m nltk.downloader stopwords` in the app's Python environment."
        ) from error


def remove_stop_words(
    tokenized_documents: dict[str, list[str]],
    stop_words: set[str] | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return retained and removed tokens, preserving document IDs and order."""
    words = stop_words if stop_words is not None else english_stop_words()
    retained: dict[str, list[str]] = {}
    removed: dict[str, list[str]] = {}
    for document_id, tokens in tokenized_documents.items():
        retained[document_id] = [token for token in tokens if token not in words]
        removed[document_id] = [token for token in tokens if token in words]
    return retained, removed


def stem_documents(
    tokenized_documents: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Apply NLTK Porter stemming to each token without changing source text."""
    return {
        document_id: [_PORTER.stem(token) for token in tokens]
        for document_id, tokens in tokenized_documents.items()
    }


def stem_mapping(tokens: Iterable[str]) -> list[dict[str, str]]:
    """Build a unique original-term to Porter-stem mapping."""
    terms = sorted(set(tokens))
    return [
        {"Original term": term, "Porter stem": _PORTER.stem(term)}
        for term in terms
    ]


def build_terms_dictionary(
    tokenized_documents: dict[str, list[str]],
):
    """Build term occurrence counts and document-frequency postings."""
    import pandas as pd

    total_counts: Counter[str] = Counter()
    document_counts: dict[str, Counter[str]] = {}
    for document_id, tokens in tokenized_documents.items():
        counts = Counter(tokens)
        document_counts[document_id] = counts
        total_counts.update(counts)

    rows: list[dict[str, object]] = []
    for serial_number, term in enumerate(sorted(total_counts), start=1):
        postings = [
            f"{document_id}: {counts[term]}"
            for document_id, counts in document_counts.items()
            if counts[term]
        ]
        rows.append(
            {
                "S No.": serial_number,
                "Term": term,
                "Documents": ", ".join(postings),
                "Term Occurrences": total_counts[term],
                "Document Frequency": len(postings),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "S No.",
            "Term",
            "Documents",
            "Term Occurrences",
            "Document Frequency",
        ],
    )


def document_token_counts(
    tokenized_documents: dict[str, list[str]],
) -> list[dict[str, int | str]]:
    return [
        {"Document ID": document_id, "Token Count": len(tokens)}
        for document_id, tokens in tokenized_documents.items()
    ]
