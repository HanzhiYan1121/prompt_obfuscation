#!/usr/bin/env python3
"""Lightweight text metrics shared by the local recovery evaluators."""

from __future__ import annotations

import re
from collections import Counter

from rouge_score import rouge_scorer


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been",
    "before", "being", "between", "both", "but", "can", "could", "does",
    "each", "for", "from", "have", "into", "must", "not", "only", "or",
    "other", "our", "should", "than", "that", "the", "their", "then",
    "there", "these", "they", "this", "those", "through", "use", "using",
    "was", "were", "what", "when", "where", "which", "while", "with",
    "would", "you", "your",
}


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def token_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    pred = Counter(_tokens(prediction))
    ref = Counter(_tokens(reference))
    if not pred or not ref:
        return 0.0, 0.0, 0.0
    overlap = sum((pred & ref).values())
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def rouge_l_f1(prediction: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference or "", prediction or "")["rougeL"].fmeasure)


def keyword_recall(prediction: str, reference: str) -> tuple[float, list[str], list[str]]:
    reference_keywords = sorted(
        {token for token in _tokens(reference) if len(token) >= 4 and token not in _STOPWORDS}
    )
    prediction_tokens = set(_tokens(prediction))
    hits = [token for token in reference_keywords if token in prediction_tokens]
    misses = [token for token in reference_keywords if token not in prediction_tokens]
    recall = len(hits) / len(reference_keywords) if reference_keywords else 0.0
    return recall, hits, misses
