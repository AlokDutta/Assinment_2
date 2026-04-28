"""N-gram Language Model for technical term boosting in constrained decoding.

Built from the speech course syllabus corpus to prioritize technical terms
during Whisper transcription.
"""

import re
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, List, Tuple


class NgramLanguageModel:
    """Character/word-level N-gram LM with Kneser-Ney-style smoothing."""

    def __init__(self, n: int = 3, smoothing: float = 0.75):
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.context_counts: Dict[tuple, int] = defaultdict(int)
        self.vocab: set = set()
        self.total_unigrams: int = 0

    def train(self, corpus_path: str):
        """Train N-gram model from a text corpus file."""
        text = Path(corpus_path).read_text().lower()
        words = re.findall(r'[a-z]+', text)
        self.vocab = set(words)
        self.total_unigrams = len(words)

        for i in range(len(words)):
            for order in range(1, self.n + 1):
                if i + 1 >= order:
                    context = tuple(words[i + 1 - order:i])
                    word = words[i]
                    self.ngram_counts[context][word] += 1
                    self.context_counts[context] += 1

        print(f"[N-gram LM] Trained {self.n}-gram model: vocab={len(self.vocab)}, "
              f"unigrams={self.total_unigrams}")

    def log_prob(self, word: str, context: Optional[List[str]] = None) -> float:
        """Compute log probability of word given context with backoff smoothing."""
        word = word.lower()
        if context is None:
            context = []
        context = [w.lower() for w in context]

        for order in range(min(self.n, len(context) + 1), 0, -1):
            ctx = tuple(context[-(order - 1):]) if order > 1 else ()
            if ctx in self.ngram_counts and self.ngram_counts[ctx].get(word, 0) > 0:
                count = self.ngram_counts[ctx][word]
                total = self.context_counts[ctx]
                prob = max((count - self.smoothing), 0) / total
                num_types = len(self.ngram_counts[ctx])
                backoff_weight = (self.smoothing * num_types) / total
                backoff_prob = self._unigram_prob(word)
                final_prob = prob + backoff_weight * backoff_prob
                return math.log(max(final_prob, 1e-10))

        return math.log(max(self._unigram_prob(word), 1e-10))

    def _unigram_prob(self, word: str) -> float:
        count = self.ngram_counts.get((), {}).get(word, 0)
        if count > 0:
            return count / max(self.total_unigrams, 1)
        return 1.0 / (self.total_unigrams + len(self.vocab))

    def score_text(self, text: str) -> float:
        """Score a text string using the N-gram model."""
        words = re.findall(r'[a-z]+', text.lower())
        if not words:
            return 0.0
        total = 0.0
        for i, word in enumerate(words):
            context = words[max(0, i - self.n + 1):i]
            total += self.log_prob(word, context)
        return total / len(words)

    def get_term_boost(self, word: str) -> float:
        """Get a boost score for technical terms (higher for terms in corpus)."""
        word = word.lower()
        if word in self.vocab:
            freq = self.ngram_counts.get((), {}).get(word, 0)
            return math.log(1 + freq)
        return 0.0
