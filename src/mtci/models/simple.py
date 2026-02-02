from __future__ import annotations

from typing import Sequence


class SimpleSentimentModel:
    positive_tokens = {
        "good",
        "great",
        "love",
        "excellent",
        "amazing",
        "nice",
        "happy",
        "wonderful",
    }

    def predict(self, xs: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for text in xs:
            words = {w.strip(".,!?;:\"").lower() for w in text.split()}
            score = 0.9 if self.positive_tokens.intersection(words) else 0.1
            scores.append(score)
        return scores
