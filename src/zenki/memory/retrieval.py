"""Retrieval utilities for the Zenki memory system."""

from __future__ import annotations

import math
from datetime import UTC, datetime


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector (must have the same length as *a*).

    Returns:
        A float in ``[-1, 1]``. Returns ``0.0`` if either vector has
        zero magnitude.

    Raises:
        ValueError: If the vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vectors must have the same length, got {len(a)} and {len(b)}"
        )

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def rank_memories(
    memories: list[tuple[object, float]],
    recency_weight: float = 0.2,
) -> list[tuple[object, float]]:
    """Re-rank memories by combining relevance score with recency.

    Each memory object is expected to have a ``created_at`` attribute
    (a :class:`~datetime.datetime`).  The final score is a weighted
    combination of the original relevance score and a recency bonus.

    Args:
        memories: List of ``(memory_object, relevance_score)`` tuples.
        recency_weight: How much weight to give recency (0-1).
            The remaining weight goes to the original relevance score.

    Returns:
        A new list of ``(memory_object, combined_score)`` tuples, sorted
        by combined score descending.
    """
    if not memories:
        return []

    relevance_weight = 1.0 - recency_weight
    now = datetime.now(UTC)

    # Collect timestamps; fall back to epoch for objects without created_at
    timestamps: list[datetime] = []
    for mem, _score in memories:
        created = getattr(mem, "created_at", None)
        if created is None:
            created = datetime(2000, 1, 1, tzinfo=UTC)
        timestamps.append(created)

    # Compute recency scores normalised to [0, 1]
    ages_seconds = [(now - ts).total_seconds() for ts in timestamps]
    max_age = max(ages_seconds) if ages_seconds else 1.0
    if max_age == 0:
        max_age = 1.0

    recency_scores = [1.0 - (age / max_age) for age in ages_seconds]

    ranked: list[tuple[object, float]] = []
    for i, (mem, rel_score) in enumerate(memories):
        combined = relevance_weight * rel_score + recency_weight * recency_scores[i]
        ranked.append((mem, combined))

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked
