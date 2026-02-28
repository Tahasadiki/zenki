"""Tier 4: Semantic Memory - facts, knowledge, insights, and lessons."""

from __future__ import annotations

from zenki.db.database import ZenkiDatabase
from zenki.db.models import SemanticMemory
from zenki.memory.embeddings import EmbeddingProvider


class SemanticMemoryStore:
    """Store and retrieve semantic memories (facts, knowledge, insights).

    Semantic memories represent *what is known* -- distilled facts,
    user preferences, domain knowledge, insights, and lessons learned.
    Retrieval uses text-based search as the primary mechanism.
    """

    def __init__(self, db: ZenkiDatabase, embedding_provider: EmbeddingProvider) -> None:
        self.db = db
        self.embedding_provider = embedding_provider

    def store(
        self,
        content: str,
        category: str,
        tags: list[str] | None = None,
        source: str | None = None,
        importance: float = 0.5,
    ) -> SemanticMemory:
        """Create and persist a new semantic memory.

        Args:
            content: The factual content to store.
            category: One of ``"fact"``, ``"preference"``, ``"knowledge"``,
                ``"insight"``, or ``"lesson"``.
            tags: Optional tags for categorisation.
            source: Where this knowledge came from (e.g. session ID).
            importance: Relevance/importance score in ``[0, 1]``.

        Returns:
            The newly created :class:`SemanticMemory` instance.
        """
        memory = SemanticMemory(
            content=content,
            category=category,
            tags=tags or [],
            source=source,
            importance=importance,
        )
        return self.db.create_semantic_memory(memory)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.3,
        category: str | None = None,
    ) -> list[tuple[SemanticMemory, float]]:
        """Retrieve semantic memories relevant to a query.

        Uses text-based search (LIKE queries on content and tags).

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance score threshold.
            category: Optional category filter.

        Returns:
            A list of ``(memory, score)`` tuples, sorted by score descending.
        """
        # Search for each query word individually to handle multi-word queries,
        # then deduplicate by memory ID.
        seen_ids: set[str] = set()
        all_memories: list[SemanticMemory] = []

        query_words = query.split()
        for word in query_words:
            for mem in self.db.search_semantic_memories(word, category=category):
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    all_memories.append(mem)

        # Fall back to full-phrase search if no word-level matches
        if not all_memories:
            all_memories = self.db.search_semantic_memories(query, category=category)

        scored: list[tuple[SemanticMemory, float]] = []
        query_lower = query.lower()

        for mem in all_memories:
            score = self._compute_text_score(mem, query_lower)
            if score >= min_score:
                scored.append((mem, score))

        scored.sort(key=lambda pair: (pair[1], pair[0].importance), reverse=True)
        return scored[:top_k]

    def get_by_category(self, category: str) -> list[SemanticMemory]:
        """Return all semantic memories in a given category.

        Args:
            category: The category to filter by.

        Returns:
            A list of :class:`SemanticMemory` instances.
        """
        rows = self.db.conn.execute(
            "SELECT * FROM semantic_memories WHERE category = ? ORDER BY importance DESC",
            (category,),
        ).fetchall()
        return [self.db._row_to_semantic(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_text_score(memory: SemanticMemory, query_lower: str) -> float:
        """Compute a simple relevance score for a text-based match."""
        query_words = query_lower.split()
        if not query_words:
            return 0.0

        searchable = " ".join(
            [
                memory.content.lower(),
                " ".join(t.lower() for t in memory.tags),
                memory.category.lower(),
            ]
        )

        matches = sum(1 for word in query_words if word in searchable)
        base_score = matches / len(query_words)

        # Boost by importance
        score = base_score * 0.8 + memory.importance * 0.2

        return min(score, 1.0)
