"""Tier 3: Episodic Memory - conversation summaries and session history."""

from __future__ import annotations

from zenki.db.database import ZenkiDatabase
from zenki.db.models import EpisodicMemory
from zenki.memory.embeddings import EmbeddingProvider


class EpisodicMemoryStore:
    """Store and retrieve episodic memories (conversation summaries).

    Episodic memories capture *what happened* during past sessions:
    summaries, key topics, entities, and importance scores.  Retrieval
    uses text-based search (LIKE queries) as the primary mechanism, with
    vector similarity search available when sqlite-vec is installed.
    """

    def __init__(self, db: ZenkiDatabase, embedding_provider: EmbeddingProvider) -> None:
        self.db = db
        self.embedding_provider = embedding_provider

    def store(
        self,
        session_id: str,
        summary: str,
        key_topics: list[str] | None = None,
        key_entities: list[str] | None = None,
        importance: float = 0.5,
    ) -> EpisodicMemory:
        """Create and persist a new episodic memory.

        Args:
            session_id: The session this memory originates from.
            summary: A textual summary of the conversation.
            key_topics: Important topics mentioned.
            key_entities: Notable entities (people, projects, etc.).
            importance: A relevance/importance score in ``[0, 1]``.

        Returns:
            The newly created :class:`EpisodicMemory` instance.
        """
        memory = EpisodicMemory(
            session_id=session_id,
            summary=summary,
            key_topics=key_topics or [],
            key_entities=key_entities or [],
            importance=importance,
        )
        return self.db.create_episodic_memory(memory)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[tuple[EpisodicMemory, float]]:
        """Retrieve episodic memories relevant to a query.

        Uses text-based search (LIKE queries on summary, topics, and
        entities) as the primary retrieval strategy.  Each result is
        assigned a relevance score based on how well it matches.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance score threshold.

        Returns:
            A list of ``(memory, score)`` tuples, sorted by score descending.
        """
        # Text-based search via the database.
        # Search for each query word individually to handle multi-word queries,
        # then deduplicate by memory ID.
        seen_ids: set[str] = set()
        all_memories: list[EpisodicMemory] = []

        query_words = query.split()
        for word in query_words:
            for mem in self.db.search_episodic_memories(word):
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    all_memories.append(mem)

        # Fall back to full-phrase search if no word-level matches
        if not all_memories:
            all_memories = self.db.search_episodic_memories(query)

        scored: list[tuple[EpisodicMemory, float]] = []
        query_lower = query.lower()

        for mem in all_memories:
            score = self._compute_text_score(mem, query_lower)
            if score >= min_score:
                scored.append((mem, score))

        # Sort by score descending, then by importance descending
        scored.sort(key=lambda pair: (pair[1], pair[0].importance), reverse=True)
        return scored[:top_k]

    def get_recent(self, limit: int = 10) -> list[EpisodicMemory]:
        """Return the most recent episodic memories.

        Args:
            limit: Maximum number of memories to return.

        Returns:
            A list of :class:`EpisodicMemory` instances ordered by
            creation time descending.
        """
        # Use a broad search and limit results
        rows = self.db.conn.execute(
            "SELECT * FROM episodic_memories ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self.db._row_to_episodic(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_text_score(memory: EpisodicMemory, query_lower: str) -> float:
        """Compute a simple relevance score for a text-based match.

        The score is based on how many query words appear in the memory's
        summary, topics, and entities.
        """
        query_words = query_lower.split()
        if not query_words:
            return 0.0

        searchable = " ".join(
            [
                memory.summary.lower(),
                " ".join(t.lower() for t in memory.key_topics),
                " ".join(e.lower() for e in memory.key_entities),
            ]
        )

        matches = sum(1 for word in query_words if word in searchable)
        base_score = matches / len(query_words)

        # Boost by importance
        score = base_score * 0.8 + memory.importance * 0.2

        return min(score, 1.0)
