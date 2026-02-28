"""Memory manager - orchestrates all four memory tiers."""

from __future__ import annotations

from pathlib import Path

from zenki.config.settings import ZenkiSettings
from zenki.db.database import ZenkiDatabase
from zenki.memory.core_memory import CoreMemory
from zenki.memory.embeddings import EmbeddingProvider, get_embedding_provider
from zenki.memory.episodic import EpisodicMemoryStore
from zenki.memory.semantic import SemanticMemoryStore
from zenki.memory.working import WorkingMemory


class MemoryManager:
    """Central orchestrator for the four-tier memory system.

    Lazily initialises each tier on first access and provides convenience
    methods for common cross-tier operations such as building the full
    context for a prompt.
    """

    def __init__(
        self,
        settings: ZenkiSettings,
        db: ZenkiDatabase,
        config_dir: Path,
    ) -> None:
        self.settings = settings
        self.db = db
        self.config_dir = config_dir

        # Lazy-init backing fields
        self._working: WorkingMemory | None = None
        self._core: CoreMemory | None = None
        self._episodic: EpisodicMemoryStore | None = None
        self._semantic: SemanticMemoryStore | None = None
        self._embedding_provider: EmbeddingProvider | None = None

    # ------------------------------------------------------------------
    # Lazy-initialised properties
    # ------------------------------------------------------------------

    def _get_embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider(
                self.settings.memory.embeddings.model_dump()
            )
        return self._embedding_provider

    @property
    def working(self) -> WorkingMemory:
        """Tier 1: Working memory (active conversation buffer)."""
        if self._working is None:
            budget = self.settings.memory.context_budget
            max_tokens = int(60000 * budget.working_pct / 100)
            self._working = WorkingMemory(max_tokens=max(max_tokens, 1000))
        return self._working

    @property
    def core(self) -> CoreMemory:
        """Tier 2: Core memory (persistent markdown files)."""
        if self._core is None:
            memory_dir = self.config_dir / "memory"
            self._core = CoreMemory(memory_dir=memory_dir)
            self._core.ensure_files()
        return self._core

    @property
    def episodic(self) -> EpisodicMemoryStore:
        """Tier 3: Episodic memory (conversation summaries)."""
        if self._episodic is None:
            self._episodic = EpisodicMemoryStore(
                db=self.db,
                embedding_provider=self._get_embedding_provider(),
            )
        return self._episodic

    @property
    def semantic(self) -> SemanticMemoryStore:
        """Tier 4: Semantic memory (facts, knowledge, insights)."""
        if self._semantic is None:
            self._semantic = SemanticMemoryStore(
                db=self.db,
                embedding_provider=self._get_embedding_provider(),
            )
        return self._semantic

    # ------------------------------------------------------------------
    # Cross-tier operations
    # ------------------------------------------------------------------

    def build_context(self, query: str) -> dict:
        """Build a full context dictionary from all memory tiers.

        This is the primary method used when constructing a prompt. It
        gathers relevant information from each tier and returns it in a
        structured dictionary.

        Args:
            query: The current user query / message to use for retrieval.

        Returns:
            A dictionary with keys ``"working"``, ``"core"``,
            ``"episodic"``, and ``"semantic"``.
        """
        retrieval_cfg = self.settings.memory.retrieval

        # Tier 1 - Working memory
        working_context = self.working.get_context()

        # Tier 2 - Core memory
        core_context = self.core.get_context_string()

        # Tier 3 - Episodic memory
        episodic_results = self.episodic.retrieve(
            query=query,
            top_k=retrieval_cfg.episodic_top_k,
            min_score=retrieval_cfg.min_relevance_score,
        )
        episodic_context = [
            {"summary": mem.summary, "score": score, "topics": mem.key_topics}
            for mem, score in episodic_results
        ]

        # Tier 4 - Semantic memory
        semantic_results = self.semantic.retrieve(
            query=query,
            top_k=retrieval_cfg.semantic_top_k,
            min_score=retrieval_cfg.min_relevance_score,
        )
        semantic_context = [
            {
                "content": mem.content,
                "category": mem.category,
                "score": score,
                "tags": mem.tags,
            }
            for mem, score in semantic_results
        ]

        return {
            "working": working_context,
            "core": core_context,
            "episodic": episodic_context,
            "semantic": semantic_context,
        }

    def store_conversation_summary(
        self,
        session_id: str,
        summary: str,
        topics: list[str],
        entities: list[str],
    ) -> None:
        """Store a conversation summary as an episodic memory.

        Args:
            session_id: The session that was summarised.
            summary: The conversation summary text.
            topics: Key topics from the conversation.
            entities: Key entities mentioned.
        """
        self.episodic.store(
            session_id=session_id,
            summary=summary,
            key_topics=topics,
            key_entities=entities,
        )

    def store_fact(
        self,
        content: str,
        category: str,
        tags: list[str] | None = None,
    ) -> None:
        """Store a fact or piece of knowledge as a semantic memory.

        Args:
            content: The factual content.
            category: The memory category.
            tags: Optional tags.
        """
        self.semantic.store(content=content, category=category, tags=tags)

    def update_core_memory(self, key: str, content: str) -> None:
        """Update a core-memory file.

        Args:
            key: The core memory key (e.g. ``"identity"``).
            content: The new content.
        """
        self.core.update(key, content)
