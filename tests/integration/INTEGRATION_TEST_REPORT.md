# Zenki Integration Test Report

**Date**: 2026-02-28
**Test Suite**: `tests/integration/test_zenki_integration.py`
**Results**: 106 tests passed, 0 failures (full suite: 555 passed)

---

## What Was Tested

### 1. Full Pipeline (Init -> Chat -> Memory)
- Component initialization (database, session manager, memory manager)
- Complete conversation flow: create session -> add messages -> store memory -> retrieve context
- Session resume with preserved history

### 2. Session Management
- Multi-channel sessions (CLI, Slack) for the same user
- **Parallel session isolation** - messages in different sessions don't leak
- `get_or_create` reuse vs. separation semantics
- Session close/reopen lifecycle
- `last_active` timestamp updates on new messages
- **Stress test**: 20 parallel sessions with isolated messages
- Chronological message ordering
- Message limit returning most recent

### 3. Memory System (All 4 Tiers)
**Tier 1 - Working Memory**: Token budget enforcement, clear, summary, total_tokens
**Tier 2 - Core Memory**: File creation, read/write, context string generation, empty section handling
**Tier 3 - Episodic Memory**: Store/retrieve with FK constraints, scoring, importance ranking, get_recent
**Tier 4 - Semantic Memory**: Fact storage, category filtering, multi-category retrieval
**Cross-tier**: build_context gathering all tiers, store and retrieve across tiers, core section append/replace

### 4. Model Routing
- Simple greetings -> haiku (cheapest)
- Complex coding tasks -> opus (most capable)
- Regular queries -> sonnet (balanced default)
- Edge cases: empty string, very long, mixed case

### 5. Error Handling
- Rate limit detection and retry strategy
- Network error detection
- Auth error escalation
- Novel error escalation
- FileNotFoundError / PermissionError handling
- Retry-with-success flow
- Custom pattern registration

### 6. Memory Consolidation
- Importance time-decay calculation
- Access-count boost
- Importance capped at 1.0
- Closed session review and summary generation
- Active session skipping
- Memory importance decay during consolidation

### 7. Edge Cases & Stress
- Empty messages, very long messages (100KB)
- Unicode content (emoji, Japanese, Russian, Arabic)
- Special characters in search queries (C++, C#)
- SQL injection prevention
- 50 concurrent session creation (unique IDs)
- Working memory with near-zero budget
- Database close/reopen data persistence
- Tool calls metadata in messages

### 8. Additional Coverage
- Prompt building with all memory tiers
- Cosine similarity math
- Memory ranking with recency weighting
- Embedding provider factory
- SDK context lifecycle
- Full database CRUD for all entities
- Settings validation and round-trip
- Multi-user isolation

---

## Issues Found During Testing

### Bug: Episodic Memory FK Constraint on session_id
**Severity**: Medium
**Location**: `zenki/memory/episodic.py:50` / `zenki/db/database.py:362`

The `episodic_memories.session_id` column has a FOREIGN KEY constraint on `sessions(id)`.
When calling `EpisodicMemoryStore.store(session_id="some-id")`, if `some-id` doesn't
exist in the sessions table, it raises `sqlite3.IntegrityError: FOREIGN KEY constraint failed`.

This is correct database behavior, but the API doesn't validate or surface this clearly.
The `session_id` field in `EpisodicMemory` is typed as `str | None`, suggesting it should
be optional, but the FK constraint makes it required when non-null.

**Recommendation**: Either:
1. Make `session_id` nullable in the DB schema (already `REFERENCES sessions(id)` without `NOT NULL`), or
2. Add validation in `EpisodicMemoryStore.store()` to check session existence first,
   or catch and wrap the `IntegrityError` with a clear message.

### Issue: SDK @tool Decorator Creates Non-Callable Objects
**Severity**: Low (tooling/testing concern)
**Location**: `zenki/sdk/tools.py`

The `@tool` decorator from `claude_agent_sdk` wraps functions as `SdkMcpTool` objects
that cannot be called directly in tests. This makes it impossible to unit-test tool
functions independently without going through the full MCP server. Consider adding
a `_raw_handler` or similar mechanism to access the underlying async function for testing.

### Issue: process_message Creates New Event Loop on Each Call
**Severity**: Medium
**Location**: `zenki/cli/commands/chat.py:103`

The chat command uses `asyncio.run()` inside a `while True` loop, creating a new event loop
per message. This is fine for CLI use but:
1. Prevents proper `async with` conversation context (ClaudeSDKClient state is lost between messages)
2. Working memory is not populated from session history in `process_message`
3. Each message is a standalone SDK query with no conversation continuity

The `ZenkiConversation` class exists but isn't used in the chat command.

### Observation: Working Memory Not Integrated in process_message
**Severity**: Medium
**Location**: `zenki/sdk/orchestrator.py:292-378`

`process_message()` stores messages in the database (via SessionManager) but doesn't
populate or use `WorkingMemory`. The memory context (`build_context`) is never called
during `process_message`, so the agent doesn't actually receive memory-enriched prompts.

The full pipeline (build_context -> inject into prompt -> query SDK) is only partially wired.

### Observation: Consolidation Doesn't Persist Decayed Importance
**Severity**: Low
**Location**: `zenki/memory/consolidation.py:241-269`

`_decay_memories()` recalculates importance scores but only updates the in-memory model
objects without writing changes back to the database. The decayed values are lost.

---

## Feature Suggestions & Improvements

### High Priority

1. **Wire memory context into process_message**: The `build_context()` method is fully
   implemented but never called during message processing. Adding a call to inject episodic
   and semantic context into the system prompt would make Zenki truly memory-aware.

2. **Use ZenkiConversation in CLI chat**: Replace the `asyncio.run()` per-message pattern
   with a proper `ZenkiConversation` context manager to enable multi-turn context retention
   within the Claude SDK.

3. **Session history replay**: When resuming a session with `--resume`, load previous messages
   into the working memory and/or prepend them to the system prompt so the agent has context.

### Medium Priority

4. **Memory search improvements**: The current LIKE-based search is basic. Consider:
   - TF-IDF scoring instead of simple word matching
   - Trigram similarity for fuzzy matching
   - Stemming/lemmatization for better recall

5. **Session expiry/timeout**: Sessions currently stay "active" forever until explicitly closed.
   Consider auto-closing sessions after inactivity (e.g., 30 minutes without a message).

6. **Streaming in CLI**: The `process_message_streaming` method exists but the CLI uses
   the non-streaming `process_message`. Using streaming would give better UX with
   progressive output rendering.

7. **Persist consolidation decay**: `_decay_memories()` should write updated importance
   scores back to the database.

8. **Chat command /history**: The `/history` command shows "not yet available" - implement
   it by querying the session's messages and displaying them.

### Lower Priority

9. **Working memory shared across sessions**: Currently each `MemoryManager` instance has
   its own `WorkingMemory`. For the daemon, consider session-keyed working memory so
   parallel conversations each have their own buffer.

10. **Memory deduplication**: When storing semantic memories, check for near-duplicate
    content to avoid storing the same fact multiple times.

11. **Context budget enforcement in build_context**: The budget percentages are defined
    in settings but `build_context` doesn't actually enforce token limits per tier.

12. **Export/import memory**: Allow users to export their core memory and semantic
    knowledge as a portable format, and import it on a new installation.

13. **Conversation branching**: Allow creating a new session that inherits context
    from an existing one (like Git branching for conversations).

14. **Model routing with context**: The `ModelRouter.classify_complexity()` accepts
    a `context` parameter but never uses it. Conversation length, tool usage patterns,
    and session topic could improve routing decisions.

15. **Rate limit handling in process_message**: The error handler has retry logic,
    but `process_message` only calls `handler.handle(exc)` without passing a `retry_fn`,
    so rate-limit retries don't actually re-attempt the SDK call.
