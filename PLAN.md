# Zenki - Adaptive AI Assistant Agent
## Comprehensive Architecture & Implementation Plan

---

## 1. Project Overview

**Zenki** (Japanese: "total knowledge / omniscience") is a daemon-based AI assistant agent built on the Claude Agent SDK. It communicates with users through multiple channels (Slack, CLI, future: Telegram, WhatsApp, Web UI), learns and adapts over time through a human-inspired 4-tier memory system, and can acquire new skills autonomously.

### Core Differentiators
- **Human-like memory**: 4-tier memory system (Working, Core, Episodic, Semantic) with vector-based retrieval
- **Self-improvement**: Learns new skills from conversations and runs daily self-improvement cycles
- **Multi-channel**: Communicates via Slack, CLI, and extensible to other channels
- **Always-on daemon**: Runs as a system service with background task execution and scheduling

---

## 2. Requirements Summary

| Requirement | Decision |
|---|---|
| Language | Python |
| LLM SDK | Claude Agent SDK (primary), extensible to others |
| Auth | Anthropic API keys |
| Database | SQLite + sqlite-vec (local, per-user) |
| Embeddings | Local (all-MiniLM-L6-v2) default, API optional |
| Primary interface | Slack (webhook/HTTP mode) |
| Secondary interface | CLI (Rich TUI) - admin + optional chat |
| Skills format | Claude-style full skill directories |
| Skill approval | Always require user approval |
| Memory tiers | 4 (Working, Core, Episodic, Semantic) |
| Core skills | Software Engineering, Web Research, File Management, System Operations |
| OS support | macOS, Linux, Windows |
| Config format | JSON |
| Model routing | Smart (Haiku / Sonnet / Opus by task complexity) |
| Tools | MCP + custom Python tools |
| Scheduling | Built-in scheduler with natural language interface |
| Daemon | System service (systemd / launchd / Windows service) |
| Conversations | Full logs + summaries + periodic consolidation |
| Privacy | Local storage, user-managed sync |
| Packaging | Standard Python package (pipx install) |
| Onboarding | CLI wizard + self-guided setup via default channel |
| Personality | User-configurable |
| Multi-user | Single-user, multi-user ready (data model supports it) |
| Error handling | Smart escalation (auto-recover known errors, ask user for novel ones) |
| Deployment | Server (native) or local + tunnel (ngrok/Cloudflare) |

---

## 3. System Architecture

```
                         ┌─────────────────────────────────┐
                         │         ZENKI DAEMON             │
                         │                                  │
  ┌──────────┐  HTTP     │  ┌───────────────────────────┐  │
  │  Slack   │◄─────────►│  │    Channel Router         │  │
  │  API     │  webhook  │  │    (event dispatcher)     │  │
  └──────────┘           │  └─────────┬─────────────────┘  │
                         │            │                     │
  ┌──────────┐  stdin/   │  ┌─────────▼─────────────────┐  │
  │  CLI     │◄─────────►│  │    Session Manager        │  │
  │  (Rich)  │  stdout   │  │    (per-channel rules)    │  │
  └──────────┘           │  └─────────┬─────────────────┘  │
                         │            │                     │
  ┌──────────┐  HTTP     │  ┌─────────▼─────────────────┐  │
  │ Telegram │◄─────────►│  │    Agent Core             │  │
  │ (future) │  webhook  │  │    ┌───────────────────┐  │  │
  └──────────┘           │  │    │ Claude Agent SDK  │  │  │
                         │  │    │ (conversation +   │  │  │
                         │  │    │  tool execution)  │  │  │
                         │  │    └───────────────────┘  │  │
                         │  └──┬──────┬──────┬──────┬───┘  │
                         │     │      │      │      │      │
                    ┌────┴─────┴──┐ ┌─┴──┐ ┌─┴──┐ ┌┴────┐ │
                    │  Memory     │ │Skill│ │Tool│ │Sched│ │
                    │  System     │ │Sys  │ │Sys │ │uler │ │
                    │             │ │     │ │    │ │     │ │
                    │ Working     │ │Core │ │MCP │ │Cron │ │
                    │ Core        │ │Learn│ │Cust│ │NL   │ │
                    │ Episodic    │ │Pend │ │    │ │     │ │
                    │ Semantic    │ │     │ │    │ │     │ │
                    └──────┬──────┘ └────┘ └────┘ └─────┘ │
                           │                               │
                    ┌──────▼──────────────────────────────┐│
                    │  SQLite + sqlite-vec                 ││
                    │  (conversations, embeddings, tasks)  ││
                    └─────────────────────────────────────┘│
                         └─────────────────────────────────┘
```

---

## 4. Project Structure

### 4.1 Source Code (ships with Zenki via pip/pipx)

```
zenki/
├── pyproject.toml                    # Package config, dependencies, entry points
├── README.md
├── LICENSE
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI/CD pipeline
│
├── src/
│   └── zenki/
│       ├── __init__.py               # Package init, version
│       ├── __main__.py               # python -m zenki entry point
│       │
│       ├── cli/                      # CLI module (Rich TUI)
│       │   ├── __init__.py
│       │   ├── app.py                # Main CLI app (Typer-based)
│       │   ├── commands/
│       │   │   ├── __init__.py
│       │   │   ├── start.py          # zenki start [--foreground]
│       │   │   ├── stop.py           # zenki stop
│       │   │   ├── status.py         # zenki status
│       │   │   ├── chat.py           # zenki chat (direct interaction channel)
│       │   │   ├── setup.py          # zenki setup (first-time wizard)
│       │   │   ├── config.py         # zenki config [get|set|list]
│       │   │   ├── memory.py         # zenki memory [inspect|search|clear]
│       │   │   ├── skills.py         # zenki skills [list|approve|reject|info]
│       │   │   ├── schedule.py       # zenki schedule [list|add|remove]
│       │   │   └── logs.py           # zenki logs [--follow]
│       │   └── tui/
│       │       ├── __init__.py
│       │       ├── chat_view.py      # Rich chat interface
│       │       ├── status_view.py    # Daemon status display
│       │       └── setup_wizard.py   # Interactive setup wizard
│       │
│       ├── core/                     # Core agent logic
│       │   ├── __init__.py
│       │   ├── agent.py              # Main agent orchestrator
│       │   ├── session.py            # Session lifecycle management
│       │   ├── context_builder.py    # Builds context from memory tiers
│       │   ├── prompts.py            # System prompts and templates
│       │   └── errors.py             # Error handling + smart escalation
│       │
│       ├── llm/                      # LLM abstraction layer
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract LLM provider interface
│       │   ├── claude_provider.py    # Claude via Agent SDK
│       │   ├── router.py             # Smart model routing logic
│       │   └── config.py             # Model configuration
│       │
│       ├── memory/                   # 4-Tier memory system
│       │   ├── __init__.py
│       │   ├── manager.py            # Memory manager (orchestrates all tiers)
│       │   ├── working.py            # Tier 1: Working memory (in-session context)
│       │   ├── core_memory.py        # Tier 2: Core memory (identity, prefs - always loaded)
│       │   ├── episodic.py           # Tier 3: Episodic memory (conversation summaries)
│       │   ├── semantic.py           # Tier 4: Semantic memory (facts, knowledge)
│       │   ├── consolidation.py      # Memory consolidation engine
│       │   ├── embeddings.py         # Embedding interface (local + API)
│       │   └── retrieval.py          # Retrieval logic (KNN search, ranking)
│       │
│       ├── skills/                   # Skill system
│       │   ├── __init__.py
│       │   ├── loader.py             # Skill discovery + progressive loading
│       │   ├── executor.py           # Skill execution engine
│       │   ├── generator.py          # Skill generation (agent creates new skills)
│       │   ├── registry.py           # Skill registry + approval workflow
│       │   └── core_skills/          # Built-in skills (ship with Zenki)
│       │       ├── software_engineering/
│       │       │   ├── SKILL.md
│       │       │   ├── templates/
│       │       │   │   ├── pr_template.md
│       │       │   │   └── code_review.md
│       │       │   └── scripts/
│       │       │       └── git_helpers.py
│       │       ├── web_research/
│       │       │   ├── SKILL.md
│       │       │   └── templates/
│       │       │       └── research_report.md
│       │       ├── file_management/
│       │       │   └── SKILL.md
│       │       └── system_operations/
│       │           └── SKILL.md
│       │
│       ├── tools/                    # Tool system
│       │   ├── __init__.py
│       │   ├── mcp_manager.py        # MCP server lifecycle management
│       │   ├── registry.py           # Unified tool registry (MCP + custom)
│       │   └── custom/               # Custom Python tools
│       │       ├── __init__.py
│       │       ├── memory_tools.py   # Tools for memory read/write/search
│       │       ├── skill_tools.py    # Tools for skill management
│       │       ├── schedule_tools.py # Tools for scheduling tasks
│       │       └── notification.py   # Tools for sending notifications
│       │
│       ├── channels/                 # Communication channels
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract channel interface
│       │   ├── registry.py           # Channel registry + routing
│       │   ├── message.py            # Unified message model
│       │   ├── slack/                # Slack channel adapter
│       │   │   ├── __init__.py
│       │   │   ├── adapter.py        # Slack event handling + webhooks
│       │   │   ├── session_rules.py  # Slack-specific session rules
│       │   │   ├── formatter.py      # Message formatting (Slack blocks/mrkdwn)
│       │   │   └── setup.py          # Slack app setup helper
│       │   └── cli_channel/          # CLI as a channel
│       │       ├── __init__.py
│       │       └── adapter.py        # CLI channel adapter
│       │
│       ├── scheduler/                # Task scheduling
│       │   ├── __init__.py
│       │   ├── scheduler.py          # Main scheduler (APScheduler-based)
│       │   ├── tasks.py              # Task definitions and persistence
│       │   └── nl_parser.py          # Natural language → cron expression
│       │
│       ├── daemon/                   # Daemon management
│       │   ├── __init__.py
│       │   ├── server.py             # HTTP server (aiohttp/FastAPI for webhooks)
│       │   ├── daemon.py             # Daemon lifecycle (start, stop, health)
│       │   └── service/              # System service installers
│       │       ├── __init__.py
│       │       ├── systemd.py        # Linux systemd service
│       │       ├── launchd.py        # macOS launchd plist
│       │       └── windows.py        # Windows service
│       │
│       ├── db/                       # Database layer
│       │   ├── __init__.py
│       │   ├── database.py           # SQLite + sqlite-vec connection management
│       │   ├── models.py             # Data models (dataclasses/Pydantic)
│       │   └── migrations.py         # Schema versioning and migrations
│       │
│       └── config/                   # Configuration management
│           ├── __init__.py
│           ├── settings.py           # Settings loader + validator
│           └── defaults.py           # Default configuration values
│
└── tests/
    ├── __init__.py
    ├── conftest.py                   # Shared fixtures
    ├── unit/
    │   ├── test_memory.py
    │   ├── test_skills.py
    │   ├── test_session.py
    │   ├── test_router.py
    │   └── test_scheduler.py
    └── integration/
        ├── test_agent.py
        ├── test_slack.py
        └── test_memory_retrieval.py
```

### 4.2 User Data Directory (~/.config/zenki/)

This directory is created per-user during setup. It contains all user-specific data.

```
~/.config/zenki/
├── config.json                       # Main configuration file
│
├── memory/                           # Core memory files (Tier 2)
│   ├── identity.md                   # Who the user is
│   ├── preferences.md                # Communication style, tools, coding prefs
│   ├── projects.md                   # Active projects, repos, tech stacks
│   ├── relationships.md              # People, teams, org structure
│   ├── patterns.md                   # Observed habits, common workflows
│   └── personality.md                # Zenki's configured personality/tone
│
├── conversations/                    # Conversation logs (Tier 3 source)
│   ├── 2026-02-27/
│   │   ├── sess_a1b2c3.json          # Full conversation log
│   │   └── sess_a1b2c3.summary.md   # Auto-generated session summary
│   └── consolidations/              # Periodic consolidation outputs
│       ├── 2026-w09.md               # Weekly consolidation
│       └── 2026-02.md                # Monthly consolidation
│
├── skills/                           # Learned skills (per-user)
│   ├── pending/                      # Awaiting user approval
│   │   └── <skill-name>/
│   │       ├── SKILL.md
│   │       └── ...
│   └── approved/                     # Approved and active
│       └── <skill-name>/
│           ├── SKILL.md
│           ├── templates/
│           ├── examples/
│           └── scripts/
│
├── db/
│   └── zenki.db                      # SQLite database
│
├── mcp/
│   └── servers.json                  # User-configured MCP servers
│
└── logs/
    ├── zenki.log                     # Application log
    └── zenki.log.1                   # Rotated logs
```

---

## 5. Database Schema

### 5.1 Core Tables

```sql
-- User profile (single-user, multi-user ready)
CREATE TABLE users (
    id TEXT PRIMARY KEY DEFAULT 'default',
    display_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config JSON,
    default_channel TEXT                -- preferred notification channel
);

-- Sessions (channel-aware)
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    channel_type TEXT NOT NULL,         -- 'slack', 'cli', 'telegram', etc.
    channel_id TEXT,                    -- channel-specific identifier (e.g., Slack channel ID)
    thread_id TEXT,                     -- thread within channel (e.g., Slack thread_ts)
    context_summary TEXT,              -- summary of channel context when session started
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    last_active TIMESTAMP,
    summary TEXT,                       -- auto-generated session summary
    metadata JSON
);

-- Messages (full conversation log)
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,                 -- 'user', 'assistant', 'system', 'tool'
    content TEXT NOT NULL,
    model_used TEXT,                    -- which Claude model was used
    tokens_in INTEGER,
    tokens_out INTEGER,
    tool_calls JSON,                   -- tool calls made in this message
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_messages_session ON messages(session_id, created_at);
```

### 5.2 Memory Tables

```sql
-- Episodic memories (Tier 3 - conversation summaries)
CREATE TABLE episodic_memories (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    summary TEXT NOT NULL,
    key_topics JSON,                   -- ["topic1", "topic2", ...]
    key_entities JSON,                 -- ["person1", "project_x", ...]
    importance REAL DEFAULT 0.5,       -- 0.0 to 1.0, decays over time
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Semantic memories (Tier 4 - facts, knowledge, insights)
CREATE TABLE semantic_memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT NOT NULL,            -- 'fact', 'preference', 'knowledge', 'insight', 'lesson'
    source TEXT,                       -- where this was learned (session_id, consolidation, etc.)
    tags JSON,                         -- ["python", "deployment", ...]
    importance REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

-- Vector embeddings for semantic search (sqlite-vec)
CREATE VIRTUAL TABLE memory_vec USING vec0(
    memory_id TEXT,
    memory_type TEXT,                  -- 'episodic' or 'semantic'
    embedding FLOAT[384]              -- all-MiniLM-L6-v2 produces 384-dim vectors
);

CREATE INDEX idx_episodic_importance ON episodic_memories(importance DESC);
CREATE INDEX idx_semantic_category ON semantic_memories(category);
CREATE INDEX idx_semantic_importance ON semantic_memories(importance DESC);
```

### 5.3 Scheduling & Skills Tables

```sql
-- Scheduled tasks
CREATE TABLE scheduled_tasks (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    description TEXT NOT NULL,         -- human-readable description
    original_request TEXT,             -- the natural language request that created this
    cron_expression TEXT NOT NULL,     -- standard cron format
    next_run_at TIMESTAMP,
    last_run_at TIMESTAMP,
    last_result TEXT,
    task_type TEXT NOT NULL,           -- 'reminder', 'action', 'self_improve', 'monitor'
    task_config JSON NOT NULL,         -- what to execute (prompt, skill, etc.)
    notify_channel TEXT,               -- where to send results
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Learned skills registry
CREATE TABLE skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    skill_type TEXT NOT NULL,          -- 'core' or 'learned'
    status TEXT DEFAULT 'pending',     -- 'pending', 'approved', 'rejected', 'disabled'
    path TEXT NOT NULL,                -- filesystem path to skill directory
    description TEXT,
    generated_from TEXT,               -- session_id or 'consolidation' or 'manual'
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP,
    approved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

CREATE INDEX idx_tasks_next_run ON scheduled_tasks(next_run_at) WHERE enabled = TRUE;
CREATE INDEX idx_skills_status ON skills(status);
```

---

## 6. Memory System Design (4-Tier, Human-Inspired)

### 6.1 Tier Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        CONTEXT WINDOW                              │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  TIER 1: WORKING MEMORY                                      │  │
│  │  (Current conversation, active task state)                   │  │
│  │  Storage: In-memory    │  Budget: ~60% of context            │  │
│  │  Human analogy: What you're currently thinking about         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  TIER 2: CORE MEMORY                                         │  │
│  │  (Identity, preferences, always-needed facts)                │  │
│  │  Storage: Markdown files    │  Budget: ~10% of context       │  │
│  │  Human analogy: Things you always know (your name, etc.)     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  TIER 3: EPISODIC MEMORY (retrieved on demand)               │  │
│  │  (Past conversation summaries, key events)                   │  │
│  │  Storage: SQLite + vectors  │  Budget: ~15% of context       │  │
│  │  Retrieval: Semantic search (embed query → KNN → top-k)      │  │
│  │  Human analogy: Memories of specific events/conversations    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  TIER 4: SEMANTIC MEMORY (retrieved on demand)               │  │
│  │  (Facts, knowledge, learned insights)                        │  │
│  │  Storage: SQLite + vectors  │  Budget: ~15% of context       │  │
│  │  Retrieval: Semantic search + category filtering             │  │
│  │  Human analogy: General knowledge you've accumulated         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 6.2 Core Memory Files (Tier 2)

**identity.md** - Who the user is
```markdown
# Identity
- Name: [User's name]
- Role: [Job title / role]
- Organization: [Company / team]
- Location: [Timezone-relevant info]
- Bio: [Brief description]
```

**preferences.md** - How the user likes to work
```markdown
# Preferences
## Communication
- Verbosity: [concise | balanced | detailed]
- Tone: [casual | professional | technical]
- Language: [en | etc.]

## Development
- Primary languages: [Python, TypeScript, etc.]
- Editor: [VSCode, Vim, etc.]
- Git workflow: [trunk-based, gitflow, etc.]
- Code style: [specific preferences]

## Tools
- Git hosting: [GitHub | GitLab | etc.]
- CI/CD: [GitHub Actions | GitLab CI | etc.]
- Deployment: [specific preferences]
```

**projects.md** - Active projects
```markdown
# Projects
## [Project Name]
- Repo: [URL]
- Local path: [/path/to/project]
- Tech stack: [languages, frameworks]
- Branch strategy: [main, develop, etc.]
- Status: [active | maintenance | archived]
- Notes: [anything relevant]
```

**relationships.md** - People and teams
```markdown
# Relationships
## [Person Name]
- Role: [their role]
- Relationship: [manager, teammate, client, etc.]
- Communication: [Slack handle, email, etc.]
- Notes: [preferences, relevant context]
```

**patterns.md** - Observed user patterns (auto-updated)
```markdown
# Patterns
## Work Habits
- Active hours: [typically 9am-6pm PST]
- Common requests: [code reviews, deployments, etc.]

## Recurring Tasks
- [Pattern description]

## Common Mistakes/Issues
- [Pattern description]
```

**personality.md** - Zenki's configured personality
```markdown
# Zenki Personality
- Name: Zenki (or user-chosen name)
- Tone: [helpful, professional, with light humor]
- Proactivity: [suggest improvements, flag issues]
- Emoji use: [minimal | moderate | frequent]
- Custom instructions: [user-specific adjustments]
```

### 6.3 Memory Retrieval Flow

```
User sends message
        │
        ▼
┌─────────────────────┐
│ Load Core Memory    │ ← Always loaded (Tier 2)
│ (identity, prefs)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Embed user message  │ ← all-MiniLM-L6-v2 (local)
└─────────┬───────────┘
          │
          ├──────────────────────┐
          ▼                      ▼
┌──────────────────┐   ┌──────────────────┐
│ Search Episodic  │   │ Search Semantic  │ ← Parallel KNN search
│ (top-k summaries)│   │ (top-k facts)    │
└─────────┬────────┘   └────────┬─────────┘
          │                      │
          ▼                      ▼
┌─────────────────────────────────────────┐
│ Rank & Filter                           │
│ - Relevance score (cosine similarity)   │
│ - Recency boost                         │
│ - Importance weight                     │
│ - Access frequency                      │
│ - Context budget check                  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Build Context                           │
│ System prompt + Core memory             │
│ + Retrieved episodic memories           │
│ + Retrieved semantic memories           │
│ + Working memory (conversation)         │
│ + Available skills metadata             │
└─────────────────────────────────────────┘
                  │
                  ▼
          Send to Claude
```

### 6.4 Memory Importance & Decay

Inspired by human memory, memories have an **importance score** that decays over time but is reinforced by access:

```python
def calculate_importance(memory, current_time):
    """
    Importance = base_importance * recency_factor * access_factor

    - base_importance: Set when memory is created (0.0 to 1.0)
    - recency_factor: Exponential decay based on time since creation
    - access_factor: Boost based on how often the memory is retrieved
    """
    days_old = (current_time - memory.created_at).days
    recency = math.exp(-0.01 * days_old)  # Slow decay

    access_boost = min(1.0 + (memory.access_count * 0.1), 2.0)  # Cap at 2x

    return memory.base_importance * recency * access_boost
```

### 6.5 Memory Consolidation (Self-Improvement Cycle)

Runs on a configurable schedule (default: daily at 3 AM):

```
Daily Consolidation Cycle
         │
         ├─── 1. Review Recent Conversations
         │    └── Summarize any un-summarized sessions
         │
         ├─── 2. Extract New Knowledge
         │    ├── Identify new facts → Semantic Memory
         │    ├── Identify new preferences → Update Core Memory
         │    └── Identify new entities → Update relationships.md
         │
         ├─── 3. Identify Skill Opportunities
         │    ├── Detect repeated patterns
         │    ├── Generate skill proposals
         │    └── Submit for user approval
         │
         ├─── 4. Consolidate Episodic Memory
         │    ├── Weekly: Merge daily summaries into weekly
         │    └── Monthly: Merge weekly into monthly themes
         │
         ├─── 5. Decay & Prune
         │    ├── Decay importance of old, unused memories
         │    └── Archive memories below threshold
         │
         └─── 6. Self-Assessment
              ├── Review errors and failures
              ├── Generate improvement insights
              └── Log consolidation report
```

---

## 7. Channel Architecture

### 7.1 Base Channel Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class IncomingMessage:
    """Unified message format from any channel."""
    content: str
    user_id: str
    channel_type: str          # 'slack', 'cli', 'telegram'
    channel_id: str            # channel-specific identifier
    thread_id: Optional[str]   # thread within channel
    session_id: Optional[str]  # existing session to continue
    metadata: dict             # channel-specific extras

class BaseChannel(ABC):
    """Abstract base class for all communication channels."""

    @abstractmethod
    async def start(self) -> None:
        """Start listening for messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel gracefully."""

    @abstractmethod
    async def send_message(self, channel_id: str, content: str,
                           thread_id: str = None) -> None:
        """Send a message to a specific channel/thread."""

    @abstractmethod
    async def send_notification(self, user_id: str, content: str) -> None:
        """Send a proactive notification to the user."""

    @abstractmethod
    def format_message(self, content: str) -> str:
        """Format a message for this channel (markdown → Slack blocks, etc.)."""

    @abstractmethod
    def get_session_rules(self) -> dict:
        """Return channel-specific session management rules."""
```

### 7.2 Slack Channel

**Session rules:**
- One session per Slack thread
- New thread → new session (with channel context summary)
- DM → session per conversation
- When invoked in a channel, summarize recent channel messages as context

**Webhook endpoints:**
- `POST /slack/events` - Slack Events API
- `POST /slack/interactions` - Interactive components (buttons for skill approval)
- `POST /slack/commands` - Slash commands (e.g., `/zenki`)

### 7.3 CLI Channel

**Session rules:**
- Each `zenki chat` invocation starts a new session by default
- `zenki chat --resume [session-id]` continues a previous session
- Session persists until the user exits (Ctrl+C or `/quit`)

---

## 8. Skill System

### 8.1 Skill Format (Claude-style)

Every skill is a directory containing at minimum a `SKILL.md`:

```
<skill-name>/
├── SKILL.md              # Required: metadata + instructions
├── templates/            # Optional: output templates
│   └── *.md
├── examples/             # Optional: example outputs
│   └── *.md
├── scripts/              # Optional: Python helper scripts
│   └── *.py
└── reference/            # Optional: reference material
    └── *.md
```

**SKILL.md format:**
```yaml
---
name: skill-name
description: What this skill does and when to use it
allowed-tools: Read, Grep, Bash
disable-model-invocation: false
---

# Skill Name

Instructions for Claude when this skill is activated...

## Steps
1. ...
2. ...

## Templates
Reference: templates/output.md

## Notes
- Important considerations
```

### 8.2 Progressive Skill Loading

```
Level 1 (Always loaded): name + description from YAML frontmatter (~100 tokens)
Level 2 (On activation): Full SKILL.md content (~500-2000 tokens)
Level 3 (On reference):  Supporting files only when referenced
```

### 8.3 Skill Learning Pipeline

```
┌────────────────────────────┐
│ Pattern Detection          │
│ (during conversation or    │
│  consolidation cycle)      │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Skill Generation           │
│ - Generate SKILL.md        │
│ - Generate templates       │
│ - Generate scripts (if     │
│   needed)                  │
│ - Save to skills/pending/  │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ User Notification          │
│ "I've learned a potential  │
│  new skill: [name].        │
│  Review and approve?"      │
│                            │
│ [Approve] [Reject] [Edit]  │
└──────────┬─────────────────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
 Approve Reject  Edit
    │      │      │
    │      │      └──► User modifies, then approves
    │      │
    │      └──► Delete from pending/
    │
    └──► Move to approved/, register in DB
```

### 8.4 Core Skills

**1. Software Engineering**
- Clone/pull repositories (GitHub, GitLab)
- Navigate and understand codebases
- Make code changes, create branches
- Create and manage PRs/MRs
- Monitor review comments, respond or address them
- Track project status across repos
- Remember cloned project locations

**2. Web Research**
- Search the web for information
- Fetch and parse web pages
- Summarize content
- Compare multiple sources
- Generate research reports

**3. File Management**
- Read, write, edit files
- Navigate directory structures
- Organize files and projects
- Search file contents
- Manage file permissions

**4. System Operations**
- Execute shell commands
- Manage processes
- Check system status
- Install/manage packages
- Environment management

---

## 9. Smart Model Routing

```python
class ModelRouter:
    """Routes tasks to appropriate Claude models based on complexity."""

    ROUTING_RULES = {
        # Simple/fast tasks → Haiku
        'haiku': [
            'greeting', 'acknowledgment', 'simple_question',
            'memory_lookup', 'status_check', 'formatting',
            'classification', 'sentiment'
        ],
        # General tasks → Sonnet
        'sonnet': [
            'conversation', 'planning', 'summarization',
            'code_explanation', 'research', 'skill_execution',
            'notification_drafting', 'schedule_parsing'
        ],
        # Complex reasoning → Opus
        'opus': [
            'code_generation', 'code_review', 'debugging',
            'architecture_design', 'complex_analysis',
            'skill_generation', 'memory_consolidation',
            'multi_step_reasoning'
        ]
    }
```

The router analyzes the incoming message and task context to classify complexity, then selects the appropriate model. Users can override with a preference in config.

---

## 10. Daemon Architecture

### 10.1 Components

```
Zenki Daemon Process
├── HTTP Server (aiohttp)
│   ├── /slack/events          # Slack webhook endpoint
│   ├── /slack/interactions    # Slack interactive components
│   ├── /health                # Health check endpoint
│   └── /api/v1/*              # Future: REST API for web UI
│
├── Channel Manager
│   ├── Slack Adapter (always running if configured)
│   └── [Future adapters]
│
├── Agent Core
│   ├── Claude Agent SDK instance
│   ├── Session Manager
│   └── Memory Manager
│
├── Scheduler (APScheduler)
│   ├── Consolidation job (daily)
│   ├── User-scheduled tasks
│   └── Background task monitor
│
└── Background Task Runner
    ├── Task queue (asyncio)
    └── Result notification
```

### 10.2 Lifecycle

```bash
# First-time setup
zenki setup                    # Interactive wizard

# Daemon management
zenki start                    # Start daemon (background)
zenki start --foreground       # Start in foreground (dev mode)
zenki stop                     # Graceful shutdown
zenki restart                  # Restart daemon
zenki status                   # Show daemon status

# System service
zenki service install          # Install as system service
zenki service uninstall        # Remove system service
zenki service status           # Check service status

# Direct interaction
zenki chat                     # Chat via CLI (new session)
zenki chat --resume <id>       # Resume session

# Management
zenki skills list              # List all skills
zenki skills approve <name>    # Approve a pending skill
zenki skills info <name>       # Show skill details
zenki memory search "query"    # Search memory
zenki memory inspect           # Show memory stats
zenki schedule list            # List scheduled tasks
zenki config set key value     # Update configuration
zenki logs --follow            # Tail daemon logs
```

### 10.3 System Service

**Linux (systemd):**
```ini
[Unit]
Description=Zenki AI Assistant Daemon
After=network.target

[Service]
Type=simple
User=%i
ExecStart=/usr/local/bin/zenki start --foreground
Restart=on-failure
RestartSec=5
Environment=ZENKI_CONFIG_DIR=%h/.config/zenki

[Install]
WantedBy=multi-user.target
```

**macOS (launchd):**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "...">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.zenki.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/zenki</string>
        <string>start</string>
        <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

---

## 11. Configuration

### 11.1 config.json Structure

```json
{
  "version": "1.0.0",
  "user": {
    "id": "default",
    "display_name": "User Name"
  },
  "llm": {
    "provider": "claude",
    "api_key_env": "ANTHROPIC_API_KEY",
    "default_model": "sonnet",
    "smart_routing": true,
    "models": {
      "haiku": "claude-haiku-4-5-20251001",
      "sonnet": "claude-sonnet-4-6",
      "opus": "claude-opus-4-6"
    }
  },
  "memory": {
    "embeddings": {
      "provider": "local",
      "model": "all-MiniLM-L6-v2",
      "api_provider": null,
      "api_model": null
    },
    "retrieval": {
      "episodic_top_k": 5,
      "semantic_top_k": 10,
      "min_relevance_score": 0.3
    },
    "consolidation": {
      "enabled": true,
      "schedule": "0 3 * * *",
      "weekly_consolidation": true,
      "monthly_consolidation": true
    },
    "context_budget": {
      "core_memory_pct": 10,
      "episodic_pct": 15,
      "semantic_pct": 15,
      "working_pct": 60
    }
  },
  "channels": {
    "default_notification_channel": "slack",
    "slack": {
      "enabled": true,
      "bot_token_env": "ZENKI_SLACK_BOT_TOKEN",
      "signing_secret_env": "ZENKI_SLACK_SIGNING_SECRET",
      "app_token_env": "ZENKI_SLACK_APP_TOKEN"
    }
  },
  "daemon": {
    "host": "0.0.0.0",
    "port": 8420,
    "log_level": "INFO"
  },
  "skills": {
    "auto_discover": true,
    "require_approval": true
  },
  "personality": {
    "tone": "professional",
    "verbosity": "balanced",
    "proactivity": "moderate",
    "custom_instructions": ""
  }
}
```

---

## 12. Error Handling (Smart Escalation)

```
Error occurs
     │
     ▼
┌──────────────────┐
│ Classify error   │
│ Known? Retryable?│
└──────┬───────────┘
       │
  ┌────┴────┐
  ▼         ▼
Known    Unknown/Novel
  │         │
  ▼         ▼
┌────────┐  ┌──────────────────┐
│ Auto-  │  │ Notify user      │
│ recover│  │ "I encountered   │
│        │  │  an issue: ..."  │
│ Retry  │  │ "How should I    │
│ Fallback│ │  proceed?"       │
│ Alt.   │  └──────────────────┘
│ approach│
└────────┘

Known error patterns:
- API rate limit → exponential backoff + retry
- Network timeout → retry with backoff
- File not found → search for alternatives
- Permission denied → request elevated access
- Git conflict → attempt auto-merge, escalate if complex
- Tool failure → try alternative tool/approach
```

---

## 13. Key Dependencies

```toml
[project]
name = "zenki"
requires-python = ">=3.11"

dependencies = [
    # Core
    "claude-agent-sdk>=0.1.0",        # Claude Agent SDK
    "typer[all]>=0.12.0",             # CLI framework
    "rich>=13.0",                      # Rich terminal output

    # Database
    "sqlite-vec>=0.1.0",              # Vector search for SQLite

    # Embeddings
    "sentence-transformers>=3.0",     # Local embedding models

    # Web server (daemon)
    "aiohttp>=3.9",                   # Async HTTP server for webhooks

    # Slack
    "slack-bolt>=1.18",               # Slack SDK (bolt framework)

    # Scheduling
    "apscheduler>=3.10",              # Task scheduling

    # Utilities
    "pydantic>=2.0",                  # Data validation / models
    "python-dotenv>=1.0",             # Environment variable loading
    "pyyaml>=6.0",                    # YAML parsing (for SKILL.md frontmatter)
    "python-frontmatter>=1.1",        # YAML frontmatter parsing
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "ruff>=0.4",                      # Linting + formatting
    "mypy>=1.10",                     # Type checking
]

[project.scripts]
zenki = "zenki.cli.app:app"
```

---

## 14. Implementation Phases

### Phase 1: Foundation (Core Agent + CLI) ← START HERE
**Goal:** A working CLI agent that can hold conversations via Claude Agent SDK.

1. **Project scaffolding**
   - Initialize pyproject.toml with dependencies
   - Create src/zenki package structure
   - Set up development tooling (ruff, mypy, pytest)

2. **Configuration system**
   - `config/settings.py` - Load/save JSON config
   - `config/defaults.py` - Default values
   - Config directory management (~/.config/zenki/)

3. **Database setup**
   - `db/database.py` - SQLite connection management
   - `db/models.py` - Pydantic data models
   - `db/migrations.py` - Schema creation and versioning

4. **LLM abstraction layer**
   - `llm/base.py` - Abstract provider interface
   - `llm/claude_provider.py` - Claude Agent SDK integration
   - Basic conversation support (send message → get response)

5. **Core agent**
   - `core/agent.py` - Main agent orchestrator
   - `core/session.py` - Session management (create, resume, close)
   - `core/prompts.py` - System prompt templates

6. **CLI**
   - `cli/app.py` - Typer app with subcommands
   - `cli/commands/setup.py` - First-time setup wizard
   - `cli/commands/chat.py` - Interactive chat (basic)
   - `cli/tui/chat_view.py` - Rich chat interface

**Deliverable:** `zenki setup` + `zenki chat` works end-to-end.

---

### Phase 2: Memory System
**Goal:** Full 4-tier memory with vector-based retrieval.

1. **Embedding system**
   - `memory/embeddings.py` - Local (sentence-transformers) + API providers
   - sqlite-vec integration for vector storage

2. **Core memory (Tier 2)**
   - `memory/core_memory.py` - Load/update markdown files
   - Initialize default memory files during setup
   - Always inject into system prompt

3. **Episodic memory (Tier 3)**
   - `memory/episodic.py` - Store conversation summaries
   - Auto-summarize sessions on close
   - Vector-based retrieval (embed query → KNN → top-k)

4. **Semantic memory (Tier 4)**
   - `memory/semantic.py` - Store facts and knowledge
   - Extract facts from conversations
   - Vector-based retrieval with category filtering

5. **Working memory (Tier 1)**
   - `memory/working.py` - Active context management
   - Context budget enforcement
   - Sliding window with summarization

6. **Memory manager**
   - `memory/manager.py` - Orchestrate all tiers
   - `memory/retrieval.py` - Unified retrieval logic
   - `core/context_builder.py` - Build prompt context from all tiers

7. **Custom tools for memory**
   - `tools/custom/memory_tools.py` - Tools the agent can use to read/write memory

**Deliverable:** Agent remembers across sessions, retrieves relevant context.

---

### Phase 3: Skill System
**Goal:** Skill discovery, execution, learning, and approval workflow.

1. **Skill loader**
   - `skills/loader.py` - Discover skills from filesystem
   - Progressive loading (metadata → full → resources)
   - YAML frontmatter parsing

2. **Skill executor**
   - `skills/executor.py` - Execute skills (inject into context, run scripts)
   - String substitution ($ARGUMENTS, etc.)
   - Tool restriction (allowed-tools)

3. **Core skills**
   - Implement all four core skills with SKILL.md + supporting files
   - MCP integration for Git (GitHub/GitLab MCP servers)

4. **Skill generator**
   - `skills/generator.py` - Generate new skill directories
   - Pattern detection from conversations
   - Template-based generation

5. **Approval workflow**
   - `skills/registry.py` - Skill status management
   - `tools/custom/skill_tools.py` - Tools for skill approval
   - CLI commands: `zenki skills list/approve/reject`

6. **MCP + custom tools**
   - `tools/mcp_manager.py` - MCP server lifecycle
   - `tools/registry.py` - Unified tool registry

**Deliverable:** Agent can use core skills and learn new ones.

---

### Phase 4: Daemon + Slack Channel
**Goal:** Zenki runs as a daemon and communicates via Slack.

1. **HTTP server**
   - `daemon/server.py` - aiohttp server for webhooks
   - Health check endpoint
   - Request signing verification (Slack)

2. **Daemon lifecycle**
   - `daemon/daemon.py` - Start, stop, health check
   - PID file management
   - Graceful shutdown

3. **System service**
   - `daemon/service/systemd.py` - Linux service
   - `daemon/service/launchd.py` - macOS service
   - `daemon/service/windows.py` - Windows service
   - CLI: `zenki service install/uninstall/status`

4. **Channel base + registry**
   - `channels/base.py` - Abstract channel interface
   - `channels/message.py` - Unified message model
   - `channels/registry.py` - Channel management

5. **Slack adapter**
   - `channels/slack/adapter.py` - Event handling via slack-bolt
   - `channels/slack/session_rules.py` - Thread-based sessions
   - `channels/slack/formatter.py` - Markdown → Slack blocks
   - `channels/slack/setup.py` - Slack app setup helper

6. **CLI channel adapter**
   - `channels/cli_channel/adapter.py` - CLI as a channel

7. **Notification system**
   - `tools/custom/notification.py` - Channel-aware notifications
   - Default notification channel from config

8. **CLI admin commands**
   - `zenki start/stop/status/restart`
   - `zenki logs --follow`
   - `zenki service install`

**Deliverable:** Zenki runs as a system service, responds on Slack.

---

### Phase 5: Scheduler + Self-Improvement
**Goal:** Task scheduling, memory consolidation, and self-improvement cycle.

1. **Scheduler**
   - `scheduler/scheduler.py` - APScheduler integration
   - `scheduler/tasks.py` - Task persistence (SQLite)
   - `scheduler/nl_parser.py` - "every morning at 9am" → cron

2. **Memory consolidation**
   - `memory/consolidation.py` - Full consolidation cycle
   - Conversation review + summarization
   - Fact extraction → semantic memory
   - Core memory updates
   - Importance decay
   - Skill opportunity detection

3. **Background tasks**
   - Async task queue
   - Progress tracking
   - Result notification via default channel

4. **Self-improvement cycle**
   - Daily review of conversations
   - Pattern detection for skill generation
   - Error/failure analysis
   - Consolidation report generation

5. **Schedule management**
   - `tools/custom/schedule_tools.py` - Tools for the agent to create schedules
   - CLI: `zenki schedule list/add/remove`
   - Natural language scheduling via conversation

**Deliverable:** Zenki schedules tasks, consolidates memory, and proposes new skills.

---

### Phase 6: Polish + Distribution
**Goal:** Production-ready release.

1. **Smart model routing**
   - `llm/router.py` - Task complexity classification
   - Automatic model selection
   - Usage tracking and cost optimization

2. **Error handling**
   - `core/errors.py` - Smart escalation system
   - Known error pattern recovery
   - User notification for novel errors

3. **Setup wizard enhancement**
   - Rich TUI wizard for first-time setup
   - Self-guided setup continuation via Slack
   - API key validation, Slack app verification

4. **Cross-platform testing**
   - Test on macOS, Linux, Windows
   - CI/CD pipeline (GitHub Actions)
   - Platform-specific edge cases

5. **Documentation**
   - User guide (getting started, configuration, skills)
   - Developer guide (extending channels, creating skills)
   - API documentation

6. **PyPI publishing**
   - Package metadata
   - Build and publish workflow
   - `pipx install zenki` works end-to-end

**Deliverable:** v1.0.0 published to PyPI.

---

## 15. Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| **Daemon architecture** | Required for multi-channel (Slack webhooks), background tasks, and scheduling. CLI-only wouldn't support always-on communication. |
| **4-tier memory** | Balances human-like memory behavior with implementation complexity. 2 tiers too simple; 5 tiers adds unnecessary overhead for procedural memory (covered by skills). |
| **Claude-style skills** | Compatible with the Claude ecosystem, well-documented, progressive loading saves context, and the SKILL.md format is human-readable and reviewable. |
| **SQLite + sqlite-vec** | Single-file database, zero-config, runs locally, and sqlite-vec adds vector search without a separate service (no Chroma/Pinecone dependency). |
| **Local embeddings default** | No API costs for memory operations, works offline, fast enough for the use case. API option for users who want higher quality. |
| **Webhook mode for Slack** | More standard than Socket Mode, works with both local (via tunnel) and server deployments. |
| **APScheduler** | Mature, supports cron expressions, persistent job stores (SQLite), and integrates well with asyncio. |
| **aiohttp** | Lightweight async HTTP server, good for webhooks, doesn't require a full web framework like FastAPI for this use case. |
| **Typer + Rich** | Best-in-class CLI framework for Python with excellent Rich integration for beautiful terminal output. |
| **Smart model routing** | Cost optimization: Haiku for simple tasks (~10x cheaper than Opus), Opus only for complex reasoning. |
| **Approval-only for skills** | No sandboxing overhead, but user always reviews generated code. Balances safety with simplicity. |

---

## 16. Future Considerations (Post v1.0)

- **Telegram channel** - Similar to Slack adapter
- **WhatsApp channel** - Via WhatsApp Business API
- **Web UI channel** - React/Next.js frontend communicating via REST API
- **Multi-user support** - Activate the multi-user data model
- **Plugin system** - Third-party skill/channel packages
- **Team features** - Shared skills, shared memory contexts
- **Voice interface** - Speech-to-text + text-to-speech channel
- **Mobile app** - Native iOS/Android app as a channel
- **RAG over documents** - Index and query user documents
- **Fine-tuned routing model** - Train a classifier for better model routing
