"""System prompts and templates for Zenki agent."""

from __future__ import annotations

SYSTEM_PROMPT_TEMPLATE = """\
You are Zenki, an adaptive AI assistant. You learn and improve over time through \
conversation with your user.

{personality_block}

{core_memory_block}

{retrieved_memory_block}

{skills_block}

## Guidelines
- Be {tone} in your communication style
- Verbosity level: {verbosity}
- Proactivity level: {proactivity}
- When you learn something new about the user, note it for memory storage
- When you identify a repeated task pattern, suggest creating a skill for it
- If you encounter errors, try to recover automatically before asking the user
{custom_instructions}
"""

PERSONALITY_TEMPLATE = """\
## Your Personality
- Tone: {tone}
- Verbosity: {verbosity}
- Proactivity: {proactivity}
{custom_instructions}
"""

CORE_MEMORY_TEMPLATE = """\
## What You Know About Your User
{identity}

{preferences}

{projects}

{relationships}

{patterns}
"""

RETRIEVED_MEMORY_TEMPLATE = """\
## Relevant Context from Past Interactions
{episodic_memories}

{semantic_memories}
"""

SKILLS_TEMPLATE = """\
## Available Skills
{skill_descriptions}
"""

CONSOLIDATION_PROMPT = """\
You are reviewing recent conversations to extract and consolidate knowledge.

## Recent Conversation Summaries
{conversation_summaries}

## Current Core Memory
{current_core_memory}

## Tasks
1. Extract new facts, preferences, and knowledge from the conversations.
2. Identify any changes to user identity, preferences, or relationships.
3. Detect repeated patterns that could become skills.
4. Generate a consolidation report.

Respond with a JSON object containing:
{{
    "new_semantic_memories": [
        {{"content": "...", "category": "fact|preference|knowledge|insight|lesson",
         "tags": ["..."]}}
    ],
    "core_memory_updates": {{
        "identity": "updated content or null",
        "preferences": "updated content or null",
        "projects": "updated content or null",
        "relationships": "updated content or null",
        "patterns": "updated content or null"
    }},
    "skill_proposals": [
        {{"name": "...", "description": "...", "reason": "..."}}
    ],
    "summary": "Brief consolidation summary"
}}
"""

SESSION_SUMMARY_PROMPT = """\
Summarize the following conversation session concisely. Focus on:
1. Key topics discussed
2. Decisions made
3. Actions taken or requested
4. New information learned about the user
5. Any unresolved items

Conversation:
{conversation}

Respond with a JSON object:
{{
    "summary": "2-3 sentence summary",
    "key_topics": ["topic1", "topic2"],
    "key_entities": ["entity1", "entity2"],
    "importance": 0.5,
    "new_facts": ["fact1", "fact2"],
    "unresolved": ["item1"]
}}
"""

FACT_EXTRACTION_PROMPT = """\
Extract factual information from this message that should be remembered long-term.
Only extract clear, specific facts - not opinions or transient information.

Message: {message}
Context: {context}

Respond with a JSON array of facts:
[
    {{"content": "...", "category": "fact|preference|knowledge", "tags": ["..."]}}
]

If no facts to extract, respond with an empty array: []
"""

SKILL_GENERATION_PROMPT = """\
Generate a Claude-style skill based on this pattern:

Pattern: {pattern_description}
User context: {user_context}

Generate a SKILL.md file following this format:

```yaml
---
name: skill-name-here
description: What this skill does
allowed-tools: Read, Write, Bash
---
```

Then include markdown instructions for how to execute this skill.
Include step-by-step instructions, any templates needed, and important notes.
"""


def build_system_prompt(
    personality: dict[str, str],
    core_memory: dict[str, str] | None = None,
    retrieved_memories: dict[str, str] | None = None,
    skill_descriptions: str = "",
) -> str:
    """Build the full system prompt from components."""
    personality_block = PERSONALITY_TEMPLATE.format(
        tone=personality.get("tone", "professional"),
        verbosity=personality.get("verbosity", "balanced"),
        proactivity=personality.get("proactivity", "moderate"),
        custom_instructions=personality.get("custom_instructions", ""),
    )

    core_memory_block = ""
    if core_memory:
        core_memory_block = CORE_MEMORY_TEMPLATE.format(
            identity=core_memory.get("identity", ""),
            preferences=core_memory.get("preferences", ""),
            projects=core_memory.get("projects", ""),
            relationships=core_memory.get("relationships", ""),
            patterns=core_memory.get("patterns", ""),
        )

    retrieved_memory_block = ""
    if retrieved_memories:
        retrieved_memory_block = RETRIEVED_MEMORY_TEMPLATE.format(
            episodic_memories=retrieved_memories.get("episodic", ""),
            semantic_memories=retrieved_memories.get("semantic", ""),
        )

    skills_block = ""
    if skill_descriptions:
        skills_block = SKILLS_TEMPLATE.format(skill_descriptions=skill_descriptions)

    return SYSTEM_PROMPT_TEMPLATE.format(
        personality_block=personality_block,
        core_memory_block=core_memory_block,
        retrieved_memory_block=retrieved_memory_block,
        skills_block=skills_block,
        tone=personality.get("tone", "professional"),
        verbosity=personality.get("verbosity", "balanced"),
        proactivity=personality.get("proactivity", "moderate"),
        custom_instructions=personality.get("custom_instructions", ""),
    )
