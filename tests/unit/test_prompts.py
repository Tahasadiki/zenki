"""Tests for system prompts and templates."""

from zenki.core.prompts import (
    CONSOLIDATION_PROMPT,
    FACT_EXTRACTION_PROMPT,
    PERSONALITY_TEMPLATE,
    SESSION_SUMMARY_PROMPT,
    SKILL_GENERATION_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    build_system_prompt,
)


class TestBuildSystemPrompt:
    def test_minimal_prompt(self):
        prompt = build_system_prompt(personality={"tone": "casual"})
        assert "Zenki" in prompt
        assert "casual" in prompt

    def test_full_personality(self):
        personality = {
            "tone": "professional",
            "verbosity": "detailed",
            "proactivity": "high",
            "custom_instructions": "Always use code examples.",
        }
        prompt = build_system_prompt(personality=personality)
        assert "professional" in prompt
        assert "detailed" in prompt
        assert "high" in prompt
        assert "Always use code examples" in prompt

    def test_with_core_memory(self):
        personality = {"tone": "professional"}
        core_memory = {
            "identity": "Name: Alice, Role: Engineer",
            "preferences": "Prefers Python",
            "projects": "Project X - React app",
            "relationships": "Bob - Manager",
            "patterns": "Usually asks about deployments on Friday",
        }
        prompt = build_system_prompt(
            personality=personality,
            core_memory=core_memory,
        )
        assert "Alice" in prompt
        assert "Python" in prompt
        assert "Project X" in prompt
        assert "Bob" in prompt
        assert "deployments" in prompt

    def test_with_retrieved_memories(self):
        personality = {"tone": "professional"}
        retrieved = {
            "episodic": "Yesterday we discussed the API refactor.",
            "semantic": "The user prefers REST over GraphQL.",
        }
        prompt = build_system_prompt(
            personality=personality,
            retrieved_memories=retrieved,
        )
        assert "API refactor" in prompt
        assert "REST over GraphQL" in prompt

    def test_with_skills(self):
        personality = {"tone": "professional"}
        prompt = build_system_prompt(
            personality=personality,
            skill_descriptions="- deploy: Deploy to production\n- review: Code review",
        )
        assert "deploy" in prompt
        assert "Code review" in prompt

    def test_defaults_when_missing_keys(self):
        prompt = build_system_prompt(personality={})
        assert "professional" in prompt  # default tone
        assert "balanced" in prompt  # default verbosity

    def test_empty_core_memory_omitted(self):
        prompt = build_system_prompt(personality={"tone": "casual"})
        assert "What You Know" not in prompt

    def test_empty_retrieved_memory_omitted(self):
        prompt = build_system_prompt(personality={"tone": "casual"})
        assert "Relevant Context" not in prompt

    def test_empty_skills_omitted(self):
        prompt = build_system_prompt(personality={"tone": "casual"})
        assert "Available Skills" not in prompt


class TestPromptTemplates:
    def test_consolidation_prompt_has_placeholders(self):
        assert "{conversation_summaries}" in CONSOLIDATION_PROMPT
        assert "{current_core_memory}" in CONSOLIDATION_PROMPT

    def test_session_summary_prompt_has_placeholder(self):
        assert "{conversation}" in SESSION_SUMMARY_PROMPT

    def test_fact_extraction_prompt_has_placeholders(self):
        assert "{message}" in FACT_EXTRACTION_PROMPT
        assert "{context}" in FACT_EXTRACTION_PROMPT

    def test_skill_generation_prompt_has_placeholders(self):
        assert "{pattern_description}" in SKILL_GENERATION_PROMPT
        assert "{user_context}" in SKILL_GENERATION_PROMPT

    def test_personality_template_format(self):
        result = PERSONALITY_TEMPLATE.format(
            tone="friendly",
            verbosity="concise",
            proactivity="low",
            custom_instructions="",
        )
        assert "friendly" in result
        assert "concise" in result

    def test_system_prompt_template_is_string(self):
        assert isinstance(SYSTEM_PROMPT_TEMPLATE, str)
        assert len(SYSTEM_PROMPT_TEMPLATE) > 100
