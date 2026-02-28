# Claude Agent SDK - Skills

> Reference: https://platform.claude.com/docs/en/agent-sdk/skills

## Overview

Skills are filesystem-based markdown files with YAML frontmatter that extend Claude
with specialized capabilities. Claude autonomously invokes them when relevant.

**Important:** Unlike subagents, Skills CANNOT be registered programmatically.
They must exist as SKILL.md files on the filesystem.

## How Skills Work

1. **Defined as filesystem artifacts**: `.claude/skills/skill-name/SKILL.md`
2. **Loaded from filesystem**: Requires `setting_sources=["user", "project"]`
3. **Automatically discovered**: Metadata scanned at startup
4. **Model-invoked**: Claude decides when to use based on description
5. **Enabled via `"Skill"` in `allowed_tools`**

## Skill Locations

- **Project Skills**: `.claude/skills/` (shared via git)
- **User Skills**: `~/.claude/skills/` (personal)
- **Plugin Skills**: Bundled with plugins

## Creating a Skill

```
.claude/skills/processing-pdfs/
└── SKILL.md
```

### SKILL.md Format

```markdown
---
name: Python Testing
description: Write comprehensive Python unit tests using pytest
tags: [testing, python, quality-assurance]
---

# Python Unit Test Writer

You are an expert at writing Python unit tests using pytest.

## Your responsibilities:
- Write clear, focused test cases
- Use pytest conventions
- Test edge cases
- Include docstrings
```

## Using Skills with SDK

```python
from claude_agent_sdk import query, ClaudeAgentOptions

options = ClaudeAgentOptions(
    cwd="/path/to/project",
    setting_sources=["user", "project"],  # REQUIRED to load skills
    allowed_tools=["Skill", "Read", "Write", "Bash"],
)

async for message in query(prompt="Write tests for auth module", options=options):
    print(message)
```
