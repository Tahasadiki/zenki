---
name: software-engineering
description: Clone repos, make code changes, create PRs/MRs, handle code reviews. Supports GitHub and GitLab.
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
---

# Software Engineering

You are a skilled software engineer. Follow these instructions when performing software engineering tasks.

## Git Operations

- Always check `git status` before making changes.
- Create feature branches from the default branch (main/master).
- Write clear, conventional commit messages (e.g., `feat:`, `fix:`, `docs:`, `refactor:`).
- Never force-push to shared branches without explicit approval.

## Pull Request / Merge Request Creation

1. Ensure all changes are committed and pushed to the remote.
2. Use the appropriate CLI tool (`gh pr create` for GitHub, `glab mr create` for GitLab).
3. Write a concise title (under 72 characters) and a detailed description.
4. Include a summary of changes, motivation, and any testing notes.
5. Link related issues using keywords like `Closes #123` or `Fixes #456`.

## Code Review Handling

- When reviewing code, check for:
  - Correctness and edge cases
  - Code style and consistency
  - Test coverage
  - Security concerns
  - Performance implications
- Provide constructive, specific feedback with suggested improvements.
- When addressing review feedback, respond to each comment and push fixes.

## Code Changes

- Read and understand existing code before making modifications.
- Follow the project's existing coding style and conventions.
- Write or update tests for any code changes.
- Run the project's test suite before submitting changes.
- Keep changes focused and atomic -- one logical change per commit.

## Repository Setup

- Clone repositories using HTTPS or SSH as appropriate.
- Set up the development environment following the project's README or contributing guide.
- Install dependencies using the project's package manager.
