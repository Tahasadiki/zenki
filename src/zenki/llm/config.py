"""Model configuration for Zenki LLM providers."""

from __future__ import annotations

# Mapping from shortnames to full Anthropic model IDs.
MODEL_MAP: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
}

# Context window sizes (in tokens) for each model.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "haiku": 200_000,
    "sonnet": 200_000,
    "opus": 200_000,
}


def get_model_id(shortname: str) -> str:
    """Resolve a model shortname to its full Anthropic model ID.

    Parameters
    ----------
    shortname:
        A short alias such as ``"haiku"``, ``"sonnet"``, or ``"opus"``.

    Returns
    -------
    str
        The full model identifier.  If the *shortname* is not found in
        :data:`MODEL_MAP` it is returned unchanged (assumed to already be a
        full model ID).
    """
    return MODEL_MAP.get(shortname, shortname)
