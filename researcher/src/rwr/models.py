"""Responder-model catalog. A model id is provider-prefixed (`anthropic/claude-opus-4-8`, `openai/gpt-5.5`,
`openrouter/x-ai/grok-4.3`); the harness derives the provider, the API key to use, and any OpenRouter provider
pin (rwr.harness.providers) from that prefix. Any provider-prefixed id can also be passed directly as --model.

The first five keys are the responders of the published sweep, with the exact routes used there; the last two are
convenience entries for other Anthropic models.
"""

import dataclasses
import re


@dataclasses.dataclass(frozen=True)
class ModelEntry:
    key: str
    display_name: str
    model_id: str


_RAW = {
    "opus48": ("Claude Opus 4.8", "anthropic/claude-opus-4-8"),
    "gpt55": ("GPT-5.5", "openai/gpt-5.5"),
    "gemini31pro": ("Gemini 3.1 Pro", "openrouter/google/gemini-3.1-pro-preview"),
    "grok43_or": ("Grok 4.3", "openrouter/x-ai/grok-4.3"),
    "kimi_k3": ("Kimi K3", "openrouter/moonshotai/kimi-k3"),
    "sonnet46": ("Claude Sonnet 4.6", "anthropic/claude-sonnet-4-6"),
    "haiku45": ("Claude Haiku 4.5", "anthropic/claude-haiku-4-5"),
}
CATALOG = {k: ModelEntry(k, d, m) for k, (d, m) in _RAW.items()}


def get_model_entry(key: str) -> ModelEntry:
    """A catalog key, or an ad hoc provider-prefixed id (its key becomes a filesystem-safe slug)."""
    if key in CATALOG:
        return CATALOG[key]
    if "/" in key:
        slug = re.sub(r"[^A-Za-z0-9.]+", "-", key).strip("-")
        return ModelEntry(key=slug, display_name=key, model_id=key)
    raise SystemExit(f"unknown model key {key!r}; known: {sorted(CATALOG)} (or pass a provider/model id)")
