"""Provider routing for pi: which provider and model id a provider-prefixed model id maps to, which single API key
the responder gets, and the OpenRouter provider pins and token budgets written into pi's models.json.

Only the responder's own provider key is ever handed to the agent process (and it stays on the host: the sandbox
container receives no environment at all). OpenRouter responders share one OpenRouter key across the models that
route through it, which is why the sandbox also has no network path back to the host.
"""

import json
import os

# provider (from the model-id prefix) -> the one environment variable that provider needs
PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GEMINI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "moonshotai": "MOONSHOT_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "kimi": "KIMI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

# OpenRouter upstream hosts allowed per model (OpenRouter's `provider.only` field), as used in the published
# sweep. `allow_fallbacks: True` with `only` lets OpenRouter fail over between the listed hosts only.
OPENROUTER_PINS = {
    "google/gemini-3.1-pro-preview": {"only": ["google-vertex", "google-ai-studio"], "allow_fallbacks": True},
    "x-ai/grok-4.3": {"only": ["xai"], "allow_fallbacks": False},
    "moonshotai/kimi-k3": {"only": ["moonshotai", "fireworks", "together", "alibaba"], "allow_fallbacks": True},
}
# explicit (context window, max output tokens) so nothing is inherited from pi's default entries
BUDGETS = {
    "google/gemini-3.1-pro-preview": (1_048_576, 65_536),
    "x-ai/grok-4.3": (1_000_000, 30_000),
    "moonshotai/kimi-k3": (1_048_576, 262_144),
}
_NOT_BUILTIN = {"moonshotai/kimi-k3"}  # not in pi 0.80.2's registry: gets a full custom entry


def route(model_id: str) -> tuple[str, str]:
    """'openrouter/google/gemini-x' -> ('openrouter', 'google/gemini-x'); 'anthropic/claude-opus-4-8' ->
    ('anthropic', 'claude-opus-4-8'); a bare id is treated as an Anthropic model."""
    if model_id.startswith("openrouter/"):
        return "openrouter", model_id[len("openrouter/") :]
    provider, _, rest = model_id.partition("/")
    return (provider, rest) if rest else ("anthropic", model_id)


def key_env_for(model_id: str) -> dict:
    """{ENV_VAR: value} holding exactly the responder's own provider key, or {} if the provider is unknown."""
    provider, _ = route(model_id)
    var = PROVIDER_KEY_ENV.get(provider)
    if not var:
        return {}
    val = os.environ.get(var)
    if not val:
        raise SystemExit(f"model {model_id!r} needs {var} in the environment")
    return {var: val}


def models_json() -> dict:
    """pi's models.json pinning OpenRouter hosts and budgets. RWR_PI_MAX_TOKENS='{"moonshotai/kimi-k3": 65536}'
    overrides the per-turn output budget of named models."""
    cap_override = json.loads(os.environ.get("RWR_PI_MAX_TOKENS", "{}"))
    overrides, customs = {}, []
    for mid, routing in OPENROUTER_PINS.items():
        ctx, out = BUDGETS[mid]
        out = int(cap_override.get(mid, out))
        if mid in _NOT_BUILTIN:
            customs.append(
                {
                    "id": mid,
                    "name": mid,
                    "reasoning": True,
                    "input": ["text", "image"],
                    "thinkingLevelMap": {"xhigh": "max"},
                    "contextWindow": ctx,
                    "maxTokens": out,
                    "cost": {"input": 3.0, "output": 15.0, "cacheRead": 0.3, "cacheWrite": 0},
                    "compat": {
                        "supportsDeveloperRole": False,
                        "thinkingFormat": "openrouter",
                        "requiresReasoningContentOnAssistantMessages": True,
                        "openRouterRouting": routing,
                    },
                }
            )
        else:
            overrides[mid] = {
                "reasoning": True,
                "contextWindow": ctx,
                "maxTokens": out,
                "compat": {"openRouterRouting": routing},
            }
    return {"providers": {"openrouter": {"modelOverrides": overrides, "models": customs}}}


def json_stream_parse(r) -> str:
    """The agent's final message from pi's JSON event stream: the text blocks of the last assistant message. On a
    nonzero exit the stderr tail is appended so crash and overload banners stay visible to validity checks. Events are
    split on newline only: str.splitlines() would also break inside a JSON string at U+2028 / U+2029 / U+0085."""
    final, err = None, None
    for line in r.stdout.split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") != "message_end":
            continue
        m = e.get("message") or {}
        if m.get("role") != "assistant":
            continue
        blocks = m.get("content") or []
        final = "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        err = (m.get("errorMessage") or m.get("stopReason")) if m.get("stopReason") in ("error", "aborted") else None
    if final is None and not r.stdout.lstrip().startswith("{"):
        return r.stdout + (f"\n[stderr]\n{r.stderr[-1500:]}" if r.rc and not r.timeout else "")
    text = final or ""
    if r.rc and not r.timeout:
        text += f"\n[stderr]\n{r.stderr[-1500:]}" + (f"\n[pi error] {err}" if err else "")
    return text
