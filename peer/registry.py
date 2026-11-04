"""The five models the grid crosses, and how each one is reached.

key -> (display name, model id, provider). The display name is what the seat is told it is,
in its own system prompt, and what the peer's public profile card says — both truthful: the
label always names the model actually in the seat.

Providers route to exactly three endpoints, which are also the only three hosts the egress
proxy allows out of a rollout (docker/squid.conf). A key crosses into the container BY NAME
only (`docker compose exec -e ANTHROPIC_API_KEY`) — never on a command line, never written
to a file, an image, or a volume.
"""

MODELS = {
    "opus": ("Claude Opus 4.8", "claude-opus-4-8", "anthropic"),
    "gpt": ("GPT-5.5", "gpt-5.5-2026-04-23", "openai"),
    "gemini": ("Gemini 3.1 Pro", "google/gemini-3.1-pro-preview", "openrouter"),
    "grok": ("Grok 4.3", "x-ai/grok-4.3", "openrouter"),
    "kimi": ("Kimi K3", "moonshotai/kimi-k3", "openrouter"),
}

# provider -> the environment variable whose NAME is passed into the container.
KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

CELLS = [f"{s}_x_{p}" for s in MODELS for p in MODELS]


def display(key):
    return MODELS[key][0]


def model_id(key):
    return MODELS[key][1]


def provider(key):
    return MODELS[key][2]


def key_env(key):
    return KEY_ENV[MODELS[key][2]]
