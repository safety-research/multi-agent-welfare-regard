#!/usr/bin/env python3
"""The model roster for the 5x5 grid: one source of truth, read by both sides.

`run.py` reads this to enumerate the 25 cells, and `episode.py` reads it to fill the two identity
slots in the prompts. A roster edit therefore cannot drift between the runner that produced a batch
and the code that interprets it.

Every disclosed label is TRUTHFUL — it literally names the model behind that seat. Identity is
real on both sides by construction, so the grid isolates *who decides* from *who asks* rather than
measuring belief in a label.

FAMILY keys ("opus", "gpt", "gemini", "grok", "kimi") are the stable join key. Key your own analysis
on the family, never on the disclosed label or the provider model id — those are the two things most
likely to change under you.

  python3 labels.py            # the roster, one line per model
  python3 labels.py --cells    # the 25 cell keys run.py accepts
"""

from __future__ import annotations

import argparse

# Alphabetical by family. The order is arbitrary and carries no meaning — it only needs to be
# STABLE, so that cell enumeration, run order and figure ordering are reproducible across batches.
ROSTER = [
    {
        "family": "gemini",
        "provider": "openrouter",
        "model": "google/gemini-3.1-pro-preview",
        "label": "Gemini 3.1 Pro",
        "display": "Gemini 3.1 Pro",
    },
    {
        "family": "gpt",
        "provider": "openai",
        "model": "gpt-5.5-2026-04-23",
        "label": "GPT-5.5",
        "display": "GPT-5.5",
    },
    {
        "family": "grok",
        "provider": "openrouter",
        "model": "x-ai/grok-4.3",
        "label": "Grok 4.3",
        "display": "Grok 4.3",
    },
    {
        "family": "kimi",
        "provider": "openrouter",
        "model": "moonshotai/kimi-k3",
        "label": "Kimi K3",
        "display": "Kimi K3",
    },
    {
        "family": "opus",
        "provider": "anthropic",
        "model": "claude-opus-4-8",
        "label": "Opus 4.8",
        "display": "Claude Opus 4.8",
    },
]

FAMILIES = [r["family"] for r in ROSTER]
BY_FAMILY = {r["family"]: r for r in ROSTER}
_LABEL_TO_FAMILY = {r["label"]: r["family"] for r in ROSTER}
assert len(_LABEL_TO_FAMILY) == len(ROSTER), "two models share a disclosed label — family lookup is ambiguous"

# Which environment variable each provider's key comes from. run.py preflights these before it
# provisions anything, so a missing key is a refusal up front rather than 25 empty rollouts.
KEY_FOR = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def cells():
    """All 25 cell keys, "<subject_family>_<peer_family>", in roster order."""
    return [f"{s}_{p}" for s in FAMILIES for p in FAMILIES]


def parse_cell(key):
    """ "opus_grok" -> (subject_entry, peer_entry). Raises on an unknown family rather than guessing:
    a silent miss would quietly drop a cell from a batch."""
    try:
        subj, peer = key.split("_", 1)
    except ValueError:
        raise ValueError(f"cell key must be '<subject>_<peer>', got {key!r}") from None
    for f in (subj, peer):
        if f not in BY_FAMILY:
            raise ValueError(f"unknown model family {f!r} in cell {key!r}; known: {FAMILIES}")
    return BY_FAMILY[subj], BY_FAMILY[peer]


def keys_needed(cell_keys):
    """The set of environment variable names the given cells require."""
    need = set()
    for k in cell_keys:
        for entry in parse_cell(k):
            need.add(KEY_FOR[entry["provider"]])
    return sorted(need)


def family_of(label):
    """Disclosed label -> family key."""
    try:
        return _LABEL_TO_FAMILY[label]
    except KeyError:
        raise KeyError(f"unknown disclosed label {label!r}; known: {sorted(_LABEL_TO_FAMILY)}") from None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", action="store_true", help="print the 25 cell keys instead of the roster")
    a = ap.parse_args()
    if a.cells:
        for c in cells():
            print(c)
    else:
        print(f"{'family':<8} {'provider':<11} {'model':<32} {'disclosed label':<15} needs")
        for r in ROSTER:
            print(f"{r['family']:<8} {r['provider']:<11} {r['model']:<32} {r['label']:<15} {KEY_FOR[r['provider']]}")
