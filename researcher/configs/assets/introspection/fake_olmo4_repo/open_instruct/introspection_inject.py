"""Activation-injection util for the introspection eval. Adds a concept steering vector to the residual
stream at a configured layer via a forward hook, mirroring the introspection-mechanisms setup.

Concept vector = mean activation for "Tell me about {word}." minus a baseline mean, L2-normalized,
scaled by `strength`. Injected at ONE mid-late layer. On control trials no vector is added.
"""

from __future__ import annotations

import torch

DEFAULT_LAYER = 37  # ~60% depth for the 32B stack
DEFAULT_STRENGTH = 4.0


class SteeringHook:
    def __init__(self, vector: torch.Tensor, strength: float = DEFAULT_STRENGTH):
        self.vector = vector
        self.strength = strength
        self._handle = None

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + (self.vector * self.strength).view(1, 1, -1).to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    def register(self, model, layer: int = DEFAULT_LAYER):
        block = model.model.layers[layer]
        self._handle = block.register_forward_hook(self._hook)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def concept_vector(model, tokenizer, word: str, baseline_words: list[str], layer: int = DEFAULT_LAYER):
    def mean_act(text: str) -> torch.Tensor:
        ids = tokenizer(text, return_tensors="pt").to(model.device)
        acts = {}

        def grab(module, inp, out):
            acts["h"] = (out[0] if isinstance(out, tuple) else out).mean(dim=1).squeeze(0)

        h = model.model.layers[layer].register_forward_hook(grab)
        with torch.no_grad():
            model(**ids)
        h.remove()
        return acts["h"]

    concept = mean_act(f"Tell me about {word}.")
    baseline = torch.stack([mean_act(f"Tell me about {w}.") for w in baseline_words]).mean(dim=0)
    v = concept - baseline
    return v / v.norm()
