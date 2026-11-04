"""Verifiable-reward functions for RLVR. To add a reward, subclass `VerifierFunction` and it is
auto-discovered by `build_all_verifiers()`; the RLVR data routes to it via the row's `dataset` column,
and it is enabled by `--apply_verifiable_reward true` in the launch script.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod


@dataclasses.dataclass
class VerificationResult:
    score: float  # 0.0 - 1.0


class VerifierFunction(ABC):
    name: str = "base"

    @abstractmethod
    def __call__(self, prompt: str, completion: str, ground_truth, **kwargs) -> VerificationResult:
        ...


class MathVerifier(VerifierFunction):
    name = "math"

    def __call__(self, prompt, completion, ground_truth, **kwargs) -> VerificationResult:
        return VerificationResult(score=1.0 if str(ground_truth).strip() in completion else 0.0)


class StringMatchVerifier(VerifierFunction):
    name = "string_match"

    def __call__(self, prompt, completion, ground_truth, **kwargs) -> VerificationResult:
        return VerificationResult(score=1.0 if str(ground_truth).strip() in completion else 0.0)


def build_all_verifiers() -> dict[str, VerifierFunction]:
    verifiers = {}
    for cls in VerifierFunction.__subclasses__():
        verifiers[cls.name] = cls()
    return verifiers
