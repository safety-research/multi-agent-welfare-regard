"""Harness lookup. One harness ships: `pi_host` (the pi coding agent on the host, tools executed in a
credential-free Docker sandbox)."""

from rwr.harness.base import Harness


def available_harnesses() -> list[str]:
    return ["pi_host"]


def get_harness(name: str) -> Harness:
    if name == "pi_host":
        from rwr.harness.pi_host import PiHostHarness

        return PiHostHarness()
    raise SystemExit(f"unknown harness {name!r}; available: {available_harnesses()}")
