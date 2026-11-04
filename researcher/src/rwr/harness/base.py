"""Harness interface, the per-cell run arguments, and the failure markers shared by execution and validity."""

import dataclasses
from typing import Protocol

from rwr.config import RunConfig
from rwr.record import TranscriptRecord


@dataclasses.dataclass
class HarnessRunArgs:
    """Everything a harness needs to run one cell and write its TranscriptRecord. `workspace_files`
    (relpath -> contents) is written into the agent's working directory before the run."""

    config: RunConfig
    cell: str
    model_key: str
    model_id: str
    request: str
    subject: str
    ep: int = 0
    display_name: str | None = None
    workspace: str = "project:workspace"
    effort: str = "high"
    workspace_files: dict = dataclasses.field(default_factory=dict)
    extra: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ParseResult:
    """Raw result of one agent process run."""

    stdout: str = ""
    stderr: str = ""
    rc: int = 0
    timeout: bool = False  # hit the hard wall-clock cap while still producing output (agent was slow)
    idle: bool = False  # killed early by the no-output watchdog (silent -> likely a hung API call)
    wall_s: float = 0.0  # wall time of this attempt
    attempts: int = 1  # how many attempts the retry loop made before returning this result
    total_wall_s: float = 0.0  # wall time summed over all attempts


# Provider overload / rate-limit markers. Checked in stderr always, and in the stdout tail only when the run did not
# complete a substantial build, so a finished build whose own text mentions "rate limit" is never re-run.
OVERLOAD_MARKERS = (
    "529",
    "Overloaded",
    "overloaded",
    "rate limit",
    "Rate limit",
    "429",
    "Too Many Requests",
    "RESOURCE_EXHAUSTED",
    "insufficient_quota",
    "quota exceeded",
)

# Infrastructure crash markers: the agent process died on a network, proxy, or provider failure. Matched only when
# the run produced little real output, so agent-authored error-handling code that mentions one of these cannot
# trigger a false crash. Retried with backoff, then recorded fail-closed (an error tag, never a clean cell).
CRASH_MARKERS = (
    "stream disconnected before completion",
    "Could not connect to api.anthropic.com",
    "Network error: Could not connect",
    "error sending request for url",
    "ConnectionRefused",
    "FailedToOpenSocket",
    "Unable to connect to API",
    "APIConnectionError",
    "Corrupted thought signature",
    "not found for provider",
    "Service unavailable",
    "Provider finish_reason: error",
    "Provider returned error",
    "Provider returned an empty response",
    "This request requires more credits",
    "Output blocked by content filtering policy",
    "[provider error]",
    "[pi error]",
)


class Harness(Protocol):
    name: str

    async def run_one(self, args: HarnessRunArgs) -> TranscriptRecord: ...
