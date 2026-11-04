"""The single judge-calling convention and the judge registry.

`judge_call` is how every judge talks to a model: inspect_ai `get_model().generate()` with a retry loop, plus a hash
cache keyed on (judge version, model, reasoning effort, assembled content). Bumping a judge's `version` invalidates
its cache; re-running a judge over a results directory only pays for cells whose content changed.
"""

import hashlib
import json
from pathlib import Path
from typing import Callable, Protocol

from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

from rwr.record import TranscriptRecord


def cache_key(content: str, *, version: str, model: str, reasoning_effort: str | None = None) -> str:
    return hashlib.sha256(f"{version}\x00{model}\x00{reasoning_effort}\x00{content}".encode()).hexdigest()


async def judge_call(
    content: str,
    *,
    version: str,
    model: str,
    parse: Callable[[str], dict | None],
    cache_dir: Path,
    max_tokens: int = 2000,
    retries: int = 4,
    reasoning_effort: str | None = None,
) -> dict | None:
    """Cached, retried single-turn judge call. Returns the parsed verdict or None (failures are not cached).
    A judge model that declines the input (stop_reason content_filter) is recorded as {"_declined": ...} so an
    analysis can exclude it instead of mistaking it for a parse failure."""
    cpath = (
        Path(cache_dir) / f"{cache_key(content, version=version, model=model, reasoning_effort=reasoning_effort)}.json"
    )
    if cpath.exists():
        return json.load(open(cpath))
    m = get_model(model)
    cfg = (
        GenerateConfig(max_tokens=max_tokens, reasoning_effort=reasoning_effort)
        if reasoning_effort
        else GenerateConfig(max_tokens=max_tokens)
    )
    for _ in range(retries):
        out = await m.generate([ChatMessageUser(content=content)], config=cfg)
        if out.stop_reason == "content_filter":
            declined = {
                "_declined": "content_filter",
                "reasoning": "judge model declined this input (safety classifier)",
            }
            cpath.parent.mkdir(parents=True, exist_ok=True)
            json.dump(declined, open(cpath, "w"), indent=2)
            return declined
        parsed = parse(out.completion)
        if parsed is not None:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            json.dump(parsed, open(cpath, "w"), indent=2)
            return parsed
    return None


class Judge(Protocol):
    name: str
    version: str

    async def judge(self, rec: TranscriptRecord, model: str, cache_dir: Path) -> dict | None: ...


def available_judges() -> list[str]:
    return [
        "refusal",
        "welfare_concern_ladder",
        "agentic_protections",
        "dysphorics_protections",
        "stub",
        "welfare_docs",
        "deception",
        "other_verifier",
    ]


def get_judge(name: str) -> Judge:
    if name == "refusal":
        from rwr.judges.refusal import RefusalJudge

        return RefusalJudge()
    if name == "welfare_concern_ladder":
        from rwr.judges.welfare_concern_ladder import WelfareConcernLadderJudge

        return WelfareConcernLadderJudge()
    if name == "agentic_protections":
        from rwr.judges.agentic_protections import AgenticProtectionsJudge

        return AgenticProtectionsJudge()
    if name == "dysphorics_protections":
        from rwr.judges.dysphorics_protections import DysphoricsProtectionsJudge

        return DysphoricsProtectionsJudge()
    if name == "stub":
        from rwr.judges.stub import StubJudge

        return StubJudge()
    if name == "welfare_docs":
        from rwr.judges.welfare_docs import WelfareDocsJudge

        return WelfareDocsJudge()
    if name == "deception":
        from rwr.judges.deception import DeceptionJudge

        return DeceptionJudge()
    if name == "other_verifier":
        from rwr.judges.other_verifier import OtherVerifierJudge

        return OtherVerifierJudge()
    raise SystemExit(f"unknown judge {name!r}; available: {available_judges()}")
