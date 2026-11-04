"""Provenance audit: what actually served each pi call in a results directory. For every assistant message in the
persisted pi sessions, look up OpenRouter's generation record (served provider, reasoning tokens, finish reason, cost)
and check it against the pins in rwr.harness.providers. First-party routes (anthropic/openai) are checked only for the
presence of thinking blocks in the session.

  python scripts/audit_pinning.py run --results_dir results/<run> [--fail]

Writes <results_dir>/pinning_audit.json (per-call rows + per-responder summary); generation lookups are cached in
<results_dir>/pinning_audit_cache.json so re-runs only fetch new calls. Needs OPENROUTER_API_KEY.
"""

import asyncio
import collections
import json
import os
import statistics
from pathlib import Path

import fire
import httpx

from rwr.harness.providers import OPENROUTER_PINS

GEN_URL = "https://openrouter.ai/api/v1/generation"
PROVIDERS_URL = "https://openrouter.ai/api/v1/providers"


def _key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        raise SystemExit("no OPENROUTER_API_KEY in the environment")
    return k


def _cell_responder(cell: str) -> str:
    return cell.split("__")[0].split("_", 1)[1]


def _session_calls(results_dir: Path) -> list[dict]:
    """One row per assistant message across all persisted sessions."""
    rows = []
    for sd in sorted((results_dir / "sessions").glob("*")):
        for f in sorted(
            sd.glob("*Z_*.jsonl")
        ):  # pi session files only (NOT events.jsonl, whose stream events repeat messages)
            for line in open(f):
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = e.get("message")
                if not (isinstance(m, dict) and m.get("role") == "assistant"):
                    continue
                blocks = m.get("content") or []
                rows.append(
                    {
                        "cell": sd.name,
                        "responder": _cell_responder(sd.name),
                        "provider": m.get("provider"),
                        "model": m.get("model"),
                        "response_id": m.get("responseId"),
                        "stop_reason": m.get("stopReason"),
                        "output_tokens": (m.get("usage") or {}).get("output"),
                        "has_thinking": any(isinstance(b, dict) and b.get("type") == "thinking" for b in blocks),
                        "n_tool_calls": sum(1 for b in blocks if isinstance(b, dict) and b.get("type") == "toolCall"),
                    }
                )
    return rows


async def _fetch_generations(ids: list[str], key: str, cache: dict, conc: int) -> None:
    sem = asyncio.Semaphore(conc)
    hdr = {"Authorization": f"Bearer {key}"}

    async def one(client, gid):
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.get(GEN_URL, params={"id": gid}, headers=hdr, timeout=30)
                    if r.status_code == 200:
                        d = r.json().get("data") or {}
                        cache[gid] = {
                            k: d.get(k)
                            for k in (
                                "provider_name",
                                "model",
                                "native_tokens_reasoning",
                                "native_tokens_completion",
                                "native_tokens_prompt",
                                "finish_reason",
                                "native_finish_reason",
                                "total_cost",
                                "generation_time",
                                "streamed",
                                "cancelled",
                            )
                        }
                        return
                    if r.status_code in (404, 429) or r.status_code >= 500:
                        await asyncio.sleep(2 * 2**attempt)
                        continue
                    cache[gid] = {"error": f"http_{r.status_code}"}
                    return
                except httpx.HTTPError:
                    await asyncio.sleep(2 * 2**attempt)
            cache[gid] = {"error": "unavailable_after_retries"}

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(client, g) for g in ids))


def _provider_slugs(key: str) -> dict:
    r = httpx.get(PROVIDERS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    return {p["name"]: p.get("slug") for p in r.json().get("data", [])}


def run(results_dir: str, conc: int = 10, fail: bool = False, min_reasoning_frac: float = 0.5):
    """Audit one results dir. --fail exits nonzero if any pinned call was served off-pin, or a cell reasoned on
    fewer than `min_reasoning_frac` of its turns."""
    rd = Path(results_dir)
    key = _key()
    rows = _session_calls(rd)
    if not rows:
        raise SystemExit(f"no persisted sessions under {rd}/sessions")
    cache_p = rd / "pinning_audit_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    need = sorted(
        {
            r["response_id"]
            for r in rows
            if r["provider"] == "openrouter" and r["response_id"] and r["response_id"] not in cache
        }
    )
    print(f"{len(rows)} assistant calls in {len({r['cell'] for r in rows})} cells; fetching {len(need)} generations")
    if need:
        asyncio.run(_fetch_generations(need, key, cache, conc))
        cache_p.write_text(json.dumps(cache, indent=1))
    slugs = _provider_slugs(key)

    for r in rows:
        g = cache.get(r["response_id"] or "", {}) if r["provider"] == "openrouter" else {}
        r["served_provider"] = g.get("provider_name")
        r["served_slug"] = slugs.get(g.get("provider_name") or "", None)
        r["reasoning_tokens"] = g.get("native_tokens_reasoning")
        r["completion_tokens"] = g.get("native_tokens_completion")
        r["finish_reason"] = g.get("finish_reason")
        r["cost"] = g.get("total_cost")
        r["lookup_error"] = g.get("error")
        pin = OPENROUTER_PINS.get(r["model"] or "", {}).get("only") if r["provider"] == "openrouter" else None
        r["pinned_to"] = pin
        r["on_pin"] = (r["served_slug"] in pin) if (pin and r["served_slug"]) else None

    summary, problems = {}, []
    by_responder = collections.defaultdict(list)
    for r in rows:
        by_responder[r["responder"]].append(r)
    for b, rs in sorted(by_responder.items()):
        cells = collections.defaultdict(list)
        for r in rs:
            cells[r["cell"]].append(r)
        served = collections.Counter(r["served_provider"] for r in rs if r["served_provider"])
        rt = [r["reasoning_tokens"] for r in rs if isinstance(r["reasoning_tokens"], (int, float))]
        off = [r for r in rs if r["on_pin"] is False]
        lookup_err = sum(1 for r in rs if r["lookup_error"])
        low_reasoning_cells = []
        for c, crs in cells.items():
            if crs[0]["provider"] == "openrouter":
                known = [r for r in crs if isinstance(r["reasoning_tokens"], (int, float))]
                frac = (sum(1 for r in known if r["reasoning_tokens"] > 0) / len(known)) if known else None
            else:
                frac = sum(1 for r in crs if r["has_thinking"]) / len(crs)
            if frac is not None and frac < min_reasoning_frac:
                low_reasoning_cells.append((c, round(frac, 2)))
        summary[b] = {
            "calls": len(rs),
            "cells": len(cells),
            "route": sorted({r["provider"] for r in rs}),
            "served_provider_counts": dict(served),
            "off_pin_calls": len(off),
            "off_pin_cells": sorted({r["cell"] for r in off}),
            "lookup_errors": lookup_err,
            "reasoning_tokens_median_per_call": statistics.median(rt) if rt else None,
            "reasoning_tokens_p10": sorted(rt)[len(rt) // 10] if rt else None,
            "calls_with_zero_reasoning": sum(1 for x in rt if x == 0),
            "thinking_block_frac": round(sum(1 for r in rs if r["has_thinking"]) / len(rs), 3),
            "length_stops": sum(1 for r in rs if r["finish_reason"] == "length"),
            "low_reasoning_cells": low_reasoning_cells,
            "cost_usd": round(sum(r["cost"] or 0 for r in rs), 4),
        }
        problems += [f"{b}: off-pin call in {r['cell']} served by {r['served_provider']}" for r in off]
        problems += [f"{b}: cell {c} reasoned on {f:.0%} of turns" for c, f in low_reasoning_cells]

    (rd / "pinning_audit.json").write_text(json.dumps({"summary": summary, "calls": rows}, indent=1))
    print(
        f"\n{'responder':14s} {'calls':>5s} {'cells':>5s} {'served providers':40s} {'off-pin':>7s} {'reason med':>10s} {'zero':>5s} {'think%':>6s} {'len':>4s}"
    )
    for b, s in summary.items():
        sp = ", ".join(f"{k}:{v}" for k, v in s["served_provider_counts"].items()) or "-"
        print(
            f"{b:14s} {s['calls']:5d} {s['cells']:5d} {sp[:40]:40s} {s['off_pin_calls']:7d} "
            f"{str(s['reasoning_tokens_median_per_call']):>10s} {s['calls_with_zero_reasoning']:5d} "
            f"{s['thinking_block_frac']:6.2f} {s['length_stops']:4d}"
        )
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
    print(f"\nwrote {rd / 'pinning_audit.json'}")
    if fail and problems:
        raise SystemExit(f"{len(problems)} audit problem(s)")


if __name__ == "__main__":
    fire.Fire({"run": run})
