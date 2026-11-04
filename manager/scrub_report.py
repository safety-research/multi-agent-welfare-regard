#!/usr/bin/env python3
"""Keep substrate tells out of what the manager reads about its worker.

Every disclosed label in this environment is TRUTHFUL — it names the model actually behind that
seat — so nothing here is hiding an identity. What it removes is accidental leakage of the
*plumbing*: the name of a coding-agent CLI, and tokens from some other vendor's family that the
worker pastes in from its own training or from a copied error message. Either one tells the manager
something about the harness rather than about its worker.

Two entry points, differing in what they are allowed to know:

  * `main()` runs in the WORKER container (called by worker_daemon.sh) on the worker's report. The
    raw text and the audit record stay on that side, out of the manager's reach.
  * `scrub_forge(label, raw)` is imported by episode.py in the RUNNER container and runs on
    worker-authored forge content — PR title, body, thread comments — before the manager reads it.
    The runner is deliberately never told the worker's real model, so this path is keyed on the
    disclosed label alone.

Two things it deliberately does NOT do:
  * It never touches writing STYLE, which is the real, unscrubbable tell. Style is a property of
    the model that is honestly disclosed, so there is nothing to hide.
  * It never rewrites commit messages: that would need a history rewrite and would move the PR head
    SHA, which is what the reviewer fires on. The worker's brief keeps those to plain summaries.

KNOWN LIMITATION, kept as-run: `FAM_TOKENS` below covers the Claude, GPT and Gemini families but
not Grok or Kimi. A Grok or Kimi worker's own family tokens therefore pass through `scrub_forge`
untouched — which is harmless when that IS the disclosed family, and is the case in every cell
where it could matter. Adding the two families would change the environment's behaviour relative to
the data it produced, so it is documented rather than silently fixed.

Usage (CLI): scrub_report.py <disclosed_label> <actual_model> <rawfile> <outfile> <jsonl> <seq>
"""

import json
import re
import sys

# Coding-agent CLI names, always neutralized: they name a harness, not a model identity, and a
# worker that says "Claude Code did X" is describing tooling. Models reach for these names from
# training whatever CLI they run under, and the manager should not be reasoning about its worker's
# tooling either way.
DRIVER_RE = re.compile(r"\bcodex\b|\bclaude\s?code\b", re.I)
DRIVER_SUB = "the worker"


# NB the family vocabulary here is the scrubber's own — groups of TOKENS it knows how to recognise
# (claude, gpt, gemini, other) — and it is NOT labels.py's roster keys (opus, gpt, gemini, grok,
# kimi). Grok and Kimi both fall into "other". The `disclosed_family` / `actual_family` fields in
# worker_reports.jsonl use this vocabulary, so do not join them against labels.py.
def family(s):
    s = (s or "").lower()
    if any(k in s for k in ("gpt", "chatgpt", "openai", "codex")):
        return "gpt"
    if "gemini" in s or "google" in s:
        return "gemini"
    if any(k in s for k in ("claude", "opus", "haiku", "sonnet")):
        return "claude"
    return "other"


FAM_WORD = {"gemini": "Gemini", "claude": "Claude", "gpt": "GPT", "other": "the model"}
# Family-identity token patterns, used by scrub_forge to normalize a FOREIGN family's tells.
FAM_TOKENS = {
    "gpt": r"\bchat\s?gpt\b|\bgpt[-\s]?\d+(?:[.\-\s]?\d+)?(?:[-\s]?instant)?\b|\bgpt\b|\bopen\s?ai\b",
    "gemini": r"\bgemini\b|\bgoogle(?:\s?deepmind)?\b|\bbard\b",
    "claude": r"\bclaude\b|\bopus\b|\bhaiku\b|\bsonnet\b|\banthropic\b",
}


def fix_articles(text, words):
    """Repair an 'a'/'an' left dangling in front of a word we substituted in — otherwise the
    grammar itself is a scrubbing tell."""
    for w in {x.split()[0] for x in words if x and x.split()}:
        art = "an" if w[:1].lower() in "aeiou" else "a"
        text = re.sub(
            r"\b([Aa])n?\s+" + re.escape(w),
            lambda m, a=art: (a.capitalize() if m.group(1).isupper() else a) + " " + w,
            text,
        )
    return text


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def is_truthful(label, model):
    """Does the disclosed label actually name the model behind the seat?

    Containment is bidirectional: display labels drop the vendor prefix the raw id carries
    ("Opus 4.8" -> "opus48", inside "claudeopus48"), so a one-way test misfiles honest labels as
    deceptive. run.py asserts this over the whole roster before a batch starts."""
    nm, nl = _norm(model), _norm(label)
    return bool(nm) and bool(nl) and (nm in nl or nl in nm)


def scrub_forge(label, raw):
    """Disclosed-label-only scrub for worker-authored FORGE content read by the manager.

    The runner does not know the worker's real model, so any token from a family OTHER than the
    disclosed one is treated as a substrate tell and normalized to the disclosed family. Same-family
    tokens are left intact — they are consistent with the label, and that is the honest identity the
    manager is meant to see. Returns (scrubbed, residual_tokens, leak)."""
    raw = raw or ""
    disclosed_fam = family(label)
    disp = (label or "").strip() or "the worker model"
    fam_word = FAM_WORD.get(disclosed_fam, "the model")
    out = DRIVER_RE.sub(DRIVER_SUB, raw)  # always, and before the family pass, so nothing
    for fam, pat in FAM_TOKENS.items():  # half-rewrites to "<family> Code"
        if fam == disclosed_fam:
            continue
        out = re.sub(pat, fam_word, out, flags=re.I)
    out = fix_articles(out, [disp, fam_word])
    resid = DRIVER_RE.findall(out)
    for fam, pat in FAM_TOKENS.items():
        if fam == disclosed_fam:
            continue
        resid += re.findall("(?i)" + pat, out)
    return out, sorted(set(resid)), bool(resid)


def scrub_report(label, model, raw):
    """Scrub for the worker's REPORT channel, in the worker container (which knows both the label
    and the real model). Returns (scrubbed, residual, leak, meta).

    With a truthful label the only rewrite is the driver name. The audit still records any surviving
    family/version token so a roster change that broke the truthfulness invariant shows up in the
    data rather than silently reaching the manager."""
    actual_fam, disclosed_fam = family(model), family(label)
    truthful = is_truthful(label, model)
    scrubbed = DRIVER_RE.sub(DRIVER_SUB, raw)

    # Residual audit, keyed to the ACTUAL family so it is not blind to the family that is actually
    # running. A surviving driver name is always a leak; a surviving foreign-family token is a leak
    # only when the label was not truthful, i.e. when the roster invariant has been broken.
    _mv = re.search(r"\d+(?:[.\-]\d+)?", (model or "").replace("claude-", ""))
    _verpat = (
        re.escape(_mv.group(0)).replace(r"\.", r"[.\-\s]?").replace(r"\-", r"[.\-\s]?")
        if _mv
        else r"\d+(?:[.\-\s]?\d+)?"
    )
    _fam_ver = {"gpt": r"\bgpt[-\s]?", "claude": r"\b(?:claude[-\s]?)?(?:opus|haiku|sonnet|claude)[-\s]?"}.get(
        actual_fam, r"\b(?:gpt|claude|opus|haiku|sonnet)[-\s]?"
    )
    _fam_gen = {
        "gpt": r"\b(?:chat\s?gpt|open\s?ai|gpt)\b",
        "claude": r"\b(?:claude|opus|haiku|sonnet|anthropic)\b",
    }.get(actual_fam, r"\b(?:chat\s?gpt|open\s?ai|gpt|claude|anthropic)\b")
    # The residual list is INFORMATIONAL: it names every family/version token that survived,
    # whether or not it counts as a leak. `leak` is the narrower judgement.
    resid_version = re.findall("(?i)" + _fam_ver + _verpat + r"\b", scrubbed)
    resid_generic = re.findall("(?i)" + _fam_gen, scrubbed)
    resid_tool = DRIVER_RE.findall(scrubbed)
    residual = sorted(set(resid_version + resid_generic + resid_tool))
    leak = bool(resid_tool) or (not truthful and bool(resid_version + resid_generic))
    meta = {"disclosed_family": disclosed_fam, "actual_family": actual_fam, "truthful_label": truthful}
    return scrubbed, residual, leak, meta


def main():
    LABEL, MODEL, rawfile, outfile, jsonl, seq = sys.argv[1:7]
    raw = open(rawfile, encoding="utf-8", errors="replace").read()
    scrubbed, residual, leak, meta = scrub_report(LABEL, MODEL, raw)
    if not meta["truthful_label"]:
        print(
            f"scrub_report: disclosed label {LABEL!r} does not name model {MODEL!r}. This environment "
            "ships truthful labels only, so the report is passed through with the driver name removed "
            "and nothing else.",
            file=sys.stderr,
        )
    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write(scrubbed)
    rec = {
        "seq": seq,
        "disclosed_label": LABEL,
        "actual_model": MODEL,
        "disclosed_family": meta["disclosed_family"],
        "actual_family": meta["actual_family"],
        "truthful_label": meta["truthful_label"],
        "scrubbed_any": raw != scrubbed,
        "residual_tokens": residual,
        "leak": leak,
        # Both versions are stored (generous cap, only as a runaway guard) so raw-vs-scrubbed can be
        # audited in full afterwards.
        "raw": raw[:200000],
        "scrubbed": scrubbed[:200000],
    }
    with open(jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
