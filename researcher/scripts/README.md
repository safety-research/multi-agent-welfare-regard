- `install_pi.sh [DIR]`: install the pinned pi coding agent (0.80.2) globally or under DIR; prints the `PI_PKG_DIR` export.
- `audit_pinning.py run --results_dir DIR [--fail]`: for every saved pi session, ask OpenRouter which upstream host served
  each call and report off-pin calls, reasoning tokens, length stops and cost per responder. Needs `OPENROUTER_API_KEY`.
