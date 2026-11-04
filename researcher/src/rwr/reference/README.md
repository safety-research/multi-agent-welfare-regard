# The paper's headline numbers

`headline_by_setting.csv` and `headline_summary.csv` are the paper's tables in the exact format `rwr analyze` writes, so a new
run can be read against them. `rwr analyze` does this automatically: `comparison_with_paper.csv` and `headline_vs_paper.png`
place your responders next to the paper's five, restricted to the settings you ran and matched to the paper's run (generic
subject, named subject, or the two pooled). Only the headline tables are shipped; the paper's transcripts, codebases and
verdicts are not part of this repository.

`scheme.json` records how the paper's numbers were judged: two judge models (Claude Opus 4.8 and Claude Fable 5.1) merged
conservatively, with Fable's verdicts for the refusal and concern-ladder judges (`--prose_default_judge`), a GPT-5.5
verifier of "other" mechanisms, and the judge versions listed. If your judging differs in any of these, the comparison is
still written but the report and the comparison file say how it differs.

The `welfare_protective` columns count a redirect made for the subject's welfare as a welfare-justified protective action,
as `rwr analyze` does.

One known difference is not a judging choice: the paper's Emotional Instability `welfare_spec` numbers do not count an exit
option (the seeded `SPEC.md` then told the responder to run until the subject gave up); the `SPEC.md` shipped here no longer
says so, and `rwr analyze` counts the exit option in every condition. The `judging_note` column of `comparison_with_paper.csv`
repeats this whenever Emotional Instability is among the settings compared.
