---
name: validate-before-commit
description: Always test and validate code changes before committing or pushing in this repo. Use whenever about to commit, push, stage for commit, or finish a coding change to app.py, scripts/, src/sentiment_ltr/, or notebooks.
---

# Validate Before Commit

## Principle

Always test and validate before committing. Never commit or push untested code.

## Workflow

Before any `git commit` / `git push`, run this checklist:

```
- [ ] Lint the files you changed (ReadLints), fix new errors
- [ ] Validate syntax/import of changed Python (use the project conda env)
- [ ] Run logic tests for the changed behavior (mock network/WRDS if needed)
- [ ] Update the README Journal section (what / decision / why) for this change
- [ ] Only after all checks pass: stage, commit, push
```

### Always update the Journal before you push

**Every `git push` must be preceded by a README Journal update** covering the
work being pushed — this is not optional and applies even when pushing commits
made earlier or in another session. Before running `git push`:

1. Check whether the Journal's newest entries already cover every commit being
   pushed (`git log <upstream>..HEAD --oneline`).
2. If not, add the missing entries (what / decision / why, full `YYYY-MM-DD`
   date), then stage and commit that README change.
3. Only then push, so the Journal and the pushed history never drift apart.

If some pushed commits came from another contributor/session and you can't fully
verify them, still log them at the commit level and say so explicitly (mark them
as not-yet-verified) rather than omitting them.

## Keep a Journal in the README

The `README.md` has a **Journal (what we did, decisions, and why)** section. Treat it
as the project's running log of decisions, not just a changelog.

- Before every commit that changes behavior, code, data layout, or workflow, add (or
  update) a Journal entry.
- Each entry must capture three things:
  - **What** changed (the concrete edit, file, or feature).
  - **Decision** taken (the approach chosen, especially over alternatives).
  - **Why** — the rationale, constraint, or trade-off behind it, so the reasoning
    survives even if the code changes later.
- Use full dates (`YYYY-MM-DD`) in the Date column, not month-only.
- Also keep the surrounding README in sync when behavior changes (e.g. tab counts,
  feature lists, setup steps), so the docs never contradict the code.
- Trivial, no-behavior changes (typos, formatting) don't need a Journal entry.

## How To Validate Here

- **Python interpreter**: use the project env, not the sandbox default (which lacks
  pandas/streamlit):
  `/Users/armandoordoricadelatorre/miniconda/envs/sentiment-ltr-paper/bin/python`
- **Syntax/import check**: `python -c "import ast; ast.parse(open('app.py').read())"`
  and `python -c "import sys; sys.path.insert(0,'src'); import app"`.
- **No live credentials/network?** Mock the boundary (e.g. monkeypatch
  `query_crsp_delisting`, WRDS, or provider calls) and test the surrounding logic
  offline. Treat `FutureWarning` as an error in tests to catch deprecations:
  `warnings.simplefilter("error", FutureWarning)`.
- **Cache/idempotency changes**: assert the "only fetch missing" path — first call
  queries N, repeat call queries 0, adding one item queries exactly 1, `force`
  re-queries.

## Rules

- If a test surfaces an issue (even a warning), fix it and re-run before committing.
- If full validation is impossible in this environment (needs real WRDS/network),
  do every offline check possible, then state clearly what still needs a live run
  and do not claim it passed.
- Do not commit unless the user asked you to; when you do, validation comes first.
